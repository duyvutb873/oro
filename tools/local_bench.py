#!/usr/bin/env python3
"""Repeat local generated-environment runs and compare agents.

One ``docker compose run test`` call scores one agent once. LLM agents and
the shopper simulator are not deterministic, so one run says little about
whether a change helped. This tool runs several agents several times on the
same task selection, then reads each run's ``summary.json``,
``episode_results.jsonl`` and ``sandbox/sandbox_output.jsonl`` to report:

* aggregate score per agent (mean, spread, min, max) and per family;
* per-task rewards side by side, with the delta between agents and flaky tasks;
* why tasks failed (reward reason, failed verifier checks, terminal reason);
* cost and latency (sandbox time, inference USD and tokens, tool calls).

Commands::

    # 3 rounds of the full 30-task pack for two agents, interleaved per round
    python3 tools/local_bench.py run --name baseline \\
        --agent ref=src/agent/environment_agent.py \\
        --agent mine=submit/my_agent.py --repeats 3

    # report on a bench (or on existing run dirs with --run label=DIR)
    python3 tools/local_bench.py report logs/bench/baseline

    # list failed episodes with the evidence needed to triage them
    python3 tools/local_bench.py failures logs/bench/baseline --label mine

Only the standard library is used, so it runs with the host's ``python3``.
Run artifacts contain sealed task data: keep them local and never commit them.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "logs" / "environment-runs"
BENCH_ROOT = REPO_ROOT / "logs" / "bench"
INDEX_NAME = "bench.json"
INDEX_SCHEMA = "oro.local_bench.v1"


# --------------------------------------------------------------------------
# Loading run artifacts
# --------------------------------------------------------------------------


@dataclass
class Episode:
    """One finalized task of one run, joined with its sandbox row."""

    label: str
    run_id: str
    task_id: str
    family: str
    outcome: str
    correct: bool
    reward: float
    error_classification: str | None = None
    error_detail: str | None = None
    terminal_reason: str | None = None
    reward_reason: str | None = None
    failed_checks: list[str] = field(default_factory=list)
    false_metrics: list[str] = field(default_factory=list)
    explanation: str | None = None
    efficiency: float | None = None
    tool_calls: int | None = None
    env_cost: float | None = None
    steps: int | None = None
    query: str | None = None
    user_messages: list[str] = field(default_factory=list)
    agent_messages: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    family_detail: dict[str, Any] = field(default_factory=dict)
    sandbox_status: str | None = None
    sandbox_error: str | None = None
    execution_time: float | None = None
    inference_cost_usd: float | None = None
    inference_total: int | None = None
    inference_failed: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class Run:
    label: str
    path: Path
    summary: dict[str, Any]
    episodes: list[Episode]

    @property
    def run_id(self) -> str:
        return str(self.summary.get("run_id") or self.path.name)

    @property
    def aggregate(self) -> float | None:
        value = self.summary.get("aggregate_score")
        if value is None and self.episodes:
            value = sum(e.reward for e in self.episodes) / len(self.episodes)
        return None if value is None else float(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _short(text: Any, limit: int = 240) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# Family metric details that explain a miss in a single line.
_DETAIL_KEYS = (
    "chosen_key",
    "chosen_level",
    "best_level",
    "event_target_key",
    "full_match_key",
    "pre_event_top_key",
    "post_event_top_key",
    "best_frontier_score",
    "chosen_priority_score",
)


def _family_detail(metrics: dict[str, Any]) -> dict[str, Any]:
    detail: dict[str, Any] = {}
    sources = [metrics] + [
        value
        for key, value in metrics.items()
        if key.endswith("_detail") and isinstance(value, dict)
    ]
    for source in sources:
        for key in _DETAIL_KEYS:
            if key in source and source[key] is not None and key not in detail:
                detail[key] = source[key]
    return detail


def _episode_from_artifact(
    label: str, run_id: str, artifact: dict[str, Any], sandbox: dict[str, dict]
) -> Episode:
    episode = artifact.get("episode", artifact)
    verdict = episode.get("verdict") or {}
    record = verdict.get("reward_record") or {}
    checks = verdict.get("checks") or {}
    metrics = verdict.get("family_metrics") or {}
    detail = verdict.get("efficiency_detail") or {}
    correct = bool(verdict.get("correct"))
    session_id = str(episode.get("session_id") or "")

    user_messages: list[str] = []
    agent_messages: list[str] = []
    events: list[str] = []
    for entry in episode.get("ledger") or []:
        payload = entry.get("payload") or {}
        if entry.get("kind") == "user_message" and payload.get("content"):
            user_messages.append(str(payload["content"]))
        elif entry.get("kind") == "model_message":
            content = payload.get("content") or payload.get("text")
            if content:
                agent_messages.append(str(content))
        elif entry.get("kind") == "harness_event" and payload.get("kind"):
            target = payload.get("target") or {}
            key = "::".join(
                str(target[k]) for k in ("product_id", "sku") if target.get(k)
            )
            change = ""
            if (
                payload.get("old_price") is not None
                or payload.get("new_price") is not None
            ):
                change = (
                    f" price {payload.get('old_price')}->{payload.get('new_price')}"
                )
            events.append(
                f"turn {entry.get('turn')}: {payload['kind']} {key}{change}".strip()
            )

    row = sandbox.get(session_id, {})
    usage = row.get("_shadow_inference_usage") or {}
    error = row.get("error") or {}
    bootstrap = episode.get("bootstrap") or {}

    return Episode(
        label=label,
        run_id=run_id,
        task_id=str(episode.get("task_id") or ""),
        family=str(episode.get("family") or "unknown"),
        outcome=str(episode.get("outcome") or ""),
        correct=correct,
        reward=float(verdict.get("paid_reward") or 0) if correct else 0.0,
        error_detail=episode.get("error_detail"),
        terminal_reason=episode.get("terminal_reason"),
        reward_reason=record.get("reason"),
        failed_checks=sorted(k for k, v in checks.items() if v is False),
        false_metrics=sorted(
            k for k, v in metrics.items() if v is False and "exploit" not in k
        ),
        explanation=verdict.get("explanation"),
        efficiency=_number(verdict.get("efficiency")),
        tool_calls=detail.get("tool_calls"),
        env_cost=_number(detail.get("cost")),
        steps=episode.get("solver_turn_count") or episode.get("step_count"),
        query=(bootstrap.get("policy_view") or {}).get("query"),
        user_messages=user_messages,
        agent_messages=agent_messages,
        events=events,
        family_detail=_family_detail(metrics),
        sandbox_status=row.get("status"),
        sandbox_error=error.get("message") if isinstance(error, dict) else None,
        execution_time=_number(row.get("execution_time")),
        inference_cost_usd=_number(usage.get("inference_cost_usd")),
        inference_total=usage.get("inference_total", row.get("inference_total")),
        inference_failed=usage.get(
            "inference_failed", row.get("inference_failure_count")
        ),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


def load_run(path: Path, label: str) -> Run:
    """Load one ``logs/environment-runs/<run>`` directory."""

    summary_path = path / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        summary = {}
    run_id = str(summary.get("run_id") or path.name)
    sandbox = {
        str(row.get("problem_id")): row
        for row in _read_jsonl(path / "sandbox" / "sandbox_output.jsonl")
    }
    episodes = [
        _episode_from_artifact(label, run_id, artifact, sandbox)
        for artifact in _read_jsonl(path / "episode_results.jsonl")
    ]
    # Tasks the summary lists but no receipt finalized still count as zero.
    seen = {e.task_id for e in episodes}
    for task in summary.get("tasks") or []:
        task_id = str(task.get("task_id") or "")
        if task_id and task_id not in seen:
            episodes.append(
                Episode(
                    label=label,
                    run_id=run_id,
                    task_id=task_id,
                    family=str(task.get("family") or "unknown"),
                    outcome=str(task.get("outcome") or "missing"),
                    correct=bool(task.get("correct")),
                    reward=float(task.get("reward") or 0),
                    error_classification=task.get("error_classification"),
                    error_detail=task.get("error_detail"),
                )
            )
    by_task = {str(t.get("task_id")): t for t in summary.get("tasks") or []}
    for episode in episodes:
        task = by_task.get(episode.task_id)
        if task is not None:
            episode.error_classification = task.get("error_classification")
    return Run(label=label, path=path, summary=summary, episodes=episodes)


# --------------------------------------------------------------------------
# Bench index
# --------------------------------------------------------------------------


def _read_index(bench_dir: Path) -> dict[str, Any]:
    path = bench_dir / INDEX_NAME
    if not path.is_file():
        raise SystemExit(f"{path} not found; is this a bench directory?")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_index(bench_dir: Path, index: dict[str, Any]) -> None:
    path = bench_dir / INDEX_NAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_bench(bench_dir: Path) -> list[Run]:
    index = _read_index(bench_dir)
    runs = []
    for entry in index.get("runs", []):
        if not entry.get("run_dir"):
            continue
        run_dir = Path(entry["run_dir"])
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        if run_dir.is_dir():
            runs.append(load_run(run_dir, entry["label"]))
    return runs


def _parse_labelled(values: Iterable[str], what: str) -> list[tuple[str, str]]:
    pairs = []
    for value in values:
        label, sep, rest = value.partition("=")
        if not sep or not label or not rest:
            raise SystemExit(f"--{what} expects label=path, got {value!r}")
        pairs.append((label, rest))
    labels = [label for label, _ in pairs]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"--{what} labels must be unique")
    return pairs


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def _run_dirs() -> set[str]:
    if not RUNS_ROOT.is_dir():
        return set()
    return {p.name for p in RUNS_ROOT.iterdir() if p.is_dir()}


def _agent_path(path: str) -> str:
    resolved = (Path.cwd() / path).resolve()
    if not resolved.is_file():
        raise SystemExit(f"agent file not found: {path}")
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        raise SystemExit(
            f"agent file must be inside the repository (it is mounted at "
            f"/workspace): {path}"
        ) from None
    return relative.as_posix()


def run_once(
    agent_file: str,
    *,
    problems: int | None,
    seed: int | None,
    log_path: Path,
    compose: list[str],
) -> tuple[int, str | None, float]:
    """Run one local test and return (exit code, run dir, seconds)."""

    command = [*compose, "run", "--rm", "test", "--agent-file", agent_file]
    if problems is not None:
        command += ["--problems", str(problems)]
        if seed is not None:
            command += ["--seed", str(seed)]
    before = _run_dirs()
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = process.wait()
    elapsed = time.monotonic() - started
    created = sorted(_run_dirs() - before)
    run_dir = None
    if created:
        # A Compose project runs one local test at a time, so this is ours.
        run_dir = (RUNS_ROOT / created[-1]).relative_to(REPO_ROOT).as_posix()
    return code, run_dir, elapsed


def command_run(args: argparse.Namespace) -> int:
    agents = [
        (label, _agent_path(path))
        for label, path in _parse_labelled(args.agent, "agent")
    ]
    name = args.name or time.strftime("bench-%Y%m%d-%H%M%S")
    bench_dir = BENCH_ROOT / name
    if (bench_dir / INDEX_NAME).exists() and not args.resume:
        raise SystemExit(f"{bench_dir} exists; pass --resume to add rounds to it")
    (bench_dir / "logs").mkdir(parents=True, exist_ok=True)
    index = (
        _read_index(bench_dir)
        if (bench_dir / INDEX_NAME).exists()
        else {"schema_version": INDEX_SCHEMA, "name": name, "runs": []}
    )
    index["agents"] = {label: path for label, path in agents}
    compose = args.compose.split()
    rng = random.Random(args.seed)
    first_round = 1 + max((r.get("round", 0) for r in index["runs"]), default=0)

    for round_no in range(first_round, first_round + args.repeats):
        # Every agent in a round sees the same sampled tasks.
        seed = None
        if args.problems is not None:
            seed = (
                args.seed
                if args.seed is not None and args.repeats == 1
                else rng.randrange(1, 2**31)
            )
        order = list(agents)
        if args.shuffle:
            rng.shuffle(order)
        for label, agent_file in order:
            print(f"\n=== round {round_no}  agent {label}  ({agent_file})", flush=True)
            log_path = bench_dir / "logs" / f"r{round_no:02d}-{label}.log"
            code, run_dir, elapsed = run_once(
                agent_file,
                problems=args.problems,
                seed=seed,
                log_path=log_path,
                compose=compose,
            )
            index["runs"].append(
                {
                    "round": round_no,
                    "label": label,
                    "agent_file": agent_file,
                    "problems": args.problems,
                    "seed": seed,
                    "exit_code": code,
                    "run_dir": run_dir,
                    "seconds": round(elapsed, 1),
                    "log": log_path.relative_to(REPO_ROOT).as_posix(),
                }
            )
            _write_index(bench_dir, index)
            if run_dir is None:
                print(f"!!! no run directory was created; see {log_path}", flush=True)
            elif code != 0:
                print(
                    f"!!! exit code {code}; partial results kept in {run_dir}",
                    flush=True,
                )
            if code != 0 and args.stop_on_error:
                return code

    runs = load_bench(bench_dir)
    report = build_report(runs)
    text = render_text(report)
    print("\n" + text)
    _save_report(bench_dir, report, text)
    return 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "stdev": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _mean(values: Iterable[float | int | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return statistics.fmean(present) if present else None


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]


def build_report(runs: list[Run]) -> dict[str, Any]:
    labels: list[str] = []
    for run in runs:
        if run.label not in labels:
            labels.append(run.label)

    agents: dict[str, Any] = {}
    task_rewards: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    task_family: dict[str, str] = {}

    for label in labels:
        mine = [r for r in runs if r.label == label]
        episodes = [e for r in mine for e in r.episodes]
        families: dict[str, list[Episode]] = defaultdict(list)
        for episode in episodes:
            families[episode.family].append(episode)
            task_rewards[episode.task_id][label].append(episode.reward)
            task_family[episode.task_id] = episode.family

        failures = [e for e in episodes if not e.correct]
        times = [e.execution_time for e in episodes if e.execution_time is not None]
        costs = [
            e.inference_cost_usd for e in episodes if e.inference_cost_usd is not None
        ]
        agents[label] = {
            "runs": [
                {"run_id": r.run_id, "path": str(r.path), "aggregate": r.aggregate}
                for r in mine
            ],
            "aggregate": _stats([r.aggregate for r in mine if r.aggregate is not None]),
            "episodes": len(episodes),
            "passed": sum(e.correct for e in episodes),
            "families": {
                family: {
                    "n": len(items),
                    "mean_reward": _mean(e.reward for e in items),
                    "pass_rate": sum(e.correct for e in items) / len(items),
                }
                for family, items in sorted(families.items())
            },
            "outcomes": dict(Counter(e.outcome for e in episodes).most_common()),
            "failure_reasons": dict(
                Counter(e.reward_reason or e.outcome for e in failures).most_common()
            ),
            "failed_checks": dict(
                Counter(c for e in failures for c in e.failed_checks).most_common()
            ),
            "terminal_reasons": dict(
                Counter(str(e.terminal_reason) for e in episodes).most_common()
            ),
            "failure_metrics_by_family": {
                family: dict(
                    Counter(
                        m for e in items if not e.correct for m in e.false_metrics
                    ).most_common(6)
                )
                for family, items in sorted(families.items())
            },
            "sandbox_status": dict(
                Counter(str(e.sandbox_status) for e in episodes).most_common()
            ),
            "cost": {
                "seconds_mean": _mean(times),
                "seconds_p90": _p90(times),
                "seconds_max": max(times) if times else None,
                "usd_per_task": _mean(costs),
                "usd_total": sum(costs) if costs else None,
                "llm_calls_per_task": _mean(e.inference_total for e in episodes),
                "llm_failed_total": sum(e.inference_failed or 0 for e in episodes),
                "prompt_tokens_per_task": _mean(e.prompt_tokens for e in episodes),
                "completion_tokens_per_task": _mean(
                    e.completion_tokens for e in episodes
                ),
                "tool_calls_per_task": _mean(e.tool_calls for e in episodes),
                "steps_per_task": _mean(e.steps for e in episodes),
                "efficiency_mean": _mean(e.efficiency for e in episodes),
            },
        }

    tasks = []
    for task_id in sorted(task_rewards, key=lambda t: (task_family[t], t)):
        per_label = task_rewards[task_id]
        row = {
            "task_id": task_id,
            "family": task_family[task_id],
            "rewards": {label: per_label.get(label, []) for label in labels},
            "mean": {label: _mean(per_label.get(label, [])) for label in labels},
        }
        row["flaky"] = sorted(
            label
            for label in labels
            if len({r > 0 for r in per_label.get(label, [])}) > 1
        )
        if len(labels) >= 2:
            base, other = row["mean"][labels[0]], row["mean"][labels[-1]]
            row["delta"] = None if base is None or other is None else other - base
        tasks.append(row)

    return {"labels": labels, "agents": agents, "tasks": tasks}


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _counter_line(counts: dict[str, int], limit: int = 6) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{k} {v}" for k, v in list(counts.items())[:limit])


def render_text(report: dict[str, Any]) -> str:
    labels = report["labels"]
    agents = report["agents"]
    out: list[str] = []
    if not labels:
        return "No runs found."

    width = max(12, *(len(label) for label in labels))
    out.append("AGGREGATE (mean of run aggregates)")
    for label in labels:
        agg = agents[label]["aggregate"]
        runs = ", ".join(_fmt(r["aggregate"], 3) for r in agents[label]["runs"])
        out.append(
            f"  {label:<{width}}  mean {_fmt(agg['mean'], 3)}  ±{_fmt(agg['stdev'], 3)}"
            f"  min {_fmt(agg['min'], 3)}  max {_fmt(agg['max'], 3)}"
            f"  passed {agents[label]['passed']}/{agents[label]['episodes']}  runs [{runs}]"
        )

    families = sorted({f for label in labels for f in agents[label]["families"]})
    out.append("")
    out.append("FAMILIES (mean reward / pass rate)")
    out.append(
        "  " + "family".ljust(26) + "".join(label[:16].rjust(18) for label in labels)
    )
    for family in families:
        cells = []
        for label in labels:
            data = agents[label]["families"].get(family)
            cells.append(
                "-".rjust(18)
                if data is None
                else f"{data['mean_reward']:.2f} / {data['pass_rate']:.0%}".rjust(18)
            )
        out.append("  " + family[:26].ljust(26) + "".join(cells))

    out.append("")
    out.append("WHY TASKS FAILED")
    for label in labels:
        data = agents[label]
        out.append(f"  {label}")
        out.append(f"    reward reason    {_counter_line(data['failure_reasons'])}")
        out.append(f"    failed checks    {_counter_line(data['failed_checks'])}")
        out.append(f"    terminal reason  {_counter_line(data['terminal_reasons'])}")
        out.append(f"    sandbox status   {_counter_line(data['sandbox_status'])}")
        for family, metrics in data["failure_metrics_by_family"].items():
            if metrics:
                out.append(f"    {family[:22]:<22} false: {_counter_line(metrics, 4)}")

    out.append("")
    out.append("COST AND LATENCY (per task)")
    keys = [
        ("seconds_mean", "sandbox s mean", 1),
        ("seconds_p90", "sandbox s p90", 1),
        ("seconds_max", "sandbox s max", 1),
        ("usd_per_task", "inference USD", 4),
        ("llm_calls_per_task", "LLM calls", 1),
        ("llm_failed_total", "LLM failed (total)", 0),
        ("prompt_tokens_per_task", "prompt tokens", 0),
        ("completion_tokens_per_task", "completion tokens", 0),
        ("tool_calls_per_task", "env tool calls", 1),
        ("steps_per_task", "env turns", 1),
        ("efficiency_mean", "verifier efficiency", 2),
    ]
    out.append(
        "  " + "metric".ljust(22) + "".join(label[:16].rjust(18) for label in labels)
    )
    for key, name, digits in keys:
        out.append(
            "  "
            + name.ljust(22)
            + "".join(
                _fmt(agents[label]["cost"][key], digits).rjust(18) for label in labels
            )
        )

    out.append("")
    header = "TASKS (mean reward over runs"
    header += f"; delta = {labels[-1]} - {labels[0]})" if len(labels) >= 2 else ")"
    out.append(header)
    tasks = report["tasks"]
    if len(labels) >= 2:
        tasks = sorted(
            tasks, key=lambda t: (t.get("delta") is None, t.get("delta") or 0)
        )
    for task in tasks:
        cells = "".join(_fmt(task["mean"][label]).rjust(8) for label in labels)
        delta = ""
        if "delta" in task and task["delta"] is not None and abs(task["delta"]) > 1e-9:
            delta = f"  {task['delta']:+.2f}"
        flaky = f"  flaky: {','.join(task['flaky'])}" if task["flaky"] else ""
        out.append(f"  {task['task_id'][:44]:<44}{cells}{delta}{flaky}")
    return "\n".join(out)


def _save_report(directory: Path, report: dict[str, Any], text: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\nReport written to {directory / 'report.txt'} and report.json")


def _collect_runs(args: argparse.Namespace) -> tuple[list[Run], Path | None]:
    runs: list[Run] = []
    bench_dir = None
    if args.bench:
        bench_dir = Path(args.bench).resolve()
        runs.extend(load_bench(bench_dir))
    for label, path in _parse_labelled(args.run or [], "run"):
        run_dir = Path(path).resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"run directory not found: {path}")
        runs.append(load_run(run_dir, label))
    if not runs:
        raise SystemExit("nothing to report: pass a bench directory or --run label=DIR")
    return runs, bench_dir


def command_report(args: argparse.Namespace) -> int:
    runs, bench_dir = _collect_runs(args)
    report = build_report(runs)
    text = render_text(report)
    print(text)
    out_dir = Path(args.out).resolve() if args.out else bench_dir
    if out_dir is not None:
        _save_report(out_dir, report, text)
    return 0


# --------------------------------------------------------------------------
# failures
# --------------------------------------------------------------------------


def render_failures(episodes: list[Episode], *, full: bool = False) -> str:
    out: list[str] = []
    limit = 10_000 if full else 240
    for e in episodes:
        out.append(f"── {e.task_id}  [{e.family}]  agent {e.label}  run {e.run_id}")
        out.append(
            f"   outcome {e.outcome}  terminal {e.terminal_reason}  "
            f"reward reason {e.reward_reason}  turns {_fmt(e.steps)}  "
            f"tool calls {_fmt(e.tool_calls)}  sandbox {_fmt(e.execution_time, 0)}s"
        )
        if e.query:
            out.append(f"   request   {_short(e.query, limit)}")
        for event in e.events:
            out.append(f"   event     {event}")
        for message in e.user_messages:
            if message.strip():
                out.append(f"   shopper   {_short(message, limit)}")
        if full:
            for message in e.agent_messages:
                out.append(f"   agent     {_short(message, limit)}")
        if e.explanation:
            out.append(f"   verdict   {_short(e.explanation, limit)}")
        if e.failed_checks:
            out.append(f"   failed    {', '.join(e.failed_checks)}")
        if e.false_metrics:
            out.append(f"   false     {', '.join(e.false_metrics)}")
        if e.family_detail:
            detail = ", ".join(f"{k}={v}" for k, v in e.family_detail.items())
            out.append(f"   detail    {_short(detail, limit)}")
        if e.error_detail:
            out.append(f"   error     {_short(e.error_detail, limit)}")
        if e.sandbox_error:
            out.append(f"   agent err {_short(e.sandbox_error, limit)}")
        out.append("")
    return "\n".join(out) if out else "No failed episodes."


def command_failures(args: argparse.Namespace) -> int:
    runs, _ = _collect_runs(args)
    episodes = [
        e
        for run in runs
        for e in run.episodes
        if not e.correct
        and (not args.label or e.label in args.label)
        and (not args.family or e.family in args.family)
        and (not args.task or any(t in e.task_id for t in args.task))
    ]
    episodes.sort(key=lambda e: (e.family, e.task_id, e.label, e.run_id))
    print(render_failures(episodes, full=args.full))
    print(f"{len(episodes)} failed episode(s)")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repeat local ORO runs, compare agents, and triage failures.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run agents repeatedly and report")
    run.add_argument(
        "--agent",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="agent file inside the repository; repeat for each agent",
    )
    run.add_argument(
        "--repeats", type=int, default=3, help="rounds per agent (default 3)"
    )
    run.add_argument(
        "--problems",
        type=int,
        help="sample this many tasks per round (default: the full pack)",
    )
    run.add_argument(
        "--seed",
        type=int,
        help="seed for the per-round task samples (and agent order with --shuffle)",
    )
    run.add_argument("--name", help="bench name under logs/bench/ (default: timestamp)")
    run.add_argument(
        "--resume", action="store_true", help="add rounds to an existing bench"
    )
    run.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle agent order inside each round",
    )
    run.add_argument(
        "--stop-on-error", action="store_true", help="stop at the first failed run"
    )
    run.add_argument(
        "--compose",
        default=os.environ.get("ORO_BENCH_COMPOSE", "docker compose"),
        help="compose command (default: 'docker compose')",
    )
    run.set_defaults(func=command_run)

    for name, func, help_text in (
        ("report", command_report, "summarize a bench or run directories"),
        ("failures", command_failures, "list failed episodes with their evidence"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("bench", nargs="?", help="bench directory (logs/bench/<name>)")
        cmd.add_argument(
            "--run",
            action="append",
            metavar="LABEL=DIR",
            help="add an existing logs/environment-runs/<run> directory",
        )
        cmd.set_defaults(func=func)
        if name == "report":
            cmd.add_argument("--out", help="directory for report.txt and report.json")
        else:
            cmd.add_argument("--label", action="append", help="only this agent label")
            cmd.add_argument("--family", action="append", help="only this family")
            cmd.add_argument("--task", action="append", help="task id substring")
            cmd.add_argument("--full", action="store_true", help="do not shorten text")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
