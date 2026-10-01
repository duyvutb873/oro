"""Tests for tools/local_bench.py, built on the viewer's episode fixtures."""

import copy
import json
from pathlib import Path

import pytest

from tools import local_bench

FIXTURES = (
    Path(__file__).resolve().parent.parent
    / "trajectory-viewer"
    / "tests"
    / "fixtures"
    / "oro-episodes"
)


def _artifact(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _passing(artifact: dict) -> dict:
    artifact = copy.deepcopy(artifact)
    verdict = artifact["episode"]["verdict"]
    verdict["correct"] = True
    verdict["paid_reward"] = 0.8
    verdict["checks"] = {key: True for key in verdict["checks"]}
    return artifact


def _write_run(root: Path, run_id: str, artifacts: list[dict], *, cost: float) -> Path:
    run_dir = root / run_id
    (run_dir / "sandbox").mkdir(parents=True)
    tasks = []
    for artifact in artifacts:
        episode = artifact["episode"]
        verdict = episode["verdict"]
        tasks.append(
            {
                "task_id": episode["task_id"],
                "family": episode["family"],
                "outcome": episode["outcome"],
                "correct": verdict["correct"],
                "reward": verdict["paid_reward"] if verdict["correct"] else 0.0,
                "error_classification": None,
                "error_detail": None,
            }
        )
    summary = {
        "run_id": run_id,
        "aggregate_score": sum(t["reward"] for t in tasks) / len(tasks),
        "tasks": tasks,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "episode_results.jsonl").write_text(
        "".join(json.dumps(a) + "\n" for a in artifacts), encoding="utf-8"
    )
    sandbox_rows = [
        {
            "problem_id": a["episode"]["session_id"],
            "status": "success",
            "execution_time": 100.0 + index,
            "inference_total": 12,
            "inference_failure_count": 0,
            "_shadow_inference_usage": {
                "inference_cost_usd": cost,
                "inference_total": 12,
                "inference_failed": 0,
                "prompt_tokens": 1000,
                "completion_tokens": 200,
            },
        }
        for index, a in enumerate(artifacts)
    ]
    (run_dir / "sandbox" / "sandbox_output.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in sandbox_rows), encoding="utf-8"
    )
    return run_dir


@pytest.fixture
def runs(tmp_path):
    recovery = _artifact("TF6-recovery-200000")
    ranking = _artifact("TF5-ranking-200000")
    base = _write_run(tmp_path, "local-base", [recovery, ranking], cost=0.02)
    mine = _write_run(tmp_path, "local-mine", [_passing(recovery), ranking], cost=0.03)
    return base, mine


def test_load_run_joins_episode_and_sandbox(runs):
    base, _ = runs
    run = local_bench.load_run(base, "ref")
    by_task = {e.task_id: e for e in run.episodes}

    recovery = by_task["TF6-recovery-200000"]
    assert recovery.family == "recovery"
    assert not recovery.correct and recovery.reward == 0.0
    assert recovery.reward_reason == "hard_gate_failed"
    assert recovery.failed_checks == ["final_in_gold"]
    assert "relaxation_aligned_success" in recovery.false_metrics
    assert recovery.family_detail["chosen_key"] == "3704520688053462282::4547174329-3"
    assert recovery.events and "stockout" in recovery.events[0]
    assert "wait, is that still available?" in recovery.user_messages
    assert recovery.query
    assert recovery.inference_cost_usd == 0.02
    assert recovery.execution_time == 100.0

    ranking = by_task["TF5-ranking-200000"]
    assert ranking.terminal_reason == "step_limit"
    assert ranking.reward_reason == "invalid_family_metric"


def test_build_report_compares_agents(runs):
    base, mine = runs
    report = local_bench.build_report(
        [local_bench.load_run(base, "ref"), local_bench.load_run(mine, "mine")]
    )

    assert report["labels"] == ["ref", "mine"]
    assert report["agents"]["ref"]["aggregate"]["mean"] == 0.0
    assert report["agents"]["mine"]["aggregate"]["mean"] == pytest.approx(0.4)
    assert report["agents"]["mine"]["families"]["recovery"]["pass_rate"] == 1.0
    assert report["agents"]["ref"]["failed_checks"] == {"final_in_gold": 1}
    assert report["agents"]["mine"]["cost"]["usd_per_task"] == pytest.approx(0.03)

    deltas = {t["task_id"]: t["delta"] for t in report["tasks"]}
    assert deltas["TF6-recovery-200000"] == pytest.approx(0.8)
    assert deltas["TF5-ranking-200000"] == 0.0

    text = local_bench.render_text(report)
    assert "AGGREGATE" in text and "+0.80" in text


def test_flaky_tasks_are_flagged(runs, tmp_path):
    base, mine = runs
    report = local_bench.build_report(
        [local_bench.load_run(base, "mine"), local_bench.load_run(mine, "mine")]
    )
    flaky = {t["task_id"]: t["flaky"] for t in report["tasks"]}
    assert flaky["TF6-recovery-200000"] == ["mine"]
    assert flaky["TF5-ranking-200000"] == []
    assert report["agents"]["mine"]["aggregate"]["n"] == 2


def test_missing_receipt_counts_as_zero(tmp_path):
    run_dir = tmp_path / "local-partial"
    run_dir.mkdir()
    summary = {
        "run_id": "local-partial",
        "aggregate_score": None,
        "tasks": [
            {
                "task_id": "TF1-x-1",
                "family": "intent",
                "outcome": "agent_error",
                "correct": False,
                "reward": 0.0,
                "error_classification": "agent",
                "error_detail": "boom",
            }
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    run = local_bench.load_run(run_dir, "a")
    assert [e.task_id for e in run.episodes] == ["TF1-x-1"]
    assert run.episodes[0].error_classification == "agent"
    assert run.aggregate == 0.0


def test_report_and_failures_commands(runs, tmp_path, capsys):
    base, mine = runs
    out = tmp_path / "out"
    assert (
        local_bench.main(
            ["report", "--run", f"ref={base}", "--run", f"mine={mine}", "--out", str(out)]
        )
        == 0
    )
    assert (out / "report.json").is_file() and (out / "report.txt").is_file()

    capsys.readouterr()
    assert (
        local_bench.main(
            ["failures", "--run", f"ref={base}", "--family", "recovery"]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "TF6-recovery-200000" in text
    assert "TF5-ranking-200000" not in text
    assert "failed    final_in_gold" in text
    assert "1 failed episode(s)" in text


def test_bench_index_round_trip(runs, tmp_path, monkeypatch):
    base, mine = runs
    bench = tmp_path / "bench"
    bench.mkdir()
    local_bench._write_index(
        bench,
        {
            "schema_version": local_bench.INDEX_SCHEMA,
            "runs": [
                {"label": "ref", "run_dir": str(base)},
                {"label": "mine", "run_dir": str(mine)},
                {"label": "mine", "run_dir": None},
            ],
        },
    )
    runs_loaded = local_bench.load_bench(bench)
    assert [r.label for r in runs_loaded] == ["ref", "mine"]


def test_run_command_records_each_round(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "agents").mkdir(parents=True)
    agent = repo / "agents" / "a.py"
    agent.write_text("def agent_main(problem_data):\n    return []\n")
    monkeypatch.setattr(local_bench, "REPO_ROOT", repo)
    monkeypatch.setattr(local_bench, "RUNS_ROOT", repo / "logs" / "environment-runs")
    monkeypatch.setattr(local_bench, "BENCH_ROOT", repo / "logs" / "bench")
    monkeypatch.chdir(repo)

    calls = []

    def fake_run_once(agent_file, *, problems, seed, log_path, compose):
        calls.append((agent_file, problems, seed))
        log_path.write_text("ok\n")
        return 0, None, 1.0

    monkeypatch.setattr(local_bench, "run_once", fake_run_once)
    code = local_bench.main(
        [
            "run",
            "--name",
            "t",
            "--agent",
            "x=agents/a.py",
            "--agent",
            "y=agents/a.py",
            "--repeats",
            "2",
            "--problems",
            "5",
            "--seed",
            "7",
        ]
    )
    assert code == 0
    assert [c[0] for c in calls] == ["agents/a.py"] * 4
    # Both agents in a round share a seed; rounds differ.
    assert calls[0][2] == calls[1][2] and calls[2][2] == calls[3][2]
    assert calls[0][2] != calls[2][2]
    index = json.loads((repo / "logs" / "bench" / "t" / "bench.json").read_text())
    assert [(r["round"], r["label"]) for r in index["runs"]] == [
        (1, "x"),
        (1, "y"),
        (2, "x"),
        (2, "y"),
    ]


def test_agent_outside_repo_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(local_bench, "REPO_ROOT", tmp_path / "repo")
    outside = tmp_path / "agent.py"
    outside.write_text("")
    with pytest.raises(SystemExit):
        local_bench._agent_path(str(outside))
