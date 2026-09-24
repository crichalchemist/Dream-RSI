import pytest

from see.policy.api import SimResult, _record_curve, finalize_result
from see.policy.observation_signal import (
    branch_failed_hard,
    classify_failure,
    is_success,
    probe_improved_vs_parent,
)
from see.synthetic import synthetic_trace
from see.world import Cell, IllegalBatch, ReplayQuestion, Trace, cell_id


def small_trace():
    # branch 0: 3 attempts; branch 1: 1 attempt then stopped; branch 2: failure then recovery
    cells = [
        Cell(0, 0, 0, 1.2),
        Cell(1, 0, 1, 0.9),
        Cell(2, 0, 2, 0.0, fail_class="compile_other", error="compilation failed", n_valid=0),
        Cell(0, 1, 3, 1.5),
        Cell(2, 1, 4, 1.8),
        Cell(0, 2, 5, 1.4),
    ]
    return Trace(cells, baseline_score=1.0, max_parallelism=2, trace_id="t", grid=(3, 2))


def test_trace_rejects_non_chain_branches():
    with pytest.raises(ValueError):
        Trace([Cell(0, 0, 0, 1.0), Cell(0, 2, 1, 1.0)], 1.0, 2)


def test_trace_round_trips_through_json(tmp_path):
    t = synthetic_trace(3)
    path = tmp_path / "trace.json"
    t.save(str(path))
    u = Trace.load(str(path))
    assert u.to_dict() == t.to_dict()


def test_initial_legality_is_roots_only_in_creation_order():
    q = ReplayQuestion(small_trace())
    assert q.legal_roots() == ["b0a0", "b1a0", "b2a0"]
    assert q.legal_actions() == q.legal_roots()
    assert q.observed() == {}
    with pytest.raises(KeyError):
        q.meta("b0a1")  # unrevealed and not legal: structure stays hidden


def test_probe_reveals_recorded_children_in_parent_child_order():
    q = ReplayQuestion(small_trace())
    q.probe_batch(["b0a0", "b2a0"])
    assert q.opened_branches() == [0, 2]
    assert set(q.legal_actions()) == {"b1a0", "b0a1", "b2a1"}
    obs = q.probe_batch(["b0a1", "b2a1"])
    assert [o.cell_id for o in obs] == ["b0a1", "b2a1"]
    # attempt 1 on branch 2 recovers from a failed parent: no parent delta, baseline delta only
    assert obs[1].delta_vs_parent is None and obs[1].delta_vs_baseline == pytest.approx(0.8)
    assert obs[0].delta_vs_parent == pytest.approx(0.3)
    assert q.meta("b0a1").parent_id == "b0a0"


@pytest.mark.parametrize("batch", [["b0a0", "b0a0"], ["b0a1"], ["b0a0", "b1a0", "b2a0"], []])
def test_illegal_batches_are_refused(batch):
    q = ReplayQuestion(small_trace())  # max_parallelism 2
    with pytest.raises(IllegalBatch):
        q.probe_batch(batch)
    assert q.budget_spent == 0 and q.decision_rounds == 0


def test_bookkeeping_and_parallel_accounting():
    q = ReplayQuestion(small_trace())
    res = SimResult()
    q.probe_batch(["b0a0", "b1a0"], on_reveal=lambda _: _record_curve(res, q))
    q.probe_batch(["b0a1"], on_reveal=lambda _: _record_curve(res, q))
    finalize_result(q, res)
    assert (res.total_probes, res.decision_rounds, res.effective_sequential_rounds) == (3, 2, 2)
    assert res.best_score == 1.5
    assert [p["probes"] for p in res.curve] == [1, 2, 3]


def test_failed_cells_never_count_as_best():
    q = ReplayQuestion(small_trace())
    q.probe_batch(["b2a0"])
    assert q.best_so_far is None


def test_replay_ends_when_every_recorded_cell_is_revealed():
    t = small_trace()
    q = ReplayQuestion(t)
    while q.legal_actions():
        q.probe_batch(q.legal_actions()[: q.max_parallelism])
    assert len(q.observed()) == len(t)
    assert q.best_so_far == t.ceiling() == 1.8


def test_round_cap_k2_empties_the_action_set():
    q = ReplayQuestion(small_trace(), max_rounds=1)
    q.probe_batch(["b0a0"])
    assert q.legal_actions() == [] and q.legal_roots() == []


def test_grid_restriction_is_clipped_to_trace_support():
    q = ReplayQuestion(small_trace(), branch_count=2, refine_count=0)
    assert q.legal_roots() == ["b0a0", "b1a0"]
    q.probe_batch(["b0a0"])
    assert "b0a1" not in q.legal_actions()  # refine_count 0: roots only
    wide = ReplayQuestion(small_trace(), branch_count=9)
    assert wide.out_of_support and wide.branch_count == 3


def test_reset_restores_the_initial_state():
    q = ReplayQuestion(small_trace())
    q.probe_batch(["b0a0"])
    q.reset()
    assert q.observed() == {} and q.budget_spent == 0 and q.best_so_far is None
    assert q.legal_actions() == ["b0a0", "b1a0", "b2a0"]


def test_success_semantics_ignore_valid_flag():
    q = ReplayQuestion(Trace([Cell(0, 0, 0, 1.3, valid=False)], 1.0, 1))
    (obs,) = q.probe_batch(["b0a0"])
    assert is_success(obs) and probe_improved_vs_parent(obs)


def test_zero_valid_compile_failure_is_a_hard_signal_but_repairable_class():
    q = ReplayQuestion(small_trace())
    (obs,) = q.probe_batch(["b2a0"])
    assert branch_failed_hard(obs)
    assert obs.fail_class == "compile_other"


@pytest.mark.parametrize(
    "text,expected",
    [
        (None, "ok"),
        ("C++ compilation failed:\nerror: expected ';'", "compile_other"),
        ("Timed out after 600s", "timeout"),
        ("CUDA error: too many resources requested for launch", "resource"),
        ("Output size mismatch: got 8, expected 16", "shape"),
        ("max_gap 3e-4 exceeds tolerance; correctness check failed", "correctness"),
        ("ModuleNotFoundError: No module named 'triton'", "env"),
        ("NameError: name 'x' is not defined", "code"),
    ],
)
def test_failure_classifier(text, expected):
    assert classify_failure(text) == expected


def test_cell_ids_are_opaque_but_stable():
    assert cell_id(3, 7) == "b3a7"
