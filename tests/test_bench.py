"""The benchmark that produces every figure in the README, and had no tests.

`bench.py` decides what the repo claims. Its grading, its verified-subset rule
and — since the table started carrying error bars — its statistics are the
things a reader trusts without being able to check them, so they are what this
file pins down. Nothing here touches the index, the encoder or TypeSafe: the
scoring is arithmetic over question records, and that is the half worth
protecting.
"""

from __future__ import annotations

import pytest

import bench


# --------------------------------------------------------------------------
# Wilson intervals
# --------------------------------------------------------------------------

def test_wilson_brackets_the_point_estimate():
    for k, n in [(0, 13), (1, 13), (8, 13), (11, 13), (12, 13), (13, 13)]:
        lo, hi = bench.wilson(k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0, (k, n, lo, hi)


def test_wilson_never_leaves_the_unit_interval_at_the_ends():
    """The normal approximation is why this is Wilson: at 13/13 it would put
    the upper bound above 1.0 and the lower bound above the point estimate."""
    lo, hi = bench.wilson(13, 13)
    assert hi == 1.0
    assert lo < 1.0            # still reports uncertainty at a perfect score

    lo, hi = bench.wilson(0, 13)
    assert lo == 0.0
    assert hi > 0.0            # and at a perfect failure


def test_wilson_narrows_as_the_question_set_grows():
    """The whole reason the interval is printed: it should visibly reward a
    wider eval set, which is the fix this repo needs."""
    def width(n: int) -> float:
        lo, hi = bench.wilson(round(0.85 * n), n)
        return hi - lo

    assert width(13) > width(50) > width(200)


def test_wilson_is_undefined_rather_than_exploding_on_no_questions():
    assert bench.wilson(0, 0) == (0.0, 0.0)


# --------------------------------------------------------------------------
# McNemar — the only legitimate comparison between two rows
# --------------------------------------------------------------------------

def _outcomes(*flags: bool) -> dict[str, bool]:
    return {f"q{i}": f for i, f in enumerate(flags)}


def test_modes_that_agree_everywhere_have_nothing_to_compare():
    a = _outcomes(True, True, False, False)
    assert bench.mcnemar(a, dict(a)) == (0, 0, 1.0)


def test_only_discordant_questions_count():
    """Ten questions both modes got right tell you nothing about which is
    better — this is the point of the pairing, so it is the test."""
    a = _outcomes(True, True, True, True, True, True, True, True, True, True)
    b = dict(a)
    a["extra"], b["extra"] = True, False
    wins_a, wins_b, _ = bench.mcnemar(a, b)
    assert (wins_a, wins_b) == (1, 0)


def test_a_one_question_lead_is_not_significant():
    """jev-page over hybrid on the shipped question set, exactly: 1-0."""
    a = _outcomes(True)
    b = _outcomes(False)
    _, _, p = bench.mcnemar(a, b)
    assert p == pytest.approx(1.0)


def test_a_four_nil_lead_is_still_not_significant_at_this_size():
    """jev-page over visual, exactly: 4-0 and p=0.125. Directionally clean,
    and 13 questions cannot resolve it — which is the finding."""
    a = _outcomes(True, True, True, True)
    b = _outcomes(False, False, False, False)
    wins_a, wins_b, p = bench.mcnemar(a, b)
    assert (wins_a, wins_b) == (4, 0)
    assert p == pytest.approx(0.125)


def test_six_nil_is_the_threshold_this_set_would_have_to_clear():
    """A clean sweep of SIX questions the other mode lost, and not one lost
    back, is the smallest evidence that licenses "better" at p<0.05. Five is
    0.0625 and does not. Worth pinning as a number: the shipped set's best
    margin is 4-0, so no pair of modes in the README is separated."""
    _, _, five = bench.mcnemar(_outcomes(*[True] * 5), _outcomes(*[False] * 5))
    _, _, six = bench.mcnemar(_outcomes(*[True] * 6), _outcomes(*[False] * 6))
    assert five == pytest.approx(0.0625) and five > 0.05
    assert six == pytest.approx(0.03125) and six < 0.05


def test_p_is_symmetric_in_its_arguments():
    a, b = _outcomes(True, True, False), _outcomes(False, False, True)
    assert bench.mcnemar(a, b)[2] == bench.mcnemar(b, a)[2]
    assert bench.mcnemar(a, b)[:2] == bench.mcnemar(b, a)[:2][::-1]


def test_p_never_exceeds_one_on_an_even_split():
    """2*tail double-counts the centre when wins are equal; the clamp is not
    cosmetic, a p above 1.0 would be printed."""
    for n in range(1, 9):
        _, _, p = bench.mcnemar(_outcomes(*([True] * n + [False] * n)),
                                _outcomes(*([False] * n + [True] * n)))
        assert 0.0 < p <= 1.0


def test_questions_only_one_mode_answered_are_ignored():
    """Two runs over different subsets must compare on the intersection rather
    than crediting a mode for a question the other never saw."""
    a = {"shared": True, "only_a": True}
    b = {"shared": False, "only_b": False}
    assert bench.mcnemar(a, b) == (1, 0, 1.0)


# --------------------------------------------------------------------------
# The subset rule that makes the table non-circular
# --------------------------------------------------------------------------

@pytest.mark.parametrize("question, subject", [
    ('Co to jest "RAL 7040"?', "RAL 7040"),
    ('Ile kosztuje "BRAMY GARAŻOWE"?', "BRAMY GARAŻOWE"),
    ("KORONY PERGOLI — jakie warianty są dostępne?", "KORONY PERGOLI"),
])
def test_subject_recovers_both_generated_shapes(question, subject):
    assert bench._subject(question) == subject


def test_a_hand_written_question_yields_no_usable_subject():
    """It must not qualify for --verified rather than qualify wrongly: the
    whole subset rule is 'the string decides', and there is no string here."""
    q = "kolory tkanin soltis"
    assert bench._subject(q) == q      # no quotes, no em dash — itself


# --------------------------------------------------------------------------
# Result arithmetic
# --------------------------------------------------------------------------

def test_percentages_are_over_scored_questions_not_over_negatives():
    """The negatives are graded by `refuse` alone. Counting them in the
    denominator would silently deflate every mode by the size of a class it is
    not being asked about."""
    r = bench.Result(mode="visual", top1=8, recall=12, scored=13,
                     covered=12, gold_total=13, ceiling=13,
                     refused=10, negatives=10)
    d = r.as_dict()
    assert d["top1"] == pytest.approx(61.5, abs=0.1)
    assert d["refused"] == 100.0
    assert d["questions"] == 13


def test_a_blocked_mode_reports_no_numbers_at_all():
    r = bench.Result(mode="jev", blocked="no TYPESAFE_API_KEY")
    d = r.as_dict()
    assert d["top1"] is None and d["refused"] is None
    assert "— no TYPESAFE_API_KEY" in bench.table([r])


def test_median_latency_survives_one_cold_model_load():
    """Documented intent of Result.ms — a 26 s first call is not the mode."""
    r = bench.Result(mode="visual", latencies=[26000.0, 9.0, 8.0, 10.0, 9.0])
    assert r.ms < 50


def test_the_table_carries_an_interval_next_to_every_rate():
    r = bench.Result(mode="hybrid", top1=11, recall=12, scored=13,
                     covered=12, gold_total=13, ceiling=13)
    out = bench.table([r])
    assert "95% CI" in out
    lo, hi = r.as_dict()["top1_ci95"]
    assert f"[{lo:.0f}-{hi:.0f}]" in out


# --------------------------------------------------------------------------
# The resolution block
# --------------------------------------------------------------------------

def _result(mode: str, flags: list[bool]) -> bench.Result:
    return bench.Result(mode=mode, top1=sum(flags), scored=len(flags),
                        recall=len(flags), covered=len(flags),
                        gold_total=len(flags), ceiling=len(flags),
                        per_q={f"q{i}": f for i, f in enumerate(flags)})


def test_resolution_names_the_size_of_one_question():
    out = bench.resolution([_result("a", [True] * 13), _result("b", [False] * 13)])
    assert "13 questions" in out and "7.7 points" in out


def test_resolution_calls_a_one_question_lead_unresolved():
    best = _result("jev-page", [True, True, True])
    other = _result("hybrid", [True, True, False])
    out = bench.resolution([best, other])
    assert "1-0 of  1 discordant" in out
    assert "not resolved" in out


def test_resolution_reports_identical_modes_as_identical():
    """jev-page and xray on the shipped set: not coincidentally tied at 92%,
    the same answers to the same questions."""
    flags = [True, True, False]
    out = bench.resolution([_result("jev-page", flags), _result("xray", flags)])
    assert "identical on every question" in out


def test_resolution_compares_against_the_best_row_not_the_first():
    worst = _result("visual", [False, False, False, False, False])
    best = _result("xray", [True, True, True, True, True])
    out = bench.resolution([worst, best])
    assert out.index("xray") < out.index("vs")


def test_resolution_is_silent_when_there_is_nothing_to_compare():
    assert bench.resolution([]) == ""
    assert bench.resolution([_result("only", [True])]) == ""
    assert bench.resolution([bench.Result(mode="jev", blocked="no key")]) == ""


def test_resolution_survives_a_mode_that_errored_on_every_question():
    """run_mode returns a Result with empty per_q when every call raised; the
    block must skip it rather than divide by zero."""
    broken = bench.Result(mode="jev", errors=13)
    out = bench.resolution([_result("hybrid", [True, False]), broken])
    assert "jev" not in out


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def test_load_defaults_a_missing_kind_and_tolerates_absent_pages(tmp_path):
    p = tmp_path / "q.yaml"
    p.write_text(
        '- q: "a?"\n  doc: "d"\n  pages: [1]\n  primary: 1\n'
        '- q: "b?"\n  doc: "d"\n  pages: []\n  answerable: false\n',
        encoding="utf-8")
    qs = bench.load(p)
    assert [q.kind for q in qs] == ["spec", "spec"]
    assert qs[1].pages == [] and qs[1].answerable is False
    assert qs[1].primary is None


def test_load_exits_rather_than_scoring_against_a_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        bench.load(tmp_path / "nope.yaml")
