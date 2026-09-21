"""The arithmetic behind the two replication claims.

All pure functions here — no network, no key. The numbers these produce are
what a decision to keep paying for a hosted model, or to swap in a local one,
would be made on, so an off-by-one in the AUC is a business error.
"""

from __future__ import annotations

import calibration


def test_a_reversed_rubric_score_is_mirrored_not_compared_raw():
    """With criteria reversed, "Direct answer" is level 0. Comparing the raw
    numbers would measure the reversal instead of the model, and report a
    perfectly stable model as maximally unstable."""
    assert calibration.mirrored(3.0) == 0.0
    assert calibration.mirrored(0.0) == 3.0
    assert calibration.mirrored(1.5) == 1.5          # the fixed point


def rows(*items):
    """(answerable, scope, is_answerable, is_in_domain)"""
    return [tuple(i) for i in items]


def test_auc_is_one_when_the_classes_do_not_overlap():
    c = calibration.Calibration()
    c.rows = rows((0.9, 0.9, True, True), (0.8, 0.9, True, True),
                  (0.1, 0.9, False, True), (0.05, 0.9, False, True))
    assert c.separation(0, label_index=2)["auc"] == 1.0


def test_auc_is_a_coin_when_the_signal_carries_nothing():
    c = calibration.Calibration()
    c.rows = rows((0.5, 0.9, True, True), (0.5, 0.9, False, True))
    assert c.separation(0, label_index=2)["auc"] == 0.5


def test_each_signal_is_graded_against_the_question_it_answers():
    """`scope` scored against answerability reads 0.5 and looks broken. It is
    not broken — every negative in the generated set is IN scope by
    construction, so that test has no negative class. Graded against domain,
    the same rows separate perfectly."""
    c = calibration.Calibration()
    c.rows = rows(
        (0.9, 0.9, True, True),        # answerable, in domain
        (0.05, 0.9, False, True),      # in domain, corpus lacks the fact
        (0.05, 0.02, False, False),    # out of domain
    )
    assert c.separation(1, label_index=2)["auc"] < 1.0      # wrong question
    assert c.separation(1, label_index=3)["auc"] == 1.0     # right question


def test_separation_reports_none_rather_than_inventing_a_class():
    c = calibration.Calibration()
    c.rows = rows((0.9, 0.9, True, True))
    assert c.separation(0, label_index=2)["auc"] is None


def test_a_missing_scope_is_skipped_not_counted_as_zero():
    """A facet that failed and a facet that answered "no" must not be the same
    number, or an outage looks like an out-of-domain question."""
    c = calibration.Calibration()
    c.rows = rows((0.9, None, True, True), (0.1, 0.02, False, False))
    assert c.separation(1, label_index=3)["pos"] == 0


def test_the_operating_point_counts_what_a_threshold_would_have_cost():
    c = calibration.Calibration()
    c.rows = rows((0.9, 0.9, True, True), (0.6, 0.9, True, True),
                  (0.1, 0.9, False, True), (0.05, 0.02, False, False))
    at = c.at(0.5)
    assert at["coverage"] == 0.5                 # 2 of 4 answered
    assert at["correct"] == 1.0                  # both were answerable
    assert at["wrongly_refused"] == 0

    tight = c.at(0.7)
    assert tight["wrongly_refused"] == 1         # the 0.6 was answerable


def test_reliability_bins_report_their_own_sample_size():
    """A bin holding one question is not a point on a curve, and a reader has
    to be able to see that without reading the source."""
    c = calibration.Calibration()
    c.rows = rows((0.9, 0.9, True, True), (0.85, 0.9, True, True),
                  (0.1, 0.9, False, True))
    by_bin = {r["bin"]: r for r in c.reliability()}
    assert by_bin["0.8-1.0"]["n"] == 2 and by_bin["0.8-1.0"]["actual"] == 1.0
    assert by_bin["0.0-0.2"]["n"] == 1 and by_bin["0.0-0.2"]["actual"] == 0.0


def test_out_of_domain_questions_are_supplied_because_they_cannot_be_mined():
    """An oracle that reads the corpus can only produce in-domain questions.
    These are the one input the calibration test needs a human for."""
    assert len(calibration.OUT_OF_DOMAIN) >= 5
    assert all(q.endswith("?") for q in calibration.OUT_OF_DOMAIN)
