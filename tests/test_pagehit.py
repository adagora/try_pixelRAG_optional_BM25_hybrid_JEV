"""The record that crosses three layers. What used to be implicit convention is
asserted here instead."""

from __future__ import annotations

import dataclasses

import pytest

import pagehit
from pagehit import Chunk, PageHit


def region(ci, score=0.5):
    return Chunk(chunk_index=ci, score=score, weighted=score, scale="region")


def gist(score=0.5):
    return Chunk(chunk_index=0, score=score, weighted=score, scale="page")


def test_page_is_one_based_and_tile_index_is_zero_based():
    """Deriving one from the other in the wrong direction is an off-by-one this
    codebase has already paid for once."""
    assert PageHit(article_id=0, tile_index=0).page == 1
    assert PageHit(article_id=0, tile_index=60).page == 61


def test_key_identifies_a_page_within_a_document():
    """Not the page number alone: both catalogues have a page 61."""
    a = PageHit(article_id=0, tile_index=60)
    b = PageHit(article_id=1, tile_index=60)
    assert a.page == b.page and a.key != b.key


def test_focus_is_the_first_region():
    p = PageHit(article_id=0, tile_index=0, chunks=(region(3), region(5)))
    assert p.focus == 3


def test_focus_skips_the_gist():
    """A gist box spans the page, so drawing it marks everything."""
    p = PageHit(article_id=0, tile_index=0, chunks=(gist(0.9), region(4)))
    assert p.focus == 4


def test_focus_none_when_only_the_gist_matched():
    assert PageHit(article_id=0, tile_index=0, chunks=(gist(),)).focus is None


def test_no_geometry_for_a_text_only_hit():
    """Not a placeholder — there genuinely is no region box for a page the text
    index found alone, and citations.py falls back to page-level."""
    p = pagehit.from_lexical({"article_id": 0, "page": 12, "score": 8.0})
    assert not p.has_geometry and p.focus is None
    assert p.tile_index == 11 and p.page == 12


def test_ranking_score_is_the_raw_score_without_fusion():
    p = PageHit(article_id=0, tile_index=0, score=0.42)
    assert p.ranking_score == 0.42


def test_ranking_score_is_rrf_once_retrievers_are_fused():
    """The two are not comparable, so which one orders the list must be
    unambiguous — that is what collapsing them into one key destroyed."""
    p = PageHit(article_id=0, tile_index=0, score=8.0, rrf=0.031)
    assert p.ranking_score == 0.031
    assert p.score == 8.0                        # raw evidence still reported


def test_rrf_of_zero_still_wins_over_the_raw_score():
    """`rrf is not None`, not truthiness: a genuine 0.0 rank score is a fused
    page nothing voted for, not an unfused one."""
    assert PageHit(article_id=0, tile_index=0, score=8.0, rrf=0.0).ranking_score == 0.0


def test_found_by_reports_provenance():
    assert PageHit(article_id=0, tile_index=0).found_by == "visual"
    assert pagehit.from_lexical(
        {"article_id": 0, "page": 1, "score": 1.0}).found_by == "lexical"
    both = PageHit(article_id=0, tile_index=0,
                   sources=frozenset({pagehit.VISUAL, pagehit.LEXICAL}))
    assert both.found_by == "both"


def test_with_does_not_mutate_the_original():
    """fuse() and rrf() used to write into the caller's dicts, so what a page
    contained depended on which functions had already seen it."""
    p = PageHit(article_id=0, tile_index=0, score=0.5)
    q = p.with_(norm=1.0, rank=0)
    assert p.norm == 0.0 and p.rank == 0
    assert q.norm == 1.0 and q.score == 0.5


def test_the_record_is_frozen():
    p = PageHit(article_id=0, tile_index=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.score = 1.0


def test_chunks_are_hashable_so_the_record_is():
    """Frozen and tuple-valued throughout, so a PageHit can key a dict."""
    p = PageHit(article_id=0, tile_index=0, chunks=(region(1),))
    assert {p: "ok"}[p] == "ok"
