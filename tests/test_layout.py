"""Paths and the chunk manifest — the two things that used to be spelled out in
eight places and resolved against the process working directory."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys

import pytest

import chunkmeta
import layout


# -- anchoring --------------------------------------------------------------

def test_root_is_the_repo_not_the_working_directory():
    assert (layout.ROOT / "pixelrag.yaml").exists()
    assert (layout.ROOT / "scripts" / "layout.py").exists()


def test_default_index_is_absolute():
    """The whole point: `python scripts/ask.py` from another directory used to
    silently find no index."""
    assert layout.DEFAULT.index_dir.is_absolute()


def test_a_relative_override_is_anchored_to_the_repo(monkeypatch):
    monkeypatch.setenv("PIXELRAG_INDEX_DIR", "index_b")
    assert layout.default().index_dir == layout.ROOT / "index_b"


def test_an_absolute_override_is_left_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("PIXELRAG_INDEX_DIR", str(tmp_path))
    assert layout.default().index_dir == tmp_path


def test_modules_resolve_paths_identically_from_any_cwd():
    """Run a fresh interpreter from / and check every module still points at
    this repo's index."""
    probe = (
        "import rag, lexical, answer_cache, imagefit;"
        "print(rag.INDEX_DIR); print(lexical.TEXT_SIDECAR);"
        "print(answer_cache.VEC_PATH); print(imagefit.CACHE_DIR)"
    )
    env = {**os.environ, "PYTHONPATH": str(layout.ROOT / "scripts")}
    out = subprocess.run([sys.executable, "-c", probe], cwd="/", env=env,
                         capture_output=True, text=True, check=True).stdout.split()
    assert out[0] == str(layout.DEFAULT.index_dir)
    assert out[1] == str(layout.DEFAULT.text_sidecar)
    assert out[2] == str(layout.DEFAULT.answer_cache_vectors)
    assert out[3] == str(layout.DEFAULT.fit_cache)


# -- the storage convention -------------------------------------------------

def test_tile_dir_names_by_article_id(tmp_path):
    idx = layout.IndexLayout(tmp_path)
    assert idx.tile_dir(3).name == "3.png.tiles"


def test_page_image_is_zero_padded_and_zero_based(tmp_path):
    idx = layout.IndexLayout(tmp_path)
    assert idx.page_image(0, 7).name == "tile_0007.jpg"


def test_tile_dirs_sort_numerically_not_lexically(tmp_path):
    """Article 10 must not land between 1 and 2."""
    idx = layout.IndexLayout(tmp_path)
    for aid in (0, 1, 2, 10, 11):
        idx.tile_dir(aid).mkdir(parents=True)
    assert [layout._article_id_of(d) for d in idx.tile_dirs()] == [0, 1, 2, 10, 11]


def test_article_id_of_rejects_a_stray_directory(tmp_path):
    assert layout._article_id_of(tmp_path / "_fit") == -1


def test_layout_is_frozen(index):
    """These are facts about a build, not knobs to reassign at runtime."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        index.index_dir = "/tmp"


# -- chunk metadata ---------------------------------------------------------

def test_chunkmeta_reads_geometry_and_scale(index):
    c = chunkmeta.get(0, 0, 1, index)
    assert (c.x, c.y, c.width, c.height) == (0, 0, 800, 500)
    assert c.scale == chunkmeta.REGION and not c.is_gist


def test_chunkmeta_identifies_the_gist(index):
    assert chunkmeta.get(0, 0, 0, index).is_gist


def test_chunkmeta_missing_manifest_is_empty_not_an_error(tmp_path):
    assert chunkmeta.for_article(0, layout.IndexLayout(tmp_path)) == {}


def test_chunkmeta_corrupt_manifest_is_empty_not_an_error(tmp_path):
    idx = layout.IndexLayout(tmp_path)
    idx.tile_dir(0).mkdir(parents=True)
    idx.chunks_manifest(0).write_text("{not json")
    assert chunkmeta.for_article(0, idx) == {}


def test_scale_of_defaults_to_region_when_unknown(tmp_path):
    """Region costs a hit the gist bonus rather than granting one it did not
    earn, and keeps the chunk eligible as a highlight box."""
    assert chunkmeta.scale_of(0, 0, 0, layout.IndexLayout(tmp_path)) == \
        chunkmeta.REGION


def test_scale_falls_back_to_the_chunk_zero_convention(tmp_path):
    """An index built before `scale` was recorded still ranks correctly."""
    idx = layout.IndexLayout(tmp_path)
    idx.tile_dir(0).mkdir(parents=True)
    idx.chunks_manifest(0).write_text(json.dumps({"chunks": [
        {"tile_index": 0, "chunk_index": 0},
        {"tile_index": 0, "chunk_index": 1},
    ]}))
    assert chunkmeta.scale_of(0, 0, 0, idx) == chunkmeta.GIST
    assert chunkmeta.scale_of(0, 0, 1, idx) == chunkmeta.REGION


def test_has_box_rejects_a_zero_area_crop(tmp_path):
    """The stock PDF path emits chunks with no recorded width."""
    idx = layout.IndexLayout(tmp_path)
    idx.tile_dir(0).mkdir(parents=True)
    idx.chunks_manifest(0).write_text(json.dumps({"chunks": [
        {"tile_index": 0, "chunk_index": 0, "width": 0, "height": 0}]}))
    assert not chunkmeta.get(0, 0, 0, idx).has_box
