"""Shared fixtures.

These tests deliberately touch no network and no model. Everything they cover
is pure logic that used to be reachable only by standing up three services and
paying for a reader call — which is why it had no tests before.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def index(tmp_path):
    """A minimal index tree: one article, two pages, gist + region chunks.

    Written through IndexLayout rather than by hand, so a test cannot drift from
    the convention the production code reads. Mirrors what chunk_multiscale.py
    emits, including the `scale` field.
    """
    import layout

    idx = layout.IndexLayout(tmp_path / "index")
    idx.tile_dir(0).mkdir(parents=True)
    chunks = []
    for tile in (0, 1):
        chunks.append({"tile_index": tile, "chunk_index": 0, "scale": "page",
                       "x_offset": 0, "y_offset": 0, "width": 800, "height": 1000})
        for ci in (1, 2):
            chunks.append({"tile_index": tile, "chunk_index": ci, "scale": "region",
                           "x_offset": 0, "y_offset": 500 * (ci - 1),
                           "width": 800, "height": 500})
    idx.chunks_manifest(0).write_text(json.dumps({"chunks": chunks}))
    return idx


@pytest.fixture
def scale_of(index):
    """retrieve.py's ScaleFn, backed by the fixture index."""
    import chunkmeta

    return lambda a, t, c: chunkmeta.scale_of(a, t, c, index)


@pytest.fixture
def hits():
    """Chunk hits as `pixelrag serve` /search returns them."""
    def make(*triples):
        return [{"article_id": a, "tile_index": t, "chunk_index": c, "score": s}
                for a, t, c, s in triples]
    return make
