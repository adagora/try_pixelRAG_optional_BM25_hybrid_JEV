"""Shared fixtures.

These tests deliberately touch no network and no model. Everything they cover
is pure logic that used to be reachable only by standing up three services and
paying for a reader call — which is why it had no tests before.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def tiles_dir(tmp_path):
    """A minimal tiles tree: one article, two pages, gist + region chunks.

    Mirrors what chunk_multiscale.py writes, including the `scale` field and the
    `{aid}.png.tiles` directory convention.
    """
    d = tmp_path / "tiles" / "0.png.tiles"
    d.mkdir(parents=True)
    chunks = []
    for tile in (0, 1):
        chunks.append({"tile_index": tile, "chunk_index": 0, "scale": "page",
                       "x_offset": 0, "y_offset": 0, "width": 800, "height": 1000})
        for ci in (1, 2):
            chunks.append({"tile_index": tile, "chunk_index": ci, "scale": "region",
                           "x_offset": 0, "y_offset": 500 * (ci - 1),
                           "width": 800, "height": 500})
    (d / "chunks.json").write_text(json.dumps({"chunks": chunks}))
    return str(tmp_path / "tiles")


@pytest.fixture
def hits():
    """Chunk hits as `pixelrag serve` /search returns them."""
    def make(*triples):
        return [{"article_id": a, "tile_index": t, "chunk_index": c, "score": s}
                for a, t, c, s in triples]
    return make
