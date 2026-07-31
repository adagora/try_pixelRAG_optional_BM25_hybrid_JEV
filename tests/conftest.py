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


@pytest.fixture
def fake_reader():
    """A Reader that answers from a script instead of from a model.

    This is what providers.py exists for. Before the Reader protocol the answer
    pipeline could only be exercised with a live faiss service, a live encoder
    and a billed API call, so it had no tests at all. This class imports no SDK,
    needs no key and touches no network, and rag.run_agent cannot tell the
    difference.
    """
    import imagefit
    import providers

    class FakeReader:
        name = "fake"

        def __init__(self, answer="Odpowiedź.", usage=None, steps=1,
                     model="fake-1", policy=None, cost=None):
            self.model = model
            self._answer = answer
            self._usage = usage or providers.Usage(input=10, output=5)
            self._steps = steps
            self._policy = policy or imagefit.PASSTHROUGH
            self._cost = cost
            # What the pipeline handed over, for assertions.
            self.system = None
            self.preamble = None
            self.pages = None
            self.headers = []
            self.calls = 0

        @property
        def image_policy(self):
            return self._policy

        def price(self, usage):
            return self._cost

        def models(self):
            return [self.model]

        def read_pages(self, system, preamble, pages, header_of, on_text):
            self.calls += 1
            self.system, self.preamble, self.pages = system, preamble, pages
            self.headers = [header_of(p) for p in pages]
            # Stream in two pieces, so a consumer that reassembles deltas is
            # exercised the way a real provider exercises it.
            half = len(self._answer) // 2
            on_text(self._answer[:half])
            on_text(self._answer[half:])
            return providers.Reply(self._answer, self._usage, self._steps)

        def browse(self, system, question, tools, dispatch, on_event, max_steps):
            self.calls += 1
            self.system = system
            return providers.Reply(self._answer, self._usage, self._steps)

    return FakeReader
