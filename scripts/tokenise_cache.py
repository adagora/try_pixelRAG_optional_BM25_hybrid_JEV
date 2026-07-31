"""Kill the per-request kwargs plumbing in `processor(text=...)`.

The `tokenise` phase costs ~5.9ms per query on this box, and only ~0.7ms of it
is tokenising. The rest is `ProcessorMixin._merge_kwargs`, which for every call
synthesises a fresh `TypedDict("merged_typed_dict", ...)` and hands it to
huggingface_hub's `validate_typed_dict`. That helper *is* lru_cached — on the
schema object — so a new class per call means the cache never hits and
`dataclasses.make_dataclass` + `exec` runs on the request path
(transformers/processing_utils.py:1372, huggingface_hub/dataclasses.py:341).

Both strategies here are parity-safe by construction rather than by measurement:
neither touches text, tokeniser, or tensors. They only stop recomputing a value
that is constant for a given call signature. `assert_identical` proves it on
real queries anyway, because "by construction" is how parity bugs get written.

    memoise_strict_classes()      # library keeps validating, just once per shape
    cache_merged_kwargs(proc)     # skip merge+validate entirely after call one
"""

import copy


def memoise_strict_classes() -> None:
    """Re-key huggingface_hub's strict-dataclass cache on schema *contents*.

    Conservative option: every validation the library would run still runs, so a
    genuinely bad kwarg still raises in the same place. Only the class
    construction is shared between calls that ask for the same shape.
    """
    from huggingface_hub import dataclasses as hf_dc

    if getattr(hf_dc.validate_typed_dict, "_pixelrag_memoised", False):
        return

    original = hf_dc.validate_typed_dict
    build = hf_dc._build_strict_cls_from_typed_dict
    canonical: dict[tuple, type] = {}

    def validate_typed_dict(schema, data):
        ann = getattr(schema, "__annotations__", None)
        if ann is None:
            return original(schema, data)
        # Name and totality both feed the built class, so both belong in the key.
        key = (schema.__name__, getattr(schema, "__total__", True),
               tuple(sorted((k, repr(v)) for k, v in ann.items())))
        cls = canonical.get(key)
        if cls is None:
            cls = canonical[key] = build(schema)
        cls(**data)

    validate_typed_dict._pixelrag_memoised = True
    hf_dc.validate_typed_dict = validate_typed_dict
    # processing_utils did `from ... import validate_typed_dict`, so rebinding the
    # module attribute alone would leave the request path on the old function.
    import transformers.processing_utils as pu

    pu.validate_typed_dict = validate_typed_dict


def cache_merged_kwargs(processor) -> None:
    """Memoise `processor._merge_kwargs` on its call signature.

    The merged result is a pure function of (schema, tokenizer init kwargs, call
    kwargs), and a query encoder passes the same call kwargs every time — only
    the text differs, and text is not a kwarg. Returns a deepcopy because
    `__call__` pops `return_tensors` and `return_mm_token_type_ids` out of it.
    """
    inner = getattr(processor._merge_kwargs, "_pixelrag_inner", None) or processor._merge_kwargs
    cache: dict[str, dict] = {}

    def merge(ModelProcessorKwargs, tokenizer_init_kwargs=None, **kwargs):
        key = f"{ModelProcessorKwargs.__qualname__}|{sorted(kwargs.items(), key=repr)!r}"
        if key not in cache:
            cache[key] = inner(ModelProcessorKwargs, tokenizer_init_kwargs, **kwargs)
        return copy.deepcopy(cache[key])

    merge._pixelrag_inner = inner
    processor._merge_kwargs = merge


def assert_identical(stock_processor, fast_processor, prompts: list[str]) -> None:
    """Prove the fast path returns the same tensors, keys and dtypes.

    Cheap enough to run at sidecar startup: a wrong tokenisation is a silent
    retrieval regression, and this is the only place it can be caught for free.
    """
    import torch

    for prompt in prompts:
        a = stock_processor(text=[prompt], return_tensors="pt", padding=True)
        b = fast_processor(text=[prompt], return_tensors="pt", padding=True)
        if set(a.keys()) != set(b.keys()):
            raise AssertionError(f"key mismatch: {sorted(a.keys())} vs {sorted(b.keys())}")
        for k in a:
            x, y = a[k], b[k]
            if isinstance(x, torch.Tensor):
                if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
                    raise AssertionError(f"tensor {k} differs")
            elif x != y:
                raise AssertionError(f"value {k} differs: {x!r} vs {y!r}")
