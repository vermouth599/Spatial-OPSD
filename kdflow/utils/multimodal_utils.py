"""Multi-modal field helpers (verl-inspired)."""
from typing import Iterable, Optional

import torch


def extract_multi_modal_inputs(
    multi_modal_inputs_list: Optional[Iterable[Optional[dict]]],
) -> dict:
    """Concat per-sample mm dicts into a batched dict for HF VLM forward.

    Args:
        multi_modal_inputs_list: an iterable of per-sample multi_modal_inputs
            dicts (each dict maps field name -> tensor with leading dim
            corresponding to that sample's patches/tokens). ``None`` entries
            (e.g. text-only samples) are skipped.

    Returns:
        A dict mapping field name -> tensor concatenated along ``dim=0``
        across all non-empty samples. Empty input returns ``{}``.
    """
    if not multi_modal_inputs_list:
        return {}
    collected: dict = {}
    for d in multi_modal_inputs_list:
        if not d:
            continue
        for k, v in d.items():
            if v is not None:
                collected.setdefault(k, []).append(v)
    # Some models (e.g. Gemma3) produce variable-size tensors per image
    # (different number of image tokens).  Pad to a common shape before
    # concatenating along dim=0.
    batched = {}
    for k, vs in collected.items():
        if len(vs) == 1:
            batched[k] = vs[0]
            continue
        max_shape = list(vs[0].shape)
        for v in vs[1:]:
            for d in range(len(max_shape)):
                max_shape[d] = max(max_shape[d], v.shape[d])
        # Pad all non-leading dims to the max.  F.pad takes padding from last
        # dim to first: (pad_left, pad_right, pad_top, pad_bottom, ...).
        padded = []
        for v in vs:
            pad = []
            for d in range(v.ndim - 1, 0, -1):
                pad.append(0)
                pad.append(max_shape[d] - v.shape[d])
            if any(p > 0 for p in pad):
                import torch.nn.functional as F
                v = F.pad(v, pad)
            padded.append(v)
        batched[k] = torch.cat(padded, dim=0)
    return batched
