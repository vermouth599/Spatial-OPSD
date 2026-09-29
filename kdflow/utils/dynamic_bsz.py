"""Compatibility hook for the fixed-microbatch OPSD training recipe."""


def rearrange_global_batch(*args, **kwargs):
    raise NotImplementedError(
        "Dynamic batching is not included in this compact training package. "
        "Use the fixed microbatch configuration in scripts/train_opsd.sh."
    )
