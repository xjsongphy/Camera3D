from __future__ import annotations

import warnings


def install_third_party_warning_filters() -> None:
    """Suppress known low-signal warnings from our current ML stack."""
    warnings.filterwarnings(
        "ignore",
        message=r"Windows does not yet support torch\.compile and the performance will be affected\.",
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.cuda\.amp\.custom_fwd\(args\.\.\.\)` is deprecated\..*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.cuda\.amp\.custom_bwd\(args\.\.\.\)` is deprecated\..*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"You are using `torch\.load` with `weights_only=False`.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Importing `spectral_angle_mapper` from `torchmetrics\.functional` was deprecated.*",
        category=FutureWarning,
    )
