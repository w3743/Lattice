"""Where the loss-model request key lives.

A loss model must travel *in the request* rather than be read from the spec by
the backend, because a simulation result has to be reproducible from its own
request and its cache key is built from it. The key therefore has to be nameable
by both the request builder and the analytic backend, so it lives here in a leaf
module that neither one can create a cycle through.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["POWER_LOSS_CONDITIONS_KEY", "loss_parameters_from_conditions"]


#: ``OperatingConditions.variables`` key carrying the loss-model coefficients.
POWER_LOSS_CONDITIONS_KEY = "power_loss_parameters"


def loss_parameters_from_conditions(conditions: Any) -> Any:
    """Loss coefficients a set of operating conditions asks for, or ``None``.

    ``None`` means "no loss model": a request that says nothing keeps the ideal
    lossless answer it has always produced.
    """

    variables = getattr(conditions, "variables", None)
    if not isinstance(variables, Mapping):
        return None
    raw = variables.get(POWER_LOSS_CONDITIONS_KEY)
    if not isinstance(raw, Mapping):
        return None
    from .power_loss import loss_parameters_from_options

    return loss_parameters_from_options(dict(raw))
