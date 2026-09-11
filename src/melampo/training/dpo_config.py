"""DPO training configuration, connected to what preference_pairs.py actually produces.

Decided in EG #15, then checked rather than assumed: `preference_pairs.py`
verifies that (prompt, chosen, rejected) triples already fall out of
`ConfirmationRegistry` and the alternatives `mechanism_enumeration` raises
for the same case, with no separate preference-collection pipeline needed.
This module is the connection promised at that decision -- not a duplicate
of the extraction logic, a consumer of its actual output, via
`as_training_records`.

DPO over full RLVR as the primary mechanism, for two reasons found in the
same research, not assumed independently. Simplicity: DPO trains offline
against a static preference dataset, no reward model, no PPO-style rollout
loop -- the heavy infrastructure a system like AReaL exists to provide (see
EG #14's evaluation of it, correctly not adopted for a different task) is
not needed here. Regulatory fit: offline, reproducible-batch training is
explicitly preferred by 2026 auditing guidance for frameworks like the EU AI
Act, over an online RL loop whose exact trajectory is harder to reconstruct
after the fact -- directly relevant to a project with MDR ambitions. RLVR
remains available as a later refinement layered on top, not replaced.

**The same boundary as dora_config.py, restated because it still applies.**
No base model is chosen, this sandboxed environment cannot install `trl` or
run a GPU, and this file produces a configuration and a data-shape adapter,
not a runnable training loop. What it verifies, without needing any of that:
whether the extracted pairs actually reach the row count and field shape a
real DPO trainer needs, using the real extraction path rather than a
hand-built fixture.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .preference_pairs import ExtractionReport, PreferencePair, as_training_records

# TRL's own defaults as of the version this project's DPO route targets;
# named here rather than left implicit, so a reviewer sees what was chosen
# and not just that something was.
DEFAULT_BETA = 0.1
DEFAULT_LEARNING_RATE = 5e-7
DEFAULT_NUM_TRAIN_EPOCHS = 1

# Below this many pairs, a DPO run is more likely to overfit to specific
# cases than to learn a generalisable preference -- the same concern
# `information_content.DEFAULT_IC`'s treatment of thin evidence and
# `hypothesis_yield`'s Wilson intervals both encode: a thin sample should be
# visibly thin, not silently trained on as if it were enough. Not a hard
# block -- `DpoReadiness.is_ready` is a recommendation the caller can
# override, not an exception this module raises on their behalf.
MIN_PAIRS_RECOMMENDED = 50


@dataclass(frozen=True)
class DpoTrainingConfig:
    """DPO hyperparameters, field-for-field compatible with trl.DPOConfig.

    Field names match TRL's own parameter names, the same discipline
    `DoraTrainingConfig` applies to `peft.LoraConfig` -- so this becomes a
    real `DPOConfig` with one line wherever TRL is actually installed,
    without a translation layer that could drift from TRL's own API.
    """

    beta: float = DEFAULT_BETA
    learning_rate: float = DEFAULT_LEARNING_RATE
    num_train_epochs: int = DEFAULT_NUM_TRAIN_EPOCHS

    def as_trl_kwargs(self) -> dict[str, Any]:
        return {"beta": self.beta, "learning_rate": self.learning_rate, "num_train_epochs": self.num_train_epochs}

    def as_dict(self) -> dict[str, Any]:
        return dict(self.as_trl_kwargs())


@dataclass(frozen=True)
class DpoReadiness:
    """Whether the pairs currently available are enough to train on -- and how many there are.

    Kept as data the caller inspects, not a gate this module enforces --
    the same posture `verify_mechanism`'s grounding states take: report what
    was found, let the caller decide what to do with a thin result rather
    than silently deciding for them.
    """

    pair_count: int
    usable_cases: int
    is_ready: bool

    def as_dict(self) -> dict[str, Any]:
        return {"pair_count": self.pair_count, "usable_cases": self.usable_cases, "is_ready": self.is_ready}


def assess_readiness(report: ExtractionReport, *, min_pairs: int = MIN_PAIRS_RECOMMENDED) -> DpoReadiness:
    """Whether an extraction report has enough pairs to be worth training on yet."""
    return DpoReadiness(
        pair_count=len(report.pairs), usable_cases=report.usable_cases, is_ready=len(report.pairs) >= min_pairs
    )


def build_training_dataset(pairs: Sequence[PreferencePair]) -> list[dict[str, str]]:
    """The dataset TRL's DPOTrainer would be given -- a direct pass-through.

    Not a reimplementation: `as_training_records` in preference_pairs.py
    already does this conversion, and duplicating it here would be exactly
    the kind of second copy that could drift from the first the way
    duplicated verdict-ranking logic once did in this project. This function
    exists so a caller configuring a DPO run imports one module for both the
    config and the dataset, without needing to know the dataset actually
    comes from a sibling file.
    """
    return as_training_records(pairs)
