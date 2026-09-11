"""DoRA fine-tuning configuration -- a decision recorded as code, not yet a model to apply it to.

Decided in EG #15 after checking measured evidence rather than reputation:
DoRA (Weight-Decomposed Low-Rank Adaptation) improves consistently over plain
LoRA at the same parameter budget (+0.84-0.88% on GLUE in the original paper,
strongest in the low-rank regime this project would use), at a small,
one-time cost (roughly 5-10% more VRAM during training, no extra cost at
inference) and no code change beyond a single flag in the PEFT library
(`use_dora=True`). GaLore was considered and rejected: it updates the full
weight matrix via low-rank gradient projection rather than training a small
adapter, costs roughly four times as much, and approaches full fine-tuning's
risk profile -- a poor trade when confirmed clinical outcomes, the only
legitimate training signal this project has (see `preference_pairs.py`),
arrive as slowly as they do.

**What this module is not.** There is no PEFT configuration anywhere else in
this project -- checking the code rather than recalling a summary found that
out directly. There is also no base model chosen yet: that is what
`evaluation/vetting_bench.py` exists to determine. Writing an executable
training script now would mean building against a model that might not be
the one eventually chosen, and this sandboxed environment has neither the
disk space to install `peft`/`torch` nor a GPU to run them regardless.

What *is* useful to build now, and is what this module does: the
configuration itself, as a plain dataclass whose field names match
`peft.LoraConfig`'s exactly, so that `LoraConfig(**config.as_peft_kwargs())`
is a straight drop-in once a base model and a real training environment
exist. The decision is captured precisely, in a form nothing about it needs
reinterpreting when the time comes to actually use it.
"""

from dataclasses import dataclass, field
from typing import Any

# Unsloth's 2026 guidance, corroborated by the DoRA paper's own ablations:
# r=16 is the recommended default starting point, in the regime where DoRA's
# advantage over plain LoRA is largest.
DEFAULT_RANK = 16
DEFAULT_ALPHA = 16
DEFAULT_DROPOUT = 0.05

# "all-linear" targets every linear projection (attention and MLP alike)
# rather than only the attention projections LoRA was originally proposed
# for -- the wider target set is standard practice by 2026 and is what
# realises DoRA's measured advantage; restricting to attention-only would be
# fine-tuning under yesterday's default, not today's.
DEFAULT_TARGET_MODULES = "all-linear"

TASK_TYPE_CAUSAL_LM = "CAUSAL_LM"


@dataclass(frozen=True)
class DoraTrainingConfig:
    """DoRA hyperparameters, field-for-field compatible with peft.LoraConfig.

    Every field name matches the PEFT library's own parameter name exactly
    -- not a paraphrase of it -- so `as_peft_kwargs()` needs no translation
    layer that could silently drift from what PEFT actually expects as its
    own API evolves.
    """

    r: int = DEFAULT_RANK
    lora_alpha: int = DEFAULT_ALPHA
    lora_dropout: float = DEFAULT_DROPOUT
    target_modules: str = DEFAULT_TARGET_MODULES
    use_dora: bool = True
    bias: str = "none"
    task_type: str = TASK_TYPE_CAUSAL_LM

    def as_peft_kwargs(self) -> dict[str, Any]:
        """The exact keyword arguments for ``peft.LoraConfig(**this)``.

        Returned as a plain dict rather than constructing a LoraConfig
        directly, since peft is not installed in every environment this
        module might be imported from (including the one that wrote it) --
        the dict is usable without the dependency being present, and becomes
        a real LoraConfig with one line wherever peft is available.
        """
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "use_dora": self.use_dora,
            "bias": self.bias,
            "task_type": self.task_type,
        }

    def as_dict(self) -> dict[str, Any]:
        return dict(self.as_peft_kwargs())


@dataclass(frozen=True)
class TrainingRunPlan:
    """What would run, and on what, once a base model is chosen.

    A record of intent, not an executable script -- `base_model` is left
    unset by default specifically so constructing a plan without having made
    that decision fails loudly rather than silently defaulting to a guess.
    """

    base_model: str
    dora: DoraTrainingConfig = field(default_factory=DoraTrainingConfig)
    learning_rate: float = 2e-4
    num_train_epochs: int = 3

    def __post_init__(self) -> None:
        if not self.base_model:
            raise ValueError(
                "TrainingRunPlan requires a base_model -- this is deliberately not "
                "defaulted, since the whole point of the vetting bench is to decide it "
                "rather than have this module guess"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "dora": self.dora.as_dict(),
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
        }
