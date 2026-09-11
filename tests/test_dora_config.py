"""Tests for the DoRA training configuration -- a decision recorded as code."""

import pytest

from melampo.training.dora_config import (
    DEFAULT_ALPHA,
    DEFAULT_RANK,
    TASK_TYPE_CAUSAL_LM,
    DoraTrainingConfig,
    TrainingRunPlan,
)

# --------------------------------------------------------------------------
# The config mirrors peft.LoraConfig's own parameter names exactly
# --------------------------------------------------------------------------


def test_as_peft_kwargs_uses_peft_s_own_parameter_names():
    """Every key must match peft.LoraConfig's actual constructor parameters,
    not a paraphrase -- this is what lets LoraConfig(**kwargs) work without a
    translation layer."""
    kwargs = DoraTrainingConfig().as_peft_kwargs()
    for expected_key in ("r", "lora_alpha", "lora_dropout", "target_modules", "use_dora", "bias", "task_type"):
        assert expected_key in kwargs


def test_use_dora_defaults_to_true():
    """The entire point of this module over a plain LoRA config: DoRA must
    be on by default, not something a caller has to remember to enable."""
    assert DoraTrainingConfig().use_dora is True


def test_defaults_match_the_recorded_2026_recommendation():
    config = DoraTrainingConfig()
    assert config.r == DEFAULT_RANK == 16
    assert config.lora_alpha == DEFAULT_ALPHA
    assert config.task_type == TASK_TYPE_CAUSAL_LM


def test_target_modules_defaults_to_all_linear_not_attention_only():
    """Restricting to attention-only would be fine-tuning under yesterday's
    default; all-linear is what realises DoRA's measured advantage."""
    assert DoraTrainingConfig().target_modules == "all-linear"


def test_a_caller_can_override_any_field():
    config = DoraTrainingConfig(r=32, lora_alpha=64, use_dora=False)
    kwargs = config.as_peft_kwargs()
    assert kwargs["r"] == 32
    assert kwargs["lora_alpha"] == 64
    assert kwargs["use_dora"] is False, "overriding to plain LoRA must remain possible, just not the default"


# --------------------------------------------------------------------------
# TrainingRunPlan: a base model is required, never guessed
# --------------------------------------------------------------------------


def test_a_plan_without_a_base_model_raises_rather_than_guessing():
    """The whole point of the vetting bench is to decide the base model;
    this module must never substitute a default for that decision."""
    with pytest.raises(ValueError, match="base_model"):
        TrainingRunPlan(base_model="")


def test_a_plan_with_a_base_model_carries_the_dora_config():
    plan = TrainingRunPlan(base_model="claude-opus-5")
    assert plan.base_model == "claude-opus-5"
    assert plan.dora.use_dora is True


def test_plan_as_dict_carries_every_field_a_reviewer_needs():
    plan = TrainingRunPlan(base_model="gpt-oss-120b")
    payload = plan.as_dict()
    for key in ("base_model", "dora", "learning_rate", "num_train_epochs"):
        assert key in payload
    assert payload["dora"]["use_dora"] is True


def test_a_plan_can_override_the_dora_config_explicitly():
    custom = DoraTrainingConfig(r=8)
    plan = TrainingRunPlan(base_model="gpt-oss-120b", dora=custom)
    assert plan.dora.r == 8
