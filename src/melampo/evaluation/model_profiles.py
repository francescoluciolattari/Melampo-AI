"""The chat models the project can use for the roles (stage 6) and train with LoRA, in one place
(13 October 2026; Frank: a development project for the future must allow both Gemma 4 and Nemotron, with
one training procedure where possible; Nemotron 3 Super dropped the same day: Gemma 4 31B and Nemotron 3
Ultra are the two choices).

A profile says where the model comes from, what it needs to run and to be trained, and how sure we are
of each of those facts. The same procedure serves every profile:

1. ``scripts/lora_roles.py prepare``  the role-only data, five folds grouped by report (no GPU);
2. ``scripts/lora_roles.py export``   the fold as chat-format JSONL (``messages``), the format the
                                       trainers below read;
3. training, by the backend of the profile:
   * ``trl``: ``scripts/lora_roles.py train --profile <name>`` (QLoRA with Hugging Face TRL, one GPU);
   * ``nemo-automodel``: NVIDIA NeMo AutoModel with the LoRA recipe NVIDIA publishes for the model,
     its dataset pointed at the exported files (``scripts/lora_roles.py plan --profile <name>`` prints
     the steps); the recipes run on one or more nodes;
4. ``scripts/lora_roles.py evaluate --endpoint``: the base model and the base model with the adapter, both
   served behind an OpenAI-compatible endpoint (vLLM, NIM, ...), asked the same question and scored the
   same way (``scripts/role_probe.py``), whatever the model.

``verified`` lists what has been run in this project; everything else comes from the vendors' pages
(``sources``) and must be checked on the hardware before it is relied on.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Profile:
    name: str
    hf_id: str                      # Hugging Face repository of the weights
    openrouter_id: str              # the same model on OpenRouter (public texts only)
    licence: str
    params: str
    run_hardware: str               # to serve it (inference)
    train_backend: str              # "trl" or "nemo-automodel"
    train_recipe: str               # the vendor's recipe, when the backend has one
    train_hardware: str
    verified: tuple[str, ...] = ()  # what this project has run with it
    sources: tuple[str, ...] = field(default=())


PROFILES: dict[str, Profile] = {
    "gemma-4-31b": Profile(
        name="gemma-4-31b",
        hf_id="google/gemma-4-31B-it",
        openrouter_id="google/gemma-4-31b-it",
        licence="Apache 2.0",
        params="31B dense (image-text checkpoint, used here for text)",
        run_hardware="1 GPU of 80 GB (BF16) or less quantised",
        train_backend="nemo-automodel",
        train_recipe="gemma4_31b_peft.yaml (LoRA); a one-GPU QLoRA with --backend trl is also possible",
        train_hardware="NVIDIA's example runs on 8 GPUs; QLoRA on 1 GPU of 80 GB is an estimate, not run",
        verified=("role-probe 13 Oct 2026 through OpenRouter: roles 91.1% alone, 99.27% agreeing with the reader",),
        sources=("https://docs.nvidia.com/nemo/automodel/model-coverage/vision-language-models/google/gemma-4-31B-it.md",
                 "https://openrouter.ai/google/gemma-4-31b-it"),
    ),
    "nemotron-3-ultra": Profile(
        name="nemotron-3-ultra",
        hf_id="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",  # also -NVFP4
        openrouter_id="nvidia/nemotron-3-ultra-550b-a55b",
        licence="NVIDIA Open Model License",
        params="550B, 55B active",
        run_hardware="one node of 8 GPUs",
        train_backend="nemo-automodel",
        train_recipe="NeMo AutoModel PEFT (LoRA) recipe for Nemotron-3-Ultra-550B",
        train_hardware="4 nodes: 16 GB200 (validated by NVIDIA) or 4 x 8 H100",
        verified=("role-probe 13 Oct 2026 through OpenRouter: roles 94.3% alone, 99.30% agreeing with the reader",),
        sources=("https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/nemotron-3-ultra",
                 "https://openrouter.ai/nvidia/nemotron-3-ultra-550b-a55b"),
    ),
}

DEFAULT_PROFILE = "gemma-4-31b"


def profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown model profile {name!r}; known: {', '.join(PROFILES)}") from None


__all__ = ["DEFAULT_PROFILE", "PROFILES", "Profile", "profile"]
