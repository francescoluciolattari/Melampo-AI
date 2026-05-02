"""Melampo package.

The package root intentionally avoids eager imports of the full runtime.
This keeps lightweight modules, CLI entry points and CI smoke checks stable even
when optional provider integrations are not installed.
"""

__all__ = [
    "build_default_runtime",
    "ClinicalPrototypeRunner",
    "PrototypeInputValidator",
    "run_prototype_case",
]


def __getattr__(name: str):
    if name == "build_default_runtime":
        from .app import build_default_runtime

        return build_default_runtime
    if name in {"ClinicalPrototypeRunner", "PrototypeInputValidator", "run_prototype_case"}:
        from .prototype import ClinicalPrototypeRunner, PrototypeInputValidator, run_prototype_case

        return {
            "ClinicalPrototypeRunner": ClinicalPrototypeRunner,
            "PrototypeInputValidator": PrototypeInputValidator,
            "run_prototype_case": run_prototype_case,
        }[name]
    raise AttributeError(f"module 'melampo' has no attribute {name!r}")
