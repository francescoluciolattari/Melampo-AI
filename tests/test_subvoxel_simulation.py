"""The sub-voxel simulation behind dicom_volume's precision thresholds still says what the thresholds assume."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("simulate_subvoxel_precision", ROOT / "scripts" / "simulate_subvoxel_precision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_thin_slices_localise_a_small_lesion_sub_voxel_and_thick_slices_do_not():
    script = _script()
    thin, thick = script.simulate((4.0,), (1.25, 5.0), trials=8, seed=1)
    # With 1.25 mm slices the centre is found to well under 0.1 mm, along z too.
    assert thin.rms_error_x_mm < 0.1 and thin.rms_error_z_mm < 0.1
    # With 5 mm slices a 4 mm lesion falls in one or two slices: z is lost.
    assert thick.rms_error_z_mm > 0.3
    assert thick.volume_sd_threshold > thin.volume_sd_threshold


def test_the_table_renders_every_configuration():
    script = _script()
    text = script.render(script.simulate((6.0,), (2.5,), trials=2, seed=1))
    assert "| 6 mm | 2.5 mm |" in text
