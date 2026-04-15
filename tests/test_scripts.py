"""Test scripts.

Test the computation scripts
"""

from unittest.mock import patch

import numpy as np

from frg.scripts.canonical_dimensions import main as cd_main
from frg.scripts.evc_distribution import main as evc_main
from frg.scripts.frg_equations import main as fe_main
from frg.scripts.frg_equations_lpa import main as felpa_main
from frg.scripts.generate_config import main as gc_main
from frg.scripts.init import main as init_main


def test_canonical_dimensions_script(tmp_path, monkeypatch):
    # Create a dummy config
    cfg_path = tmp_path / "test_cfg.yaml"
    cfg_path.write_text(
        "DATA:\n  OUTPUT_DIR: "
        + str(tmp_path)
        + "\nDIST:\n  VAR: 1.0\n  RATIO: 0.5",
    )

    # Run main with analytic flag
    args = ["--analytic", "--config", str(cfg_path), "--suffix", "var"]
    with patch("matplotlib.pyplot.savefig"):
        rc = cd_main(args)
        assert rc == 0


def test_generate_config_script(tmp_path):
    base_config = tmp_path / "base.yaml"
    base_config.write_text("DIST:\n  NUM_SAMPLES: 100")

    params_file = tmp_path / "params.json"
    params_file.write_text('{"DIST": {"RATIO": [0.1, 0.5]}}')

    args = [
        "--config",
        str(base_config),
        "--params",
        str(params_file),
        "--output_dir",
        str(tmp_path),
    ]
    rc = gc_main(args)
    assert rc == 0


def test_init_script(tmp_path):
    # init.py doesn't have --output, it uses cwd
    with patch("frg.scripts.init.copy_resource_dir") as mock_copy:
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            rc = init_main(["--force"])
            assert rc == 0
            assert mock_copy.called


def test_evc_distribution_script(tmp_path):
    import numpy as np

    data_path = tmp_path / "data.npy"
    np.save(data_path, np.cov(np.random.randn(100, 10), rowvar=False))
    cfg_path = tmp_path / "test_cfg.yaml"
    cfg_path.write_text(
        f"DATA:\n  OUTPUT_DIR: {tmp_path}\nSIG:\n  INPUT: {data_path}",
    )
    args = ["--config", str(cfg_path)]
    with patch("matplotlib.pyplot.savefig"):
        rc = evc_main(args)
        assert rc == 0


def test_frg_equations_script(tmp_path):
    args = ["--analytic"]
    with (
        patch("matplotlib.pyplot.savefig"),
        patch(
            "frg.distributions.distributions.MarchenkoPastur.frg_equations",
            return_value=np.zeros((1, 4)),
        ),
    ):
        rc = fe_main(args)
        assert rc == 0


def test_frg_equations_lpa_script(tmp_path):
    args = ["--analytic"]
    with (
        patch("matplotlib.pyplot.savefig"),
        patch(
            "frg.distributions.distributions.MarchenkoPastur.frg_equations_lpa",
            return_value=np.zeros((1, 4)),
        ),
    ):
        rc = felpa_main(args)
        assert rc == 0
