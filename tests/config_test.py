"""
Test configuration

Test the configuration initialisation
"""

from frg.utils.utils import get_cfg_defaults


def test_cfg_defaults():
    """Test the default configuration"""
    cfg = get_cfg_defaults()

    assert hasattr(cfg, "DIST")
    assert cfg.DIST.NUM_SAMPLES == 1000
    assert cfg.DIST.SIGMA == 1.0
    assert cfg.DIST.RATIO == 0.5
    assert cfg.DIST.SEED == 42

    assert hasattr(cfg, "SIG")
    assert cfg.SIG.INPUT is None
    assert cfg.SIG.SNR == 0.0

    assert hasattr(cfg, "POT")
    assert cfg.POT.UV_SCALE == 1.0e-5
    assert cfg.POT.KAPPA_INIT == 1.0e-5
    assert cfg.POT.U2_INIT == 1.0e-5
    assert cfg.POT.U4_INIT == 1.0e-5
    assert cfg.POT.U6_INIT == 1.0e-5

    assert hasattr(cfg, "DATA")
    assert cfg.DATA.OUTPUT_DIR == "results"

    assert hasattr(cfg, "PLOTS")
    assert cfg.PLOTS.OUTPUT_DIR == "plots"
