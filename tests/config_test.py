"""
Test configuration

Test the configuration initialisation
"""

from frg.utils.utils import get_cfg_defaults


def test_cfg_defaults():
    cfg = get_cfg_defaults()

    assert hasattr(cfg, "COV")
    assert cfg.COV.PATH is None

    assert hasattr(cfg, "DIST")
    assert cfg.DIST.NUM_SAMPLES == 1000
    assert cfg.DIST.SIGMA == 1.0
    assert cfg.DIST.RATIO == 0.5
    assert cfg.DIST.SEED == 42

    assert hasattr(cfg, "SIG")
    assert cfg.SIG.IMAGE is None
    assert cfg.SIG.SNR == 0.0
