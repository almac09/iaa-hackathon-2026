"""Unit tests for python/geo_tests.py (A26)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import common  # noqa: E402
import geo_tests  # noqa: E402


def test_feature_additions_copy_the_config():
    cfg = common.load_config()
    before = list(cfg["glm_lines"]["models"]["motor"]["numeric"])
    out = geo_tests.add_numeric(cfg, "motor", geo_tests.CLIMATE)
    assert out["glm_lines"]["models"]["motor"]["numeric"] == before + geo_tests.CLIMATE
    assert cfg["glm_lines"]["models"]["motor"]["numeric"] == before          # original untouched
    cat = geo_tests.add_categorical(cfg, "contents", "region_urban")
    assert cat["glm_lines"]["models"]["contents"]["categorical"][-1] == "region_urban"
    assert "region_urban" not in common.load_config()["glm_lines"]["models"]["contents"]["categorical"]
