"""Validation tests.

These exercise the two real loader failure modes: a module with ``.scales`` on
disk but absent from the map (silent default-width quantization) and a module
listed in the map with no tensors on disk (load-time quantization then missing
weights).
"""

from rco_mlx.mlx_alloc import MlxQuantConfig
from rco_mlx.validate import per_group_byte_breakdown, validate_allocation


def _cfg():
    cfg = MlxQuantConfig(default_bits=4, group_size=64)
    cfg.set("backbone.layers.0.mixer.q_proj", 4)
    cfg.set("backbone.layers.0.mixer.out_proj", 4)
    cfg.set("backbone.layers.1.mixer.switch_mlp.fc1", 2)
    cfg.set("backbone.embeddings", 8)
    return cfg


def test_clean_allocation_passes():
    cfg = _cfg()
    keys = [
        "backbone.layers.0.mixer.q_proj.scales",
        "backbone.layers.0.mixer.out_proj.scales",
        "backbone.layers.1.mixer.switch_mlp.fc1.scales",
        "backbone.embeddings.scales",
        "backbone.layers.0.mixer.q_proj.weight",
        "backbone.layers.0.norm.weight",
    ]
    report = validate_allocation(cfg, on_disk_keys=keys)
    assert report.ok, report.summary()


def test_undeclared_scales_on_disk_is_an_error():
    """A quantized tensor the map does not mention is silently re-quantized."""
    cfg = _cfg()
    keys = [
        "backbone.layers.0.mixer.q_proj.scales",
        "backbone.layers.0.mixer.out_proj.scales",
        "backbone.layers.1.mixer.switch_mlp.fc1.scales",
        "backbone.embeddings.scales",
        "backbone.layers.9.mixer.q_proj.scales",  # not declared
    ]
    report = validate_allocation(cfg, on_disk_keys=keys)
    assert not report.ok
    assert any("absent from the quantization map" in e for e in report.errors)


def test_declared_but_missing_on_disk_is_an_error():
    cfg = _cfg()
    keys = [
        "backbone.layers.0.mixer.q_proj.scales",
        "backbone.layers.0.mixer.out_proj.scales",
        "backbone.embeddings.scales",
        # backbone.layers.1.mixer.switch_mlp.fc1 is declared but absent
    ]
    report = validate_allocation(cfg, on_disk_keys=keys)
    assert not report.ok
    assert any("no .scales on disk" in e for e in report.errors)


def test_unsupported_width_is_an_error():
    cfg = MlxQuantConfig()
    cfg.paths["x"] = {"group_size": 64, "bits": 7, "mode": "affine"}
    report = validate_allocation(cfg)
    assert not report.ok
    assert any("bits=7" in e for e in report.errors)


def test_declared_path_absent_from_the_model_is_an_error():
    cfg = _cfg()
    model_params = [
        "backbone.layers.0.mixer.q_proj.weight",
        "backbone.layers.0.mixer.out_proj.weight",
        "backbone.layers.1.mixer.switch_mlp.fc1.weight",
        # embeddings missing
    ]
    report = validate_allocation(cfg, model_param_names=model_params)
    assert not report.ok
    assert any("not present in the model" in e for e in report.errors)


def test_clean_allocation_passes_with_model_params():
    cfg = _cfg()
    model_params = [
        "backbone.layers.0.mixer.q_proj.weight",
        "backbone.layers.0.mixer.out_proj.weight",
        "backbone.layers.1.mixer.switch_mlp.fc1.weight",
        "backbone.embeddings.weight",
        "backbone.layers.0.norm.weight",
    ]
    keys = [f"{p.removesuffix('.weight')}.scales" for p in model_params if "norm" not in p]
    report = validate_allocation(cfg, on_disk_keys=keys, model_param_names=model_params)
    assert report.ok, report.summary()


def test_report_counts_paths():
    cfg = _cfg()
    keys = [
        "backbone.layers.0.mixer.q_proj.scales",
        "backbone.layers.0.mixer.out_proj.scales",
        "backbone.layers.1.mixer.switch_mlp.fc1.scales",
        "backbone.embeddings.scales",
    ]
    report = validate_allocation(cfg, on_disk_keys=keys)
    assert report.declared_paths == 4
    assert report.on_disk_quantized == 4


def test_per_group_byte_breakdown_scales_with_bits():
    """Byte cost must scale with width, for a fixed number of parameters."""
    cfg = MlxQuantConfig()
    cfg.set("a", 2)
    cfg.set("b", 4)
    cfg.set("c", 8)
    params = {"a": 1000, "b": 1000, "c": 1000}

    buckets = per_group_byte_breakdown(cfg, params)

    assert "8bit" in buckets
    assert "2bit" in buckets
    assert "4bit" in buckets
    # 8-bit must cost more than 4-bit, which must cost more than 2-bit
    assert buckets["8bit"] > buckets["4bit"] > buckets["2bit"]


def test_per_group_byte_breakdown_includes_scale_overhead():
    """scales+biases are part of the quantized module, not free bf16."""
    cfg = MlxQuantConfig()
    cfg.set("m", 4)
    buckets = per_group_byte_breakdown(cfg, {"m": 64000})
    # 64000 weights at 4 bits = 32000 bytes; scales+biases add (64000/64)*4 = 4000
    assert buckets["4bit"] == 32000 + 4000
