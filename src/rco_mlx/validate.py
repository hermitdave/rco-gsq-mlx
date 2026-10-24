"""Structural validation for a mixed-precision MLX allocation.

Loading a 25-44 GB checkpoint off slow storage takes ~15 minutes, so these
checks exist to catch naming and layout errors *before* that load, for free.

The checks mirror the two failure modes the MLX loader actually has:

  1. a module with ``.scales`` on disk but absent from the per-path map is
     quantized at the default width -- so an unmapped module is a silent
     corruption, not a harmless omission;
  2. a module listed in the map but with no ``.scales`` on disk is quantized at
     load time and then fails to find its weights.

Pass condition: every module the loader will quantize is listed, every listed
module has its tensors on disk, and every declared width is one MLX supports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

from .mlx_alloc import MlxQuantConfig
from .objective import MLX_SUPPORTED_BITS, MLX_SUPPORTED_GROUP_SIZES

__all__ = ["ValidationReport", "validate_allocation", "assert_coverage"]


@dataclass
class ValidationReport:
    """Result of a structural validation."""

    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    declared_paths: int = 0
    on_disk_quantized: int = 0

    def error(self, message: str) -> None:
        self.errors.append(message)
        self.ok = False

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def summary(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        lines = [
            f"[{status}] declared={self.declared_paths} on_disk={self.on_disk_quantized}"
        ]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        return "\n".join(lines)


def _module_paths_from_scales(keys: Iterable[str]) -> Set[str]:
    """Extract module paths from a key set by stripping the quant suffix.

    ``<module>.scales`` -> ``<module>``. Only ``.scales`` is used as the marker
    because every quantized module has one, whereas ``.weight`` is shared with
    unquantized modules.
    """
    paths = set()
    for key in keys:
        if key.endswith(".scales"):
            paths.add(key[: -len(".scales")])
    return paths


def validate_allocation(
    cfg: MlxQuantConfig,
    on_disk_keys: Optional[Iterable[str]] = None,
    model_param_names: Optional[Iterable[str]] = None,
) -> ValidationReport:
    """Validate an allocation against what is on disk and what the model wants.

    Parameters
    ----------
    cfg : MlxQuantConfig
        The allocation to check.
    on_disk_keys : iterable of str, optional
        Tensor keys from the checkpoint's ``model.safetensors.index.json`` (or
        the flattened parameter names). Used to confirm each declared path
        actually has quantized tensors, and to catch quantized tensors that
        were never declared.
    model_param_names : iterable of str, optional
        The model's expected parameter names, e.g. from ``tree_flatten`` on an
        instantiated model. Used to catch declared paths the model does not
        have (a typo in a module path).

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()
    declared = set(cfg.paths.keys())
    report.declared_paths = len(declared)

    # 1. widths are all MLX-representable
    for path, spec in sorted(cfg.paths.items()):
        bits = spec.get("bits")
        gs = spec.get("group_size")
        if bits not in MLX_SUPPORTED_BITS:
            report.error(f"{path}: bits={bits} not supported (supported: {MLX_SUPPORTED_BITS})")
        if gs not in MLX_SUPPORTED_GROUP_SIZES:
            report.error(
                f"{path}: group_size={gs} not supported "
                f"(supported: {MLX_SUPPORTED_GROUP_SIZES})"
            )

    # 2. nothing declared that the model does not have
    if model_param_names is not None:
        wanted = set()
        for name in model_param_names:
            if name.endswith(".weight"):
                wanted.add(name[: -len(".weight")])
            else:
                wanted.add(name)
        extra = sorted(declared - wanted)
        if extra:
            report.error(
                f"{len(extra)} declared path(s) not present in the model "
                f"(first few: {extra[:3]})"
            )

    # 3. everything on disk that looks quantized is declared, and vice versa
    if on_disk_keys is not None:
        on_disk = _module_paths_from_scales(on_disk_keys)
        report.on_disk_quantized = len(on_disk)

        undeclared = sorted(on_disk - declared)
        if undeclared:
            report.error(
                f"{len(undeclared)} module(s) have .scales on disk but are absent "
                f"from the quantization map -- the loader will quantize them at the "
                f"default width ({cfg.default_bits}-bit). First few: {undeclared[:3]}"
            )

        missing = sorted(declared - on_disk)
        if missing:
            report.error(
                f"{len(missing)} declared module(s) have no .scales on disk -- the "
                f"loader will quantize them at load and fail to find their weights. "
                f"First few: {missing[:3]}"
            )

    # 4. coverage: every quantized path resolves to a width (never a silent default)
    for path in sorted(declared):
        if cfg.bits_for(path) not in MLX_SUPPORTED_BITS:
            report.error(f"{path}: resolves to an unsupported width")

    return report


def assert_coverage(cfg: MlxQuantConfig, **kwargs) -> None:
    """Raise ``AssertionError`` if validation fails; used as a hard gate."""
    report = validate_allocation(cfg, **kwargs)
    if not report.ok:
        raise AssertionError("allocation validation failed:\n" + report.summary())


def per_group_byte_breakdown(
    cfg: MlxQuantConfig,
    param_counts: Mapping[str, int],
    group_size: Optional[int] = None,
) -> Dict[str, int]:
    """Bytes per module, grouped by declared bit width.

    A total byte count hides allocation bugs completely -- tensor counts,
    shapes, key sets and generations can all pass while the size is wrong.
    ``scales`` and ``biases`` are part of a quantized module, so they are
    included here rather than counted as "unquantized bf16".
    """
    buckets: Dict[int, int] = {}
    for path, spec in cfg.paths.items():
        bits = int(spec["bits"])
        gs = int(spec.get("group_size", cfg.group_size))
        n = int(param_counts.get(path, 0))
        if n <= 0:
            continue
        # packed weight + bf16 scale + bf16 bias per group
        weight_bytes = n * bits / 8.0
        overhead_bytes = (n / gs) * 2 * 2
        buckets[bits] = buckets.get(bits, 0) + int(weight_bytes + overhead_bytes)

    return {f"{b}bit": buckets[b] for b in sorted(buckets)}
