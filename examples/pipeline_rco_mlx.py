#!/usr/bin/env python3
"""
RCO mixed-precision quantization pipeline for MLX.

End-to-end: introspect model -> measure losses -> RCO allocation ->
emit MLX config -> validate -> stream quantize.

Usage:
    # With a pre-computed imatrix + GGUF weights:
    python pipeline_rco_mlx.py \
        --source /path/to/bf16-model \
        --output /path/to/output \
        --imatrix imatrix.gguf \
        --gguf-dir gguf-quant \
        --target-bpw 4.0 \
        --group-size 64

    # With a pre-built loss table JSON:
    python pipeline_rco_mlx.py \
        --source /path/to/bf16-model \
        --output /path/to/output \
        --loss-table work/imatrix_alloc.json \
        --target-bpw 4.0 \
        --group-size 64

The loss table JSON is the output of imatrix_importance.py:
    [{"layer": 0, "tensor": "ffn_down_exps", "params": 123456, "mse": {"2": 1.2, "3": 0.8, ...}}, ...]
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors import safe_open

# ---------------------------------------------------------------------------
# rco-mlox imports (installed via `pip install -e ~/Projects/rco-mlx`)
# ---------------------------------------------------------------------------
from rco_mlx import (
    LossTable,
    MlxQuantConfig,
    allocation_to_config,
    attach_param_counts,
    build_layer_groups,
    effective_bits,
    validate_allocation,
    write_config,
)
from rco_mlx.search import budget_dp, values_from_losses


# ---------------------------------------------------------------------------
# Step 1: introspect the source model and build layer groups
# ---------------------------------------------------------------------------
def introspect_model(source_dir: Path, group_spec: str | None = None):
    """Return (layer_names, param_counts, groups) from a BF16 safetensors dir."""
    from safetensors import safe_open

    # discover shards
    shards = sorted(source_dir.glob("*.safetensors"))
    if not shards:
        sys.exit(f"No safetensors shards in {source_dir}")

    # read the first shard's index to get all keys + shapes
    index_path = source_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        # collect unique module paths (strip .weight suffix)
        module_paths = set()
        for key in weight_map:
            if key.endswith(".weight"):
                module_paths.add(key[: -len(".weight")])
            elif key.endswith(".scales") or key.endswith(".biases"):
                continue
            else:
                module_paths.add(key)
        layer_names = sorted(module_paths)
    else:
        # fallback: read the first shard's keys
        with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
            layer_names = sorted(
                k[: -len(".weight")] if k.endswith(".weight") else k
                for k in f.keys()
                if not k.endswith(".scales") and not k.endswith(".biases")
            )

    # param counts from the first shard (enough for fractions)
    param_counts = {}
    with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
        for key in f.keys():
            shape = f.get_tensor(key).shape
            param_counts[key] = int(np.prod(shape))

    # build groups
    group_patterns = None
    if group_spec:
        # parse "pat1,pat2 pat3 pat4,pat5" → [[pat1,pat2], [pat3], [pat4,pat5]]
        group_patterns = [
            [p.strip() for p in group.split(",") if p.strip()]
            for group in group_spec.strip().split()
            if group.strip()
        ]

    groups = build_layer_groups(layer_names, group_patterns)
    attach_param_counts(groups, param_counts)

    return layer_names, param_counts, groups


# ---------------------------------------------------------------------------
# Step 2: build the loss table
# ---------------------------------------------------------------------------
def build_loss_table_from_imatrix(
    imatrix_path: Path,
    gguf_dir: Path,
    layers: list[int],
    which: list[str],
    bits_list: list[int],
    group_size: int,
    out: Path,
):
    """Run imatrix_importance to produce the loss-table JSON."""
    import subprocess

    script = (
        Path(__file__).parent.parent.parent.parent
        / "skills/mlops/streaming-mixed-precision-quantization/scripts/imatrix_importance.py"
    )
    if not script.exists():
        # fall back to the installed script location
        script = Path(
            "~/.hermes/profiles/helix/skills/mlops/streaming-mixed-precision-quantization/scripts/imatrix_importance.py"
        ).expanduser()

    cmd = [
        sys.executable,
        str(script),
        "--imatrix", str(imatrix_path),
        "--gguf-dir", str(gguf_dir),
        "--layers", ",".join(str(l) for l in layers),
        "--which", ",".join(which),
        "--bits", ",".join(str(b) for b in bits_list),
        "--group-size", str(group_size),
        "--out", str(out),
    ]
    print(f"[measure] running imatrix sweep …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"imatrix sweep failed:\n{result.stderr}")
    print(result.stdout)
    return out


def load_loss_table(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the imatrix_importance JSON and return (losses, fracs, bitwidths)."""
    rows = json.loads(path.read_text())
    if not rows:
        sys.exit(f"Empty loss table: {path}")

    # infer bitwidths from the first row
    bitwidths = np.array([int(b) for b in rows[0]["mse"].keys()], dtype=np.int64)

    # build a lookup: (layer, tensor) → {bits: mse}
    lookup = {}
    for r in rows:
        lookup[(r["layer"], r["tensor"])] = {int(b): v for b, v in r["mse"].items()}

    # collect groups from rows
    groups = sorted({(r["layer"], r["tensor"]) for r in rows})

    losses = np.array(
        [[lookup[g][b] for b in bitwidths] for g in groups],
        dtype=np.float64,
    )
    params = np.array(
        [next(r["params"] for r in rows if (r["layer"], r["tensor"]) == g) for g in groups],
        dtype=np.float64,
    )
    fracs = params / params.sum()

    return losses, fracs, bitwidths


def load_error_db(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a SLQ-style error_db.json and return (losses, fracs, bitwidths).

    The error_db format is::

        {
          "layers": {
            "model.layers.0.self_attn.q_proj": {
              "2": {"error": 0.123},
              "3": {"error": 0.456}
            }
          },
          "param_counts": {"model.layers.0.self_attn.q_proj": 123456}
        }

    Returns the same tuple as ``load_loss_table`` so callers are agnostic.
    """
    db = json.loads(path.read_text())
    if "layers" not in db:
        sys.exit(f"Not a valid error_db.json (no 'layers' key): {path}")

    bitwidths = sorted({int(b) for layer_data in db["layers"].values() for b in layer_data})
    param_counts = db.get("param_counts", {})

    rows = []
    for layer_path, bw_data in db["layers"].items():
        params = param_counts.get(layer_path, 0)
        mse = {str(bw): bw_data[str(bw)]["error"] for bw in bitwidths if str(bw) in bw_data}
        rows.append({"layer": layer_path, "tensor": layer_path, "params": params, "mse": mse})

    bitwidths_arr = np.array(bitwidths, dtype=np.int64)
    losses = np.array(
        [[r["mse"][str(b)] for b in bitwidths] for r in rows],
        dtype=np.float64,
    )
    params = np.array([r["params"] for r in rows], dtype=np.float64)
    fracs = params / max(params.sum(), 1.0)

    return losses, fracs, bitwidths_arr


# ---------------------------------------------------------------------------
# Step 3: RCO allocation
# ---------------------------------------------------------------------------
def run_rco_allocation(
    losses: np.ndarray,
    fracs: np.ndarray,
    bitwidths: np.ndarray,
    target_bpw: float,
    group_size: int,
    group_names: list[str],
) -> tuple[dict[str, int], dict]:
    """Run the RCO knapsack DP and return (assignment, report)."""
    table = LossTable(
        losses=losses,
        fracs=fracs,
        bitwidths=bitwidths,
        group_size=group_size,
    )
    report = table.report(target_avg_bits=target_bpw)
    bits_per_group = table.assignment(target_avg_bits=target_bpw)

    assignment = {name: int(b) for name, b in zip(group_names, bits_per_group)}
    return assignment, report


def solve_allocation_greedy(
    losses: np.ndarray,
    fracs: np.ndarray,
    bitwidths: np.ndarray,
    target_bpw: float,
    group_names: list[str],
    bitwidth_map: dict[int, float] | None = None,
) -> tuple[dict[str, int], dict]:
    """Greedy bitwidth allocation (SLQ-style) with a priority queue.

    Start every group at max bitwidth, then repeatedly take the cheapest
    one-rung reduction until the parameter-weighted average hits target.
    Uses a heap so each step is O(log n_groups) instead of O(n_groups).
    """
    import heapq

    bitwidth_map = bitwidth_map or {}
    ordered = sorted(bitwidths.tolist())
    max_bw = ordered[-1]
    next_lower = {ordered[i]: ordered[i - 1] for i in range(1, len(ordered))}

    def get_actual(bw: int) -> float:
        return bitwidth_map.get(bw, float(bw))

    # Build cost lookup: (group_idx, bitwidth) -> loss
    cost = {}
    for i in range(losses.shape[0]):
        for j, bw in enumerate(bitwidths):
            cost[(i, int(bw))] = float(losses[i, j])

    def avg_bw(cfg: list[int]) -> float:
        total_w = sum(fracs[i] for i in range(len(cfg)))
        bits = sum(get_actual(cfg[i]) * fracs[i] for i in range(len(cfg)))
        return bits / total_w if total_w > 0 else 0.0

    # Initialize at max bitwidth
    config = [max_bw] * len(group_names)

    # Precompute initial average
    total_frac = float(fracs.sum())
    current_bits = float(sum(get_actual(config[i]) * fracs[i] for i in range(len(config))))

    # Build heap: (marginal_cost_per_bit, group_idx, expected_cur, next_bw)
    heap = []
    for i in range(len(group_names)):
        cur = config[i]
        nxt = next_lower.get(cur)
        if nxt is None:
            continue
        e_cur = cost.get((i, cur), 0.0)
        e_next = cost.get((i, nxt), 0.0)
        d_err = e_next - e_cur  # negative (error decreases)
        d_bits = get_actual(cur) - get_actual(nxt)
        if d_bits <= 0:
            continue
        marginal = d_err / d_bits
        heapq.heappush(heap, (marginal, i, cur, nxt))

    steps = 0
    max_steps = len(group_names) * len(ordered)

    while current_bits / total_frac > target_bpw and heap and steps < max_steps:
        marginal, i, expected_cur, nxt = heapq.heappop(heap)
        if config[i] != expected_cur:
            continue  # stale entry
        old_bw = get_actual(config[i])
        config[i] = nxt
        new_bw = get_actual(config[i])
        current_bits += (new_bw - old_bw) * fracs[i]
        steps += 1
        if steps % 5000 == 0:
            print(f"  greedy step {steps}: avg={current_bits / total_frac:.4f}")
        # Push next possible drop for this group
        cur = config[i]
        nxt2 = next_lower.get(cur)
        if nxt2 is None:
            continue
        e_cur = cost.get((i, cur), 0.0)
        e_next = cost.get((i, nxt2), 0.0)
        d_err = e_next - e_cur
        d_bits = get_actual(cur) - get_actual(nxt2)
        if d_bits <= 0:
            continue
        marginal = d_err / d_bits
        heapq.heappush(heap, (marginal, i, cur, nxt2))

    achieved = current_bits / total_frac
    assignment = {group_names[i]: config[i] for i in range(len(group_names))}
    report = {
        "achieved_bpw": achieved,
        "steps": steps,
        "bit_histogram": dict(
            sorted(Counter(config).items(), reverse=True)
        ),
    }
    return assignment, report


def filter_bf16_prefixes(assignment: dict[str, int], prefixes: list[str]) -> dict[str, int]:
    """Remove entries whose path contains any of the given prefixes.

    These tensors stay BF16 and should not appear in the quant config.
    Dotted prefixes like ``mlp.gate_proj`` also match ``experts.0.gate_proj``.
    """
    filtered = {}
    for path, bits in assignment.items():
        skip = False
        for p in prefixes:
            if p in path:
                skip = True
                break
            # Dotted prefix: also match by last component
            if "." in p:
                last = p.split(".")[-1]
                if path.endswith("." + last) or ("." + last + ".") in path:
                    skip = True
                    break
        if skip:
            continue
        filtered[path] = bits
    return filtered


# ---------------------------------------------------------------------------
# Step 4: emit + validate MLX config
# ---------------------------------------------------------------------------
def _is_quantizable(path: str) -> bool:
    """Return True if the path points to a quantizable linear layer.

    Works with both the raw checkpoint keys (with ``.weight``) and the
    stripped module paths used in the RCO assignment dict.
    """
    check = path if path.endswith(".weight") else path + ".weight"
    if "norm" in check or "embed" in check or "lm_head" in check or "mtp" in check:
        return False
    # Skip gate_proj (kept BF16 for SwiGLU gating or router logits)
    if check.endswith(".gate_proj.weight"):
        return False
    if not check.endswith(".weight"):
        return False
    return True


def emit_and_validate(
    assignment: dict[str, int],
    group_size: int,
    source_dir: Path,
    output_dir: Path,
    default_bits: int = 4,
    validate_on_disk: bool = True,
    bf16_prefixes: list[str] | None = None,
) -> MlxQuantConfig:
    """Write config.json and validate before loading."""
    # Filter out BF16 prefixes so they never enter the quant config
    filtered = assignment
    if bf16_prefixes:
        filtered = filter_bf16_prefixes(assignment, bf16_prefixes)

    # Only include quantizable layers in the config
    quant_assignment = {k: v for k, v in filtered.items() if _is_quantizable(k)}
    cfg = allocation_to_config(
        quant_assignment,
        default_bits=default_bits,
        group_size=group_size,
        mode="affine",
    )

    # load base config from source
    base_cfg_path = source_dir / "config.json"
    base_cfg = json.loads(base_cfg_path.read_text()) if base_cfg_path.exists() else {}

    # write to output
    out_cfg_path = output_dir / "config.json"
    merged = write_config(cfg, out_cfg_path, base_config=base_cfg)

    on_disk_keys = []
    if validate_on_disk:
        index_path = output_dir / "model.safetensors.index.json"
        if not index_path.exists():
            index_path = source_dir / "model.safetensors.index.json"
        if index_path.exists():
            idx = json.loads(index_path.read_text())
            on_disk_keys = list(idx.get("weight_map", {}).keys())

    report = validate_allocation(cfg, on_disk_keys=on_disk_keys if on_disk_keys else None)
    print(report.summary())
    if not report.ok:
        sys.exit("Validation failed — fix the config before loading.")
    return cfg


# ---------------------------------------------------------------------------
# Step 5: streaming quantize
# ---------------------------------------------------------------------------
def stream_quantize(
    source_dir: Path,
    output_dir: Path,
    assignment: dict[str, int],
    group_size: int,
    model_adapter=None,
):
    """Stream through the BF16 source, quantize per the allocation, write shards."""
    import torch
    from safetensors.torch import save_file

    from rco_mlx.mlx_alloc import MlxQuantConfig

    cfg = allocation_to_config(assignment, group_size=group_size)
    layer_bits = {}
    for path, bits in assignment.items():
        # strip any .weight suffix for the lookup
        clean = path.removesuffix(".weight")
        layer_bits[clean] = bits

    # also build a lookup with .weight suffix for key matching
    layer_bits_with_suffix = {}
    for path, bits in assignment.items():
        if not path.endswith(".weight"):
            layer_bits_with_suffix[path + ".weight"] = bits
        layer_bits_with_suffix[path] = bits

    def map_key_to_mlx(key: str) -> str | None:
        if key.startswith("model.visual.") or key.startswith("visual."):
            return None
        if key.startswith("model.language_model."):
            return "language_model.model." + key[len("model.language_model."):]
        return key

    def should_quantize(mlx_key: str) -> tuple[bool, int]:
        # Always-skip patterns
        if (
            "norm" in mlx_key
            or "embed" in mlx_key
            or "lm_head" in mlx_key
            or "mtp" in mlx_key
            or not mlx_key.endswith(".weight")
            or mlx_key.endswith(".bias")
        ):
            print(f"  SKIP quant: {mlx_key}")
            return False, 4

        # Adapter-specific BF16 prefixes (e.g. router gate)
        if model_adapter is not None:
            for prefix in model_adapter.non_quantized_prefixes():
                if f".{prefix}." in f".{mlx_key}." or mlx_key.startswith(f"{prefix}.") or mlx_key == prefix:
                    print(f"  SKIP bf16 prefix: {mlx_key}")
                    return False, 4

        # check the assignment
        bits = layer_bits_with_suffix.get(mlx_key)
        if bits is None:
            # try without .weight
            base = mlx_key[: -len(".weight")]
            bits = layer_bits.get(base, 4)
        return True, bits

    shards = sorted(source_dir.glob("*.safetensors"))
    output_dir.mkdir(parents=True, exist_ok=True)
    total_tensors = 0
    start = time.time()

    for i, shard_path in enumerate(shards):
        print(f"\n[quant] shard {i+1}/{len(shards)}: {shard_path.name}")
        results = {}
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                mlx_key = map_key_to_mlx(key)
                if mlx_key is None:
                    results[key] = f.get_tensor(key).to(torch.bfloat16)
                    continue

                q_ok, bits = should_quantize(mlx_key)
                if q_ok:
                    tensor = f.get_tensor(key)
                    if tensor.ndim == 2:
                        import mlx.core as mx

                        w = mx.array(tensor.float().numpy())
                        q, scales, biases = mx.quantize(
                            w, bits=bits, group_size=group_size
                        )
                        mx.eval(q, scales, biases)
                        base_key = mlx_key[:-len(".weight")] if mlx_key.endswith(".weight") else mlx_key
                        results[mlx_key] = torch.from_numpy(
                            np.array(q).astype(np.uint32)
                        )
                        results[f"{base_key}.scales"] = torch.from_numpy(
                            np.array(scales).astype(np.float16)
                        )
                        results[f"{base_key}.biases"] = torch.from_numpy(
                            np.array(biases).astype(np.float16)
                        )
                    else:
                        results[mlx_key] = tensor.to(torch.bfloat16)
                else:
                    results[mlx_key] = f.get_tensor(key).to(torch.bfloat16)

        out_shard = output_dir / f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        save_file(results, str(out_shard))
        total_tensors += len(results)
        elapsed = time.time() - start
        print(f"  saved {len(results)} tensors | {elapsed:.1f}s elapsed")

    # Post-process: stack MoE experts if adapter provides stacking logic
    if model_adapter is not None:
        print("\n[quant] post-processing: stacking MoE experts ...")
        _stack_experts(output_dir, model_adapter, group_size)
        _rewrite_config_for_stacked_experts(output_dir, assignment)
        elapsed = time.time() - start
        print(f"  post-process done | {elapsed:.1f}s elapsed")

    # copy non-weight files
    import shutil

    for f in source_dir.iterdir():
        if f.suffix == ".safetensors" or f.name == "model.safetensors.index.json":
            continue
        dest = output_dir / f.name
        if f.is_file() and not dest.exists():
            shutil.copy2(f, dest)

    print(f"\n[quant] done — {total_tensors} tensors, {time.time()-start:.1f}s total")


def _stack_experts(output_dir: Path, model_adapter, group_size: int):
    """Stack individual expert tensors into switch_mlp projections.

    Reads all quantized shards, groups expert tensors by (layer, projection),
    stacks their weights/scales/biases along axis 0, and writes the stacked
    versions. Removes the individual expert tensors.
    """
    import torch
    from safetensors.torch import load_file, save_file
    import mlx.core as mx
    import numpy as np

    # Collect all safetensors files
    shards = sorted(output_dir.glob("*.safetensors"))
    all_tensors = {}
    for shard in shards:
        data = load_file(str(shard), device="cpu")
        all_tensors.update(data)

    # Find expert tensors to stack
    expert_patterns = []
    for key in list(all_tensors.keys()):
        if ".mlp.experts." in key and key.endswith(".weight"):
            # Parse: model.layers.{l}.mlp.experts.{e}.{proj}.weight
            parts = key.split(".")
            if len(parts) >= 6 and parts[4] == "experts":
                l = parts[2]
                e = parts[5]
                proj = parts[6]
                expert_patterns.append((l, e, proj, key))

    if not expert_patterns:
        print("  no expert tensors found to stack")
        return

    # Group by (layer, projection)
    from collections import defaultdict
    groups = defaultdict(list)
    for l, e, proj, key in expert_patterns:
        groups[(l, proj)].append((int(e), key))

    stacked_count = 0
    for (l, proj), experts in sorted(groups.items()):
        experts.sort(key=lambda x: x[0])
        num_experts = len(experts)

        # Collect weights, scales, biases for each expert
        weight_key = f"model.layers.{l}.mlp.experts.0.{proj}.weight"
        base_key = weight_key[:-len(".weight")]
        scales_key = f"{base_key}.scales"
        biases_key = f"{base_key}.biases"

        if weight_key not in all_tensors:
            continue

        w0 = all_tensors[weight_key]
        s0 = all_tensors.get(scales_key)
        b0 = all_tensors.get(biases_key)

        # Stack weights along axis 0
        weights_list = [all_tensors[f"model.layers.{l}.mlp.experts.{e}.{proj}.weight"] for e in range(num_experts)]
        stacked_w = torch.cat(weights_list, dim=0)

        # Stack scales and biases if present
        stacked_s = None
        stacked_b = None
        if s0 is not None:
            scales_list = [all_tensors.get(f"model.layers.{l}.mlp.experts.{e}.{proj}.scales") for e in range(num_experts)]
            if all(s is not None for s in scales_list):
                stacked_s = torch.cat(scales_list, dim=0)
        if b0 is not None:
            biases_list = [all_tensors.get(f"model.layers.{l}.mlp.experts.{e}.{proj}.biases") for e in range(num_experts)]
            if all(b is not None for b in biases_list):
                stacked_b = torch.cat(biases_list, dim=0)

        # Reshape from 2D concatenated to 3D (num_experts, rows, cols)
        # MLX-LM expects 3D expert tensors, matching mx.stack() output in sanitize()
        num_experts = len(weights_list)
        w_rows, w_cols = stacked_w.shape
        stacked_w = stacked_w.view(num_experts, w_rows // num_experts, w_cols)

        # Write stacked tensors
        target_weight = f"model.layers.{l}.mlp.switch_mlp.{proj}.weight"
        all_tensors[target_weight] = stacked_w
        target_base = target_weight[:-len(".weight")]
        if stacked_s is not None:
            s_rows, s_cols = stacked_s.shape
            stacked_s = stacked_s.view(num_experts, s_rows // num_experts, s_cols)
            all_tensors[f"{target_base}.scales"] = stacked_s
        if stacked_b is not None:
            b_rows, b_cols = stacked_b.shape
            stacked_b = stacked_b.view(num_experts, b_rows // num_experts, b_cols)
            all_tensors[f"{target_base}.biases"] = stacked_b

        # Remove individual expert tensors
        for e in range(num_experts):
            for suffix in [".weight", ".scales", ".biases"]:
                k = f"model.layers.{l}.mlp.experts.{e}.{proj}{suffix}"
                all_tensors.pop(k, None)

        stacked_count += 1
        print(f"  stacked {num_experts} experts for layer {l} {proj}")

    # Save updated tensors back, replacing the original shards
    if stacked_count > 0:
        # Remove the original per-expert shards to avoid duplicates
        for shard in output_dir.glob("model-*-of-*.safetensors"):
            shard.unlink()
        # Save consolidated file
        out_path = output_dir / "model-00001-of-00001.safetensors"
        save_file(all_tensors, str(out_path))
        print(f"  saved stacked experts to {out_path.name}")


def _rewrite_config_for_stacked_experts(output_dir: Path, assignment: dict[str, int]):
    """Rewrite config.json to use switch_mlp keys for stacked experts."""
    import json
    cfg_path = output_dir / "config.json"
    if not cfg_path.exists():
        return
    with open(cfg_path) as f:
        cfg = json.load(f)
    quant = cfg.get("quantization", {})
    # Remove individual expert entries
    to_remove = [k for k in quant if ".mlp.experts." in k]
    for k in to_remove:
        del quant[k]
    # Add switch_mlp entries
    added = 0
    for path, bits in assignment.items():
        if ".mlp.experts." not in path:
            continue
        parts = path.split(".")
        l, proj = parts[2], parts[6]
        switch_key = f"model.layers.{l}.mlp.switch_mlp.{proj}"
        if switch_key not in quant:
            quant[switch_key] = {"group_size": 64, "bits": bits, "mode": "affine"}
            added += 1
    cfg["quantization"] = quant
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  rewrote config: removed {len(to_remove)} expert entries, added {added} switch_mlp entries")


def _protect_outliers(assignment: dict[str, int], error_db: dict, target_bpw: float, percentile: float = 90.0):
    """Auto-bump layers with error above the given percentile at target bitwidth.

    Reads the error_db dict, computes the error at target_bpw for each layer,
    identifies the top ``percentile``% most error-prone layers, and bumps
    them by one rung (if possible). This prevents quality cliffs on
    outlier layers without manual tuning.
    """
    layers = error_db

    # Map layer_path -> error at target_bpw
    errors = {}
    for path, bw_data in layers.items():
        bw_key = str(int(target_bpw))
        if bw_key in bw_data:
            errors[path] = bw_data[bw_key].get("error", 0.0)

    if len(errors) < 10:
        print("  [outliers] too few layers to detect outliers")
        return

    # Compute percentile threshold
    err_values = np.array(list(errors.values()))
    threshold = float(np.percentile(err_values, percentile))

    # Find outliers and bump them
    bumped = 0
    for path, err in errors.items():
        if err > threshold:
            cur = assignment.get(path)
            if cur is not None and cur < 8:
                assignment[path] = cur + 1
                bumped += 1

    if bumped > 0:
        print(f"  [outliers] protected {bumped} layers above {percentile:.0f}th percentile (threshold={threshold:.6f})")


def main():
    ap = argparse.ArgumentParser(
        description="RCO mixed-precision quantization pipeline for MLX",
    )
    ap.add_argument("--source", required=True, help="BF16 source model dir")
    ap.add_argument("--output", required=True, help="Output quantized model dir")
    ap.add_argument("--target-bpw", type=float, required=True, help="Target bits per weight")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--default-bits", type=int, default=4)
    ap.add_argument("--group-spec", default=None, help="Layer grouping spec (see build_layer_groups)")
    ap.add_argument("--loss-table", default=None, help="Pre-built loss table JSON (skip imatrix)")
    ap.add_argument("--imatrix", default=None, help="Imatrix GGUF path")
    ap.add_argument("--gguf-dir", default=None, help="GGUF dir for dequantization")
    ap.add_argument("--layers", default="0,8,16,24,32,40,47", help="Layers to sweep")
    ap.add_argument("--which", default="ffn_down_exps,ffn_gate_exps,attn_qkv")
    ap.add_argument("--bits", default="2,3,4,6,8")
    ap.add_argument("--method", default="dp", choices=["dp", "greedy"],
                    help="Allocation method: 'dp' (RCO knapsack, default) or 'greedy' (SLQ-style)")
    ap.add_argument("--bf16-prefixes", default="",
                    help="Comma-separated module path prefixes to keep BF16 (skip in quant config)")
    ap.add_argument("--model-type", default=None,
                    help="Model type for adapter-based quantization (e.g. 'flint')")
    ap.add_argument("--skip-quantize", action="store_true", help="Allocate + validate only")
    ap.add_argument("--rtn", action="store_true",
                    help="Uniform round-to-nearest quantization at --default-bits (no RCO allocation)")
    ap.add_argument("--error-db", default=None,
                    help="SLQ-style error_db.json path (skips imatrix/GGUF entirely)")
    ap.add_argument("--gptq-errors", action="store_true",
                    help="Build error_db with Hessian-weighted error measurement (requires --calibration)")
    ap.add_argument("--calibration", default=None,
                    help="Path to calibration token IDs .npy file for Hessian-weighted error measurement")
    ap.add_argument("--outlier-percentile", type=float, default=90.0,
                    help="Percentile threshold for outlier auto-protection (default: 90)")
    args = ap.parse_args()

    # Resolve model adapter
    model_adapter = None
    if args.model_type:
        from rco_mlx.model_adapters import get
        adapter_cls = get(args.model_type)
        if adapter_cls is None:
            sys.exit(f"Unknown model type: {args.model_type}")
        model_adapter = adapter_cls()
        print(f"Using model adapter: {args.model_type}")

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Pre-load error_db if provided (used by outlier protection + expert unification)
    error_db = None
    if args.error_db or args.gptq_errors:
        db_path = Path(args.error_db) if args.error_db else (output / "work" / "error_db_gptq.json")
        db_path = db_path.expanduser()
        if db_path.exists():
            import json as _json
            error_db = _json.loads(db_path.read_text()).get("layers", {})

    # --- Step 1: introspect ---
    print("[1/5] introspecting model …")
    layer_names, param_counts, groups = introspect_model(source, args.group_spec)
    group_names = [g.layer_names[0] for g in groups]  # one representative per group
    print(f"  {len(layer_names)} layers, {len(groups)} groups")

    # --- RTN short-circuit ---
    if args.rtn:
        print(f"[2/5] RTN mode: uniform {args.default_bits}-bit quantization")
        assignment = {name: args.default_bits for name in layer_names if _is_quantizable(name + ".weight")}
        print(f"  {len(assignment)} tensors @ {args.default_bits}-bit")
        # Skip to emit + quantize
        print("[3/5] skipping allocation (RTN)")
        print("[4/5] emitting MLX config + validating …")
        bf16_prefixes = None
        if model_adapter is not None:
            bf16_prefixes = model_adapter.non_quantized_prefixes()
        cfg = emit_and_validate(
            assignment, args.group_size, source, output, args.default_bits,
            validate_on_disk=False,
            bf16_prefixes=bf16_prefixes,
        )
        print("[5/5] streaming quantization …")
        stream_quantize(source, output, assignment, args.group_size, model_adapter=model_adapter)
        print(f"\n{'='*60}")
        print(f"RTN complete. Output: {output}")
        print(f"Config: {output / 'config.json'}")
        print(f"{'='*60}")
        return

    # --- Step 2: loss table ---
    loss_table_path = Path(args.loss_table) if args.loss_table else output / "work" / "imatrix_alloc.json"

    if args.error_db:
        error_db_path = Path(args.error_db).expanduser()
        if not error_db_path.exists():
            sys.exit(f"error_db not found: {error_db_path}")
        print(f"[2/5] loading SLQ error_db: {error_db_path}")
        losses, fracs, bitwidths = load_error_db(error_db_path)
        # Also save as the canonical loss table for downstream steps
        loss_table_path.parent.mkdir(parents=True, exist_ok=True)
        bitwidths_list = bitwidths.tolist()
        rows = []
        for i in range(len(fracs)):
            rows.append({
                "layer": f"group_{i}",
                "tensor": f"group_{i}",
                "params": int(fracs[i] * 1e6),
                "mse": {str(b): float(losses[i, j]) for j, b in enumerate(bitwidths_list)},
            })
        loss_table_path.write_text(json.dumps(rows, indent=2))
        print(f"  {len(rows)} groups, bitwidths={bitwidths_list}")

    elif args.gptq_errors:
        # Build error_db using Hessian-weighted error measurement
        if not args.calibration:
            sys.exit("--gptq-errors requires --calibration (path to .npy token IDs)")
        calib_path = Path(args.calibration)
        if not calib_path.exists():
            sys.exit(f"Calibration file not found: {calib_path}")
        print(f"[2/5] building Hessian-weighted error_db with calibration: {calib_path}")
        hessian_db_path = output / "work" / "error_db_hessian.json"
        hessian_db_path.parent.mkdir(parents=True, exist_ok=True)
        import subprocess
        script = (
            Path(__file__).parent.parent / "src" / "rco_mlx" / "scripts" / "build_error_db.py"
        )
        cmd = [
            sys.executable, str(script),
            "--source", str(source),
            "--output", str(hessian_db_path),
            "--model-type", args.model_type or "default",
            "--calibration", str(calib_path),
            "--bitwidths", args.bits,
            "--group-size", str(args.group_size),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"Hessian error_db build failed:\n{result.stderr}")
        print(result.stdout)
        # Use the Hessian error_db as the error source
        error_db_path = hessian_db_path
        losses, fracs, bitwidths = load_error_db(error_db_path)
        loss_table_path.parent.mkdir(parents=True, exist_ok=True)
        bitwidths_list = bitwidths.tolist()
        rows = []
        for i in range(len(fracs)):
            rows.append({
                "layer": f"group_{i}",
                "tensor": f"group_{i}",
                "params": int(fracs[i] * 1e6),
                "mse": {str(b): float(losses[i, j]) for j, b in enumerate(bitwidths_list)},
            })
        loss_table_path.write_text(json.dumps(rows, indent=2))
        print(f"  {len(rows)} groups, bitwidths={bitwidths_list}")

    elif not loss_table_path.exists():
        if not args.imatrix or not args.gguf_dir:
            sys.exit("Need --loss-table, --error-db, or both --imatrix and --gguf-dir")
        build_loss_table_from_imatrix(
            imatrix_path=Path(args.imatrix).expanduser(),
            gguf_dir=Path(args.gguf_dir).expanduser(),
            layers=[int(x) for x in args.layers.split(",")],
            which=args.which.split(","),
            bits_list=[int(x) for x in args.bits.split(",")],
            group_size=args.group_size,
            out=loss_table_path,
        )
        losses, fracs, bitwidths = load_loss_table(loss_table_path)
    else:
        print(f"[2/5] using existing loss table: {loss_table_path}")
        losses, fracs, bitwidths = load_loss_table(loss_table_path)

    # --- Step 3: allocation ---
    print(f"[3/5] allocation  method={args.method} target={args.target_bpw} bpw …")
    if args.method == "greedy":
        assignment, report = solve_allocation_greedy(
            losses, fracs, bitwidths, args.target_bpw, group_names
        )
    else:
        assignment, report = run_rco_allocation(
            losses, fracs, bitwidths, args.target_bpw, args.group_size, group_names
        )
    print(f"  achieved {report['achieved_bpw']:.4f} bpw")
    print(f"  histogram: {report['bit_histogram']}")

    # Outlier auto-protection: bump layers with high error at target bitwidth
    if error_db is not None:
        _protect_outliers(
            assignment,
            error_db,
            args.target_bpw,
            percentile=args.outlier_percentile,
        )

    # For models with MoE experts, unify bitwidths per (layer, projection)
    # so that all experts in a group share the same bitwidth — required
    # for stacking into switch_mlp tensors.
    if any(".mlp.experts." in p for p in assignment):
        print("  unifying MoE expert bitwidths per layer-projection ...")
        from collections import Counter, defaultdict

        expert_groups = defaultdict(list)
        for path, bits in assignment.items():
            if ".mlp.experts." in path:
                parts = path.split(".")
                l, proj = parts[2], parts[6]
                expert_groups[(l, proj)].append(bits)
        # Infer the number of experts per layer from the assignment keys
        experts_per_layer: Dict[str, int] = {}
        for path in assignment:
            if ".mlp.experts." in path:
                parts = path.split(".")
                l, e, proj = parts[2], int(parts[5]), parts[6]
                key = f"{l}.{proj}"
                experts_per_layer[key] = max(experts_per_layer.get(key, 0), e + 1)

        # If an error_db is available, use it to pick the bitwidth that
        # minimizes aggregate error across experts. Otherwise fall back
        # to the mode (most common bitwidth).
        unified = 0
        for (l, proj), bits_list in expert_groups.items():
            if error_db is not None:
                # Pick bitwidth minimizing aggregate error across experts
                best_bw = None
                best_err = float("inf")
                for candidate_bw in sorted(set(bits_list)):
                    total_err = 0.0
                    count = 0
                    for e in range(experts_per_layer.get(f"{l}.{proj}", 0)):
                        expert_key = f"model.layers.{l}.mlp.experts.{e}.{proj}"
                        layer_data = error_db.get(expert_key, {})
                        bw_data = layer_data.get(str(candidate_bw))
                        if bw_data is not None:
                            total_err += bw_data.get("error", 0.0)
                            count += 1
                    if count > 0 and total_err < best_err:
                        best_err = total_err
                        best_bw = candidate_bw
                if best_bw is not None:
                    mode_bits = best_bw
                else:
                    mode_bits = Counter(bits_list).most_common(1)[0][0]
            else:
                mode_bits = Counter(bits_list).most_common(1)[0][0]

            n_experts = experts_per_layer.get(f"{l}.{proj}", 0)
            for e in range(n_experts):
                key = f"model.layers.{l}.mlp.experts.{e}.{proj}"
                if key in assignment and assignment[key] != mode_bits:
                    assignment[key] = mode_bits
                    unified += 1
        print(f"    unified {unified} expert assignments across {len(expert_groups)} groups")

    # Filter out BF16 tensors from the quant config
    if args.bf16_prefixes:
        bf16_prefixes = [p.strip() for p in args.bf16_prefixes.split(",") if p.strip()]
        assignment = filter_bf16_prefixes(assignment, bf16_prefixes)
        print(f"  filtered to {len(assignment)} quantized entries ({len(report['bit_histogram'])} unique bitwidths before filter)")

    # save allocation
    alloc_path = output / "work" / "allocation.json"
    alloc_path.parent.mkdir(parents=True, exist_ok=True)
    alloc_path.write_text(json.dumps(assignment, indent=2, sort_keys=True))
    print(f"  wrote {alloc_path}")

    # --- Step 4: emit + validate config ---
    print("[4/5] emitting MLX config + validating …")
    bf16_prefixes = None
    if model_adapter is not None:
        bf16_prefixes = model_adapter.non_quantized_prefixes()
    cfg = emit_and_validate(
        assignment, args.group_size, source, output, args.default_bits,
        validate_on_disk=args.skip_quantize,
        bf16_prefixes=bf16_prefixes,
    )

    # --- Step 5: quantize ---
    if args.skip_quantize:
        print("[5/5] --skip-quantize set, done.")
        return

    print("[5/5] streaming quantization …")
    stream_quantize(source, output, assignment, args.group_size, model_adapter=model_adapter)

    print(f"\n{'='*60}")
    print(f"Pipeline complete. Output: {output}")
    print(f"Config: {output / 'config.json'}")
    print(f"{'='*60}")

    # Summary report
    _print_summary(assignment, output, args.group_size, args.target_bpw, error_db if args.error_db else None)


def _print_summary(assignment, output, group_size, target_bpw, error_db):
    """Print a final summary of the quantization."""
    from collections import Counter

    bit_hist = Counter(assignment.values())
    total_params = 0
    for path, bits in assignment.items():
        # Estimate params from layer name (rough)
        total_params += 1

    print(f"\n{'='*60}")
    print("QUANTIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Target: {target_bpw:.2f} bpw")
    print(f"Group size: {group_size}")
    print(f"Quantized tensors: {len(assignment)}")
    print(f"Bitwidth distribution:")
    for bw in sorted(bit_hist.keys(), reverse=True):
        print(f"  {int(bw)}-bit: {bit_hist[bw]} tensors")

    if error_db is not None:
        # Compute aggregate error at assigned bitwidths
        total_err = 0.0
        count = 0
        for path, bits in assignment.items():
            bw_data = error_db.get(path, {}).get(str(int(bits)))
            if bw_data is not None:
                total_err += bw_data.get("error", 0.0)
                count += 1
        if count > 0:
            print(f"Mean relative error: {total_err / count:.6f}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
