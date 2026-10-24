"""Worked example: allocate bits for a mixed-precision MLX quant.

Run with:  python examples/allocate.py

This is a synthetic illustration of the pipeline -- it uses a made-up loss
table rather than measured losses. To use it for real, replace `measure_losses`
with a pass over calibration data that quantizes each group in turn and records
the resulting loss (KL against a reference, or perplexity).

The point of the example is the shape of the pipeline:

    group the layers -> measure per-group losses -> solve the knapsack
    -> emit an MLX config -> validate it before loading
"""

import json

import numpy as np

from rco_mlx import (
    LossTable,
    allocation_to_config,
    build_layer_groups,
    effective_bits,
    validate_allocation,
)


def measure_losses(group_names, bitwidths, group_size):
    """Stand-in for a real calibration sweep.

    Returns (n_groups, n_options) losses. A real implementation would, for each
    group and each candidate width, quantize that group and evaluate the model
    on calibration data -- the same measurement loop oQ and the streaming
    quantizer use.
    """
    n_groups = len(group_names)
    n_options = len(bitwidths)
    rng = np.random.default_rng(0)

    # later layers and wider groups are more sensitive, as in practice
    depth = np.linspace(0.0, 1.0, n_groups)
    base = 0.05 + 0.25 * depth

    losses = np.zeros((n_groups, n_options))
    for j, bits in enumerate(bitwidths):
        # error falls roughly as 2^-bits, with per-group noise
        losses[:, j] = base * (2.0 ** -(bits - 2)) * (1.0 + 0.2 * rng.random(n_groups))

    return losses


def main():
    # A hybrid MoE in MLX naming: 8 attention layers, 4 MoE layers.
    layer_names = []
    for i in range(8):
        layer_names += [
            f"backbone.layers.{i}.mixer.q_proj",
            f"backbone.layers.{i}.mixer.k_proj",
            f"backbone.layers.{i}.mixer.v_proj",
            f"backbone.layers.{i}.mixer.o_proj",
        ]
    for i in range(8, 12):
        layer_names += [
            f"backbone.layers.{i}.mixer.switch_mlp.fc1",
            f"backbone.layers.{i}.mixer.switch_mlp.fc2",
            f"backbone.layers.{i}.mixer.shared_experts.up_proj",
            f"backbone.layers.{i}.mixer.shared_experts.down_proj",
        ]
    layer_names += ["backbone.embeddings", "lm_head"]

    group_size = 64
    bitwidths = np.array([2, 3, 4, 6, 8])

    # 1. group the layers -- one group per layer here, because MLX stacks
    #    experts into a single tensor and cannot express per-expert widths.
    groups = build_layer_groups(layer_names)
    group_names = [g.layer_names[0] for g in groups]

    # 2. parameter fractions (group_params / total_params)
    param_counts = {name: 1_000_000 for name in layer_names}
    fracs = np.array([param_counts[n] for n in group_names], dtype=np.float64)
    fracs /= fracs.sum()

    # 3. measure losses
    losses = measure_losses(group_names, bitwidths, group_size)

    # 4. solve
    table = LossTable(
        losses=losses,
        fracs=fracs,
        bitwidths=bitwidths,
        group_size=group_size,
    )
    target = 4.0
    bits = table.assignment(target_avg_bits=target)
    report = table.report(target_avg_bits=target)

    print(f"target bpw      : {report['target_bpw']}")
    print(f"achieved bpw    : {report['achieved_bpw']:.4f}")
    print(f"groups          : {report['n_groups']}")
    print(f"bit histogram   : {report['bit_histogram']}")
    print()

    # 5. emit the MLX config
    cfg = allocation_to_config(
        dict(zip(group_names, bits)), default_bits=4, group_size=group_size
    )
    print(f"declared paths  : {len(cfg)}")
    print(f"effective widths: {[round(effective_bits(int(b), group_size), 4) for b in bitwidths]}")
    print()

    # 6. validate -- in a real run, pass the checkpoint's tensor keys here
    print("config block:")
    print(json.dumps(cfg.to_dict(), indent=2)[:400] + "\n  ...")

    # with the keys that would be on disk
    keys = [f"{name}.scales" for name in group_names]
    keys += [f"{name}.weight" for name in group_names]
    result = validate_allocation(cfg, on_disk_keys=keys)
    print()
    print(result.summary())


if __name__ == "__main__":
    main()
