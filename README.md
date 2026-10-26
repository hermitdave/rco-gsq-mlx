# rco-mlx

RCO's budget-constrained mixed-precision allocator, ported from PyTorch to
NumPy, with the result wired into MLX's quantized-checkpoint format.

RCO ([ISTA-DASLab/RCO](https://github.com/IST-DASLab/RCO), *Model Compression
with Exact Budget Constraints via Riemannian Manifolds*, arXiv:2605.00649)
answers a specific question: **given a set of groups (layers), a set of
options per group (bit widths), and a hard size budget, which option do you
pick for each group?** It does so with an exact equality constraint rather than
a penalty or a Lagrangian, which is why the resulting file lands on the
requested size instead of near it.

This package ports that optimizer so it runs anywhere NumPy does, then emits
the allocation as an MLX `config["quantization"]` block and validates it before
you commit to a 15-minute checkpoint load.

---

## What is here, and how faithful it is

| Component | Source | Status |
|---|---|---|
| `manifold.py` | `src/manifold.py` | Faithful port. Verified against a finite difference of the budget function. |
| `search.budget_dp` | `round_with_budget_dp` | Faithful port of the knapsack DP. Checked against a brute-force enumeration. |
| `search.ProjectedGumbelOptimizer` | `optimize_projected_gumbel` | Faithful algorithm, different interface (see below). |
| `grouping.py` | `src/grouping.py` | Transcription; the original has no torch dependency either. |
| `objective.py` | — | New. Shapes a measured loss table for the DP. |
|| `mlx_alloc.py` | — | New. Emits the MLX config block. |
|| `validate.py` | — | New. Structural checks before a load. |
|| `scripts/build_error_db.py` | — | New. Builds a SLQ-style `error_db.json` from BF16 safetensors. |

**The one design change that matters.** The original drives the optimizer from
a live PyTorch model: it registers a parametrization on every layer's weight,
runs a forward pass, and backprops into the allocation logits. That is what
makes RCO handle a *non-decomposable* objective — a real model's loss does not
factor into a sum over layers, and RCO's paper exists because of that.

This port replaces the model with a `loss_and_grad` callable:

```python
def loss_and_grad(probs):
    loss, grad = my_backend.forward_and_backward(probs)
    return loss, grad

optimizer.run(loss_and_grad, n_steps=400)
```

Supply that from MLX (`mx.grad` over a function of `probs`), PyTorch, or
anything else. The manifold math is unchanged, so the budget still holds by
construction.

**If you only have a loss table** — which is the practical MLX case, and what
`alloc_greedy.py` already produces — skip the optimizer and call `budget_dp`
directly. For a separable objective the DP is exact, and it is strictly better
than greedy because it solves the knapsack rather than taking the local best.
The caveat is that a table *is* the separable assumption; treat it as a strong
heuristic, not the true objective.

---

## Install

```bash
pip install -e .
```

NumPy only. MLX is an optional extra (`pip install -e ".[mlx]"`) and is only
needed for quantization utilities.

---

## Quick start

### From a loss table

```python
import numpy as np
from rco_mlx import LossTable, allocation_to_config, validate_allocation

# losses[group, option] -- measured, lower is better
losses = np.array([
    [0.50, 0.30, 0.10],   # group 0 is sensitive
    [0.45, 0.28, 0.12],   # group 1 is sensitive
    [0.20, 0.15, 0.10],   # group 2 is not
    [0.18, 0.14, 0.11],   # group 3 is not
])
fracs = np.array([0.25, 0.25, 0.25, 0.25])       # group_params / total_params
bitwidths = np.array([2, 4, 8])

table = LossTable(losses=losses, fracs=fracs, bitwidths=bitwidths, group_size=64)
bits = table.assignment(target_avg_bits=5.0)      # -> per-group bit widths

cfg = allocation_to_config(dict(zip(group_names, bits)), default_bits=4)
report = validate_allocation(cfg, on_disk_keys=checkpoint_keys)
assert report.ok, report.summary()
```

### From a live, differentiable objective

```python
from rco_mlx import ProjectedGumbelOptimizer

opt = ProjectedGumbelOptimizer(
    n_groups=128, n_options=5,
    fracs=fracs, actual_bits=np.array([2.25, 3.25, 4.25, 6.25, 8.25]),
    target_avg_bits=4.0,
)
opt.init_to_bits()          # start on the manifold, not off it

probs, history = opt.run(loss_and_grad, n_steps=400)
assignment = opt.hard_assignment()
```

---

## Using across different models

The pipeline is model-agnostic once you provide a **model adapter** that
describes the architecture. The adapter tells the pipeline:

|- how to enumerate layers and their types
|- which tensors to quantize vs keep BF16
|- how to group layers for RCO allocation
|- how to load calibration data for error measurement
|- how to map source checkpoint keys to MLX module paths

Two adapters are included:

| Adapter | Model type | Architecture |
|---|---|---|
| `LFM2Adapter` | `lfm2` | Conv + full-attention hybrid (LFM2.5) |
| `FlintAdapter` | `flint` | Hybrid MoE with dense attention layers |

### Adding a new model adapter

Subclass `ModelAdapter` and implement the six methods. The smallest useful
adapter is ~40 lines — `LFM2Adapter` is the template.

```python
from rco_mlx import ModelAdapter, register

@register("my_model")
class MyAdapter(ModelAdapter):
    model_type = "my_model"
    default_group_size = 64
    default_bits = 4

    def enumerate_layers(self, source_dir):
        # Read config.json + safetensors index
        # Return [{"index": 0, "type": "dense", "weight_keys": [...], "param_count": N}, ...]
        ...

    def quant_predicate(self, layer_descriptor):
        # Return {"bits": 4, "group_size": 64} or None for BF16
        ...

    def grouping_spec(self):
        # Return "q_proj,k_proj,v_proj,o_proj mlp.gate_proj,mlp.up_proj,mlp.down_proj"
        # or None for per-layer groups
        ...

    def load_calibration(self, source_dir, n_samples=512, seq_len=2048):
        # Return (n_samples, seq_len, hidden_dim) float32 numpy array
        ...

    def map_key_to_mlx(self, key):
        # Map HF key to MLX path, return None to skip
        ...
```

Register it with `@register("my_model")` and the pipeline discovers it
automatically via `rco_mlx.get("my_model")`.

### Running the pipeline on a new model

```bash
python examples/pipeline_rco_mlx.py \
    --source /path/to/bf16-model \
    --output /path/to/output \
    --target-bpw 4.0 \
    --group-size 64 \
    --model-type my_model
```

The `--model-type` flag selects the adapter. If omitted, the pipeline
infers it from `config.json`'s `model_type` field.

### Using a SLQ-style error database

Instead of a dummy or imatrix-based loss table, you can drive allocation from
a real measured error database:

```bash
python src/rco_mlx/scripts/build_error_db.py \
    --source /path/to/bf16-model \
    --output error_db.json \
    --model-type olmoe \
    --bitwidths 2,3,4,6,8 \
    --group-size 64

python examples/pipeline_rco_mlx.py \
    --source /path/to/bf16-model \
    --output /path/to/output \
    --target-bpw 4.0 \
    --group-size 64 \
    --model-type olmoe \
    --error-db error_db.json
```

The `--error-db` flag skips the imatrix/GGUF dependency entirely. The pipeline
loads the database, converts it to the same internal format as the legacy loss
table, and runs the same RCO allocation.

---

## MLX integration

The emitted config matches what `mlx_lm.load()` expects:

```json
{
  "quantization": {
    "group_size": 64,
    "bits": 4,
    "mode": "affine",
    "backbone.layers.1.mixer.switch_mlp.fc1": {"group_size": 64, "bits": 2, "mode": "affine"},
    "backbone.embeddings":                     {"group_size": 64, "bits": 8, "mode": "affine"}
  },
  "quantization_config": "<same object>"
}
```

Keys are module paths **without** `.weight`. Three rules the emitter enforces:

1. **List every quantized path.** A module with `.scales` on disk but absent
   from the map is quantized at the *default* width — a silent corruption, not
   a harmless omission.
2. **List nothing you did not quantize.** A listed module with no `.scales` on
   disk is quantized at load and then fails to find its weights.
3. **Never rely on the default to carry an intended width.** A lookup that
   falls back cannot distinguish "intentionally 4-bit" from "I forgot this
   one".

`validate_allocation` checks all three, plus that every declared width is one
MLX supports, before you start a multi-minute load.

### MLX width constraints

Verified by probing `mx.quantize` directly on mlx 0.32.3:

- **bits** ∈ {2, 3, 4, 5, 6, 8}. `bits=1` and `bits=7` are rejected, so **2 is
  the floor**. The 1-bit rungs of EXL3/GGUF low-bit ladders have no MLX
  equivalent.
- **group_size** ∈ {32, 64, 128}. `16` is rejected.

`effective_bits(bits, group_size)` returns `bits + 16 / group_size` — the
number to feed the budget constraint. Using the nominal width instead
understates the real cost by ~0.13–0.5 bpw, and the file comes out larger than
the target.

---

## Testing

```bash
python -m pytest
```

88 tests, pure NumPy + MLX, no GPU required. The manifold primitives are checked
against a finite difference of the budget function and the DP against a
brute-force enumeration, so the port is verified against first principles
rather than against the PyTorch original.

---

## Scope, honestly stated

- **No benchmarks yet.** This port has not been run against a quantized model,
  so there are no quality numbers here. The comparison that motivated it —
  RCO versus EXL3 — is a cross-paper inference, not a head-to-head.
- **The table path is a proxy.** It assumes separability. The optimizer path
  does not, but needs a differentiable backend to be useful.
- **The MLX affine format is the target.** MLX's levels are a uniform affine
  grid. The packing coincides (16 two-bit values per uint32, LSB-first)
  but the codebook is standard, so there is no transcoding step.
- **EXL3 is not ported and is not a good first target.** Its format has no MLX
  kernel: no trellis decode, no procedural codebook, no double-sided Hadamard,
  no fused MMA. Its encode path (Hessian + LDLQ + Viterbi) is CUDA-only. And
  MLX's 2-bit floor means its best rungs are unreachable regardless.

---

## Attribution

---

## OLMoE-1B-7B-0125 validation

We validated the pipeline on [allenai/OLMoE-1B-7B-0125](https://huggingface.co/allenai/OLMoE-1B-7B-0125), a 1B-parameter mixture-of-experts model with 64 experts per layer and top-8 routing.

**What changed:**
- Added `OLMoEAdapter` in `model_adapters.py` for native MLX-LM support
- Expert stacking now reshapes to 3D `(num_experts, rows, cols)` matching MLX-LM's `sanitize()` format
- Router (`mlp.gate`) preserved in BF16
- Pipeline post-processes quantized checkpoints to consolidate per-expert tensors into `switch_mlp` projections

**Validation results (4-bit mixed precision, target 4.0 bpw):**

| Method | PPL (wikitext2) | Degradation | 5-bit tensors |
|--------|-----------------|-------------|---------------|
| BF16 | 8.890 | — | — |
| RTN 4-bit (uniform) | 9.412 | +5.8% | 0 |
| RCO 4-bit (Frobenius error_db) | 9.468 | +6.5% | 36 |
| RCO 4-bit (Hessian-weighted) | 9.320 | +4.8% | 17 |

**Quantized checkpoint:** [hermitdave/OLMoE-1B-7B-0125-4bit](https://huggingface.co/hermitdave/OLMoE-1B-7B-0125-4bit)

```bash
# Hessian-weighted error_db (best quality):
python examples/pipeline_rco_mlx.py \
    --source /path/to/olmoe-bf16 \
    --output /path/to/output \
    --target-bpw 4.0 \
    --group-size 64 \
    --model-type olmoe \
    --gptq-errors \
    --calibration /path/to/calibration.npy

# Uniform RTN baseline:
python examples/pipeline_rco_mlx.py \
    --source /path/to/olmoe-bf16 \
    --output /path/to/output \
    --target-bpw 4.0 \
    --group-size 64 \
    --model-type olmoe \
    --rtn
```


RCO is © ISTA-DASLab (Michael Helcig, Dan Alistarh), Apache-2.0. The
optimizer, manifold primitives, and grouping logic here are a port of
[ISTA-DASLab/RCO](https://github.com/IST-DASLab/RCO) and remain Apache-2.0.

Port and MLX integration by **Hermit Dave**, produced with **Hermes Agent**
(Nous Research).

---

## License

Apache-2.0, matching the upstream sources.
