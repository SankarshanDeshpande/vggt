# Results Table

Every row below is a pipeline-validation or qualitative result, not a paper-scale reproduction —
see `deviations.md` #1–6 for why. Treat this table as "evidence the pipeline works correctly,"
not "VGGT retrained from scratch."

## Training — CO3D `apple`, single-sequence subset (3 sequences, 202 frames each)

Config: `training/config/default.yaml`, `split: test` (both train/val), `debug: True`,
`limit_train_batches: 10`, `limit_val_batches: 5`, `max_epochs: 1`, checkpoint resumed from
VGGT-1B (`model.pt`), single H100 shard, in-kernel execution (no `torchrun`).

### Train (first / last step of the single epoch)

| Step | loss_objective | loss_camera | loss_T | loss_R | loss_FL | loss_conf_depth | loss_reg_depth | loss_grad_depth | Grad/depth | Grad/camera |
|---|---|---|---|---|---|---|---|---|---|---|
| 0  | 0.2370 | 0.0379 | 0.0237 | 0.0032 | 0.0220 | 0.2221 | 0.0600 | 0.0025 | 55.7748 | 1.2495 |
| 10 | 0.1776 | 0.0445 | 0.0166 | 0.0051 | 0.0457 | 0.0622 | 0.0630 | 0.0073 | 26.3357 | 1.3371 |

Checkpoint saved: `training/logs/exp001/ckpts/checkpoint.pt` (epoch 0).

### Validation (first step of the single val pass)

| Step | loss_camera | loss_T | loss_R | loss_FL | loss_conf_depth | loss_reg_depth | loss_grad_depth |
|---|---|---|---|---|---|---|---|
| 0 | 0.0223 | 0.0088 | 0.0019 | 0.0231 | 0.0664 | 0.0666 | 0.0057 |

**Interpretation:** `Grad/aggregator: 0.0000` throughout — correct, matches `frozen_module_names:
["*aggregator*"]` in the fine-tuning config. Non-zero, reasonable-magnitude camera/depth gradients
confirm backprop through the unfrozen heads is functioning. Loss values move but should **not** be
read as a converged/trained result (deviations.md #5).

---

## Inference — qualitative reconstructions

All using the local VGGT-1B checkpoint (base, not the CO3D-apple fine-tune above — see note in
each row), `img_load_resolution=1024`, `vggt_fixed_resolution=518`.

| Scene | Source | # frames used | Filtering | Result | Artifact |
|---|---|---|---|---|---|
| CO3D apple (`110_13051_23361`) | Local CO3D single-sequence subset | 202 (full orbit) | CO3D object mask (correctly aligned, see deviations.md) + confidence | Clean, correctly-oriented apple reconstruction incl. stem; verified from side/top/¾ views with equal-aspect axes | `vggt_pointcloud_co3d.ply` |
| `kitchen` (repo example) | `vggt/examples/kitchen` | 25 | Confidence only (no mask — full scene) | Coherent kitchen layout (counter, cabinets, wall) recognisable from multiple angles | `vggt_pointclouds/pyramid_*` naming convention (kitchen renders predate the naming-convention change) |
| `room` / no-overlap (repo example) | `vggt/examples/room` | 8 (`no_overlap_1..8`) | Confidence only | **Correctly fragments** into disjoint clusters — expected behaviour for a deliberately minimal-overlap stress test, not a bug (see research_log Phase C) | `room.ply` |
| `single_oil_painting` (repo example) | `vggt/examples/single_oil_painting` | — | — | Confirmed as the likely exact Figure-3-row-1 source image bundled with the repo | not yet rendered in this log |
| Pyramid (`videos/pyramid.mp4`) | 34.5s video, frame-extracted | 32 (early) → 96 (later) | Confidence (abs. threshold, tuned 5.0→1.5 across iterations) + k-NN outlier removal | Early renders had inverted convexity (reflection bug) and hollow appearance (no depth-sorting); final z-buffered render shows a solid, correctly-oriented pyramid with visible surface texture, close to paper Fig. 3 framing | `vggt_pointclouds/pyramid_final.png`, `pyramid.ply` |

---

## Not yet run

- Quantitative reproduction of any paper table (Table 1: camera pose AUC@30 on CO3Dv2/RealEstate10K;
  Table 2: DTU MVS; Table 3: ETH3D point map; Table 4: ScanNet two-view matching). Current work is
  qualitative pipeline validation only.
- Replica room-scale training/inference (download in progress as of Phase D).
- Own room video through the full pipeline.
- ConceptGraphs downstream stage (not yet started).
