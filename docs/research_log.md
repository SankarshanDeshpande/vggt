# Research Log

Dated, append-only. Each entry records what was tried, what happened, and why. This is the source
for the thesis's "challenges faced" / methodology narrative.

---

## Phase A — Environment setup on HPC (svnithpc cluster, 2× H100 NVL nodes)

- Cluster: Slurm, partitions `gpu*`, QOS `normal`, MaxSubmit=2 jobs/user, GPU access via `shard:N`
  (fractional H100 allocation), not whole-GPU `gres=gpu:N`.
- Conda env `vlm` (Python 3.10) created under `~/.conda/envs/vlm`.
- **Key recurring bug:** `conda activate` silently fails inside non-interactive Slurm batch shells and
  inside `subprocess.Popen` calls from Jupyter — falls back to base env (Python 3.11), causing
  `libtorch_global_deps.so` / wrong-python errors. **Fix adopted:** bypass conda activation entirely in
  scripts and subprocess calls; invoke the env's Python by absolute path
  (`/home/mazaveri/.conda/envs/vlm/bin/python`) or via `sys.executable` from inside a live kernel.
- **Compute nodes have no internet access.** `pypi.org`, `huggingface.co`, `dl.fbaipublicfiles.com` all
  return `Network unreachable` / connection timeout from compute nodes. All `pip install` and dataset/
  checkpoint downloads must run from the **login node**, then are visible to compute-node jobs via the
  shared home filesystem.
- Jupyter-on-compute-node workflow: submit via `sbatch` (or `srun` for interactive), tunnel via
  `ssh -L <local_port>:<node>:<remote_port>` **or**, found more reliable, connect directly to the
  compute node's IP instead of tunnelling through the login node — SSH-tunnel drops correlated with
  job cancellations during sustained GPU compute (see incident below).
- **Incident — repeated job cancellations.** Multiple `sbatch`/interactive jobs (IDs 28421, 28423,
  28426, 28447, 28450) were killed by Slurm (`CANCELLED`) within 10s–5min specifically when sustained
  GPU compute (training) was running, while idle Jupyter sessions on the same account survived 45+
  minutes without issue. Tried: different nodes (node1 vs node2), `sbatch` vs `srun` vs in-kernel
  execution — all showed the same pattern. Root cause traced to the SSH tunnel through the login node,
  not a Slurm policy: connecting to the compute node's IP directly (bypassing the tunnel) resolved it.
  A full 11-step training epoch + validation completed cleanly once this was fixed.

## Phase B — VGGT training pipeline (facebookresearch/vggt, `training/` module)

- Installed VGGT repo + training deps (`hydra-core`, `omegaconf`, `fvcore`, `wcmatch`, `tensorboard`) —
  none were bundled in `requirements.txt`, each surfaced one at a time as the training script was run
  and fixed iteratively.
- Dataset chosen: **CO3Dv2**, `apple` category — see `docs/deviations.md` for why (only dataset with a
  ready-made `Dataset` class in the released training code, alongside VKitti).
- CO3D full-category download (~100GB for `apple` alone) blocked by compute-node network access;
  worked around by downloading the `--single_sequence_subset` (3 sequences, 202 frames each, with
  images/depths/depth_masks) on a local PC and transferring via `scp`.
- Annotation files (`apple_train.jgz`, `apple_test.jgz`) obtained separately from
  `JianyuanWang/co3d_anno` on Hugging Face (succeeded via `hf-xet` fast-transfer path even when the
  standard HTTP HEAD check failed — inconsistent network behaviour on compute vs login node).
- **Bug found and patched in `training/data/datasets/co3d.py`:** the dataset loader trusts the
  annotation file's sequence list without checking whether the sequence actually exists on disk —
  when using the single-sequence subset (only 3 of ~700 annotated sequences present locally), this
  produced silent `FileNotFoundError`s deep in the DataLoader. Patched to skip sequences not present
  on disk (`if not osp.isdir(seq_dir): continue`).
- **Bug found:** `single_sequence_subset` sequences happen to belong to the CO3D **test** split, not
  train — using `split: train` in the config produced an empty training dataset
  (`ValueError: empty range for randrange()`). Fixed by pointing both train and val dataset configs at
  `split: test` for this smoke-test run — see deviations.md, this is not representative of a real
  train/val split.
- VGGT-1B checkpoint (`model.pt`, ~5.03GB) downloaded on local PC, transferred via `scp` (same network
  restriction as CO3D).
- Config edits (`training/config/default.yaml`): `CO3D_DIR`/`CO3D_ANNOTATION_DIR` set to real paths,
  `debug: True` (restricts to `apple` category only, matching what's on disk), `limit_train_batches:
  10`, `limit_val_batches: 5`, `max_epochs: 1`, `resume_checkpoint_path` set to local checkpoint.
- **Result:** full training epoch (10 batches) + validation (6 batches) completed successfully on a
  single H100 shard. Real loss values recorded (see `results/table.md`), checkpoint saved to
  `training/logs/exp001/ckpts/checkpoint.pt`. This confirms the released training code, environment,
  and data pipeline all work correctly end-to-end on this cluster.

## Phase C — VGGT inference pipeline

- Built cell-by-cell Jupyter inference pipeline directly from `demo_colmap.py`'s `run_VGGT()` function
  (aggregator → camera_head → depth_head → `unproject_depth_map_to_point_map`), since running the
  demo scripts as subprocesses hit the same conda/subprocess issues as training.
- Tested on: CO3D apple (202-frame orbit, single-sequence subset), VGGT repo's bundled example scenes
  (`kitchen`, `room`/no_overlap, `single_oil_painting`), and `videos/pyramid.mp4` (frame-extracted).
- **Bug found and fixed — coordinate reflection:** initial "fix" for an apple point cloud rendering
  upside-down was to negate the Y axis before plotting (`-points[:,1]`). This is a *reflection*, not a
  rotation — invisible on a near-spherical object (apple) but inverts convexity on a non-symmetric
  object (pyramid rendered as a hollow crater instead of a solid peak). Fixed by plotting true,
  unmirrored coordinates and controlling orientation only via camera `elev`/`azim`.
- **Bug found and fixed — unmasked padding:** `load_and_preprocess_images_square` centre-pads each
  image to a square with black fill before resizing. This padding was never excluded from the point
  cloud — VGGT predicts (meaningless) depth for solid black regions, and those points were being
  plotted alongside real geometry. Fixed using the `original_coords` the loader already returns for
  exactly this purpose, to build a per-frame valid-region mask.
- **Bug found and fixed — mask/image misalignment (CO3D apple case):** CO3D object masks were
  initially resized directly onto the target resolution without replicating the image loader's
  pad-then-resize steps, causing mask edges to be offset from the actual object silhouette. Fixed by
  writing a matching `load_and_preprocess_mask_square()` that mirrors the image preprocessing exactly.
- **Bug found and fixed — no depth sorting in visualisation:** initial matplotlib
  `Axes3D.scatter`-based rendering paints points in array order, not depth order, so far points can
  occlude near ones — this produced a "see-through"/hollow appearance on non-convex objects. Replaced
  with a custom NumPy z-buffer renderer (project → per-pixel nearest-point selection → colour) that
  correctly handles occlusion.
- **Filtering approach revised:** switched from a percentile-based confidence threshold
  (`np.percentile(conf, 50)`, which always keeps the bottom half of confidence values regardless of
  overall quality) to the repo's own absolute default (`conf_thres_value = 5.0`, loosened to 3.0 for
  sparser scenes), plus a k-NN based statistical outlier removal pass (`scipy.spatial.cKDTree`) to
  remove floating noise fragments.
- **Observation — sparse/no-overlap scenes fragment, and this is expected.** The repo's bundled
  `room` example (`no_overlap_1..8.jpg`) is explicitly a wide-baseline, minimal-overlap stress test.
  Reconstruction correctly fragments into disjoint clusters rather than forming one coherent room —
  this matches the paper's own framing of non-overlapping frames as a hard/out-of-domain case, not a
  pipeline bug. Confirmed the same tool produces a clean, continuous result on the dense-overlap
  `kitchen` scene, isolating the cause to input coverage rather than code.
- All 7-required-edits checklist (paths, debug mode, batch limits, epochs, checkpoint) reused/adapted
  cleanly across CO3D and demo-scene runs.

## Phase D — Planning for room-scale extension (Replica → own room video)

- Confirmed via inspection of `training/data/datasets/`: only **CO3D** and **Virtual KITTI** have
  ready-made dataset loader classes in the released training code. The paper's other stated training
  datasets (BlendMVS, DL3DV, MegaDepth, Kubric, WildRGB, ScanNet, HyperSim, Mapillary, Habitat,
  Replica, MVS-Synth, PointOdyssey, Aria Synthetic/Digital Twin, an Objaverse-like set) would each
  require writing a new `Dataset` class following `co3d.py`'s pattern.
- Decided next dataset: **Replica** (subset: `room0`, `room1`, `room2`, `office0`, `office1`),
  currently downloading, for the room-scale / ConceptGraphs-facing phase of the dissertation.
- Planned architecture (external consultation, recorded for reference — not yet implemented):
  VGGT used as the RGB→(camera pose, depth) front-end feeding ConceptGraphs, rather than a hard
  dependency on Replica's ground-truth depth. Phase 1 (GT depth, validates ConceptGraphs stage) →
  Phase 2 (VGGT-predicted depth on Replica RGB, the critical ablation: how much does downstream scene
  understanding degrade vs GT) → Phase 3 (ScanNet, real captured RGB-D) → Phase 4 (own room video,
  fully sensorless pipeline). See `docs/deviations.md` for the scale/metric-accuracy caveat this
  architecture carries.

---

*(Append new dated entries below as work continues.)*
