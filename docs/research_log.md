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
## Phase E — Replica room0: qualitative reconstruction + furniture failure-mode analysis
- Downloaded Replica_OCC subset (room0, room1, room2, office0, office1) via hf download the-masses/ReplicaOcc, hit an unauthenticated-request rate limit (HTTP 429) and a transient DNS drop mid-download — both resolved by authenticating (hf auth login) and retrying (downloads resume from cached partial state, not a full restart).
room0 structure confirmed: color/ (.jpg), depth/ (.png, uint16 mm, 0 = invalid), pose/ (.txt, 4×4 camera-to-world), plus intrinsic_color.txt/intrinsic_depth.txt (identical: fx=fy=600, cx=599.5, cy=339.5) and matching extrinsic_*.txt. 2000 frames total, numerically indexed (not zero-padded) — required a numeric sort key, not the default string sort.
- Verified via camera-position deltas (positions at frames 0, 500, 999, 1000, 1500, 1999) that the 2000-frame sequence is a single continuous closed loop, not a forward-then-reverse traversal — confirms plain even-index sampling across the full range is valid, no special-casing needed.
- Ran VGGT inference (base checkpoint, zero-shot, no Replica-specific fine-tuning) on room0 at increasing frame counts (64 → 128 → 256) using the existing z-buffer visualisation pipeline.
- Result — walls/floor/large structure reconstruct well, closely matching the Replica dataset's own reference rendering (same room shape, door, wall art, cabinet placement recognisable).
- Result — furniture and soft/textured surfaces (rug, sofa cushions) and the window show persistent artefacts (speckling, a hole through the rug, noisy sofa geometry) that did not improve with more frames (64→256, negligible visible change) or with looser confidence thresholds (tested 5.0 → 3.0 → 1.5, negligible visible change).
- Diagnosed the cause quantitatively, not just visually: split point confidence by approximate furniture vs wall region (pts[:,2] height threshold) and compared distributions. Furniture-region confidence (median 4.77, IQR 3.87–6.18) is systematically lower than wall-region confidence (median 6.21, IQR 4.37–8.11) — roughly 23% lower at the median. Overall confidence-filter retention rate was 96%, ruling out "over-aggressive filtering" as the cause; the model itself is less certain about furniture geometry before any threshold is applied.
- Conclusion: this is a genuine VGGT limitation on complex/soft/textured geometry (furniture, fabric, reflective glass), not a bug in the reconstruction pipeline, and not fixable via frame count or filtering. Consistent with the VGGT paper's own acknowledged weakness on repeating/homogeneous- texture regions (Sec. 4.3, Fig. 3 discussion). Logged as a documented finding rather than a defect.
- Debugging note: hit a RuntimeError: NVML_SUCCESS ... INTERNAL ASSERT FAILED when jumping frame count 128→1024 in one step (likely a genuine shard-memory ceiling being hit abruptly). The error persisted even after reducing back down to 256 in the same kernel session — a full Jupyter kernel restart (not just torch.cuda.empty_cache()) was required to clear corrupted CUDA state before 256 succeeded cleanly. Lesson: scale frame count up incrementally (e.g. 128→160→ 192→224→256...) rather than in large jumps, and restart the kernel after any CUDA allocator crash before trusting subsequent results in the same session.
- Measured actual GPU memory footprint on a shard:20 allocation: ~25.45GB baseline (model + session state) at N=128, climbing only ~0.4GB per +32 frames up to N=256 (27.06GB) — headroom appears larger than initially assumed once stale CUDA state is ruled out; not yet pushed past 256 to find the true ceiling for this shard size.

## Phase F — ConceptGraphs setup: environment, checkpoint, and LLaVA-v0 obsolescence fix
- Installed ConceptGraphs' full stack: separate conceptgraph conda env (Python 3.10, pinned torch==2.0.1+cu118, deliberately older/isolated from the vlm env's torch 2.6.0+cu124), Grounded-SAM (SAM + GroundingDINO + RAM/Tag2Text), gradslam (conceptfusion branch), chamferdist, pytorch3d.
- Recurring installation issues, same root causes as earlier phases: bare pip/python resolving to the wrong env (fixed via python -m pip, consistent with the vlm env lesson); build-time CUDA-extension compilation needing --no-build-isolation (chamferdist, GroundingDINO) since default pip build isolation hides already-installed torch from the build subprocess; a pkg_resources/ModuleNotFoundError from newer setuptools removing something torch 2.0.1's cpp_extension still expects (fixed by pinning setuptools<70 in the conceptgraph env).
- Downloaded 3 required Grounded-SAM checkpoints (SAM ViT-H, GroundingDINO SwinT, RAM Swin-Large) from the login node. One checkpoint URL initially wrong — ram_swin_large_14m.pth is hosted under xinyu1205/recognize_anything_model, not recognize-anything-plus-model (that repo hosts the different RAM++ variant) — corrected after verifying via search rather than guessing again.
- Major finding: ConceptGraphs' captioning stage depends on LLaVA-7B-v0, which is no longer downloadable. It was never released as a standalone checkpoint — only as a delta (liuhaotian/LLaVA-7b-delta-v0) requiring separate access to gated Meta LLaMA-7B weights to apply. This makes the officially documented captioning path a dead end for a from-scratch setup in 2026.
- Fix implemented: wrote a new conceptgraph/llava/llava_model_hf.py, replacing the old hand-modified LlavaLlamaForCausalLMTweaked (which reads v0-specific config fields like mm_vision_tower/mm_use_im_start_end not present in modern checkpoints) with a thin wrapper around HF's standard LlavaForConditionalGeneration + AutoProcessor, using the actively maintained, directly-downloadable llava-hf/llava-1.5-7b-hf checkpoint. Matches the original LLaVaChat class's external call signature (load_image, __call__(query, ...), reset) so build_scenegraph_cfslam.py needs only an import-path edit, not a rewrite of its calling code.
- One real architecture difference required a second iteration: the old v0 wrapper allowed encoding image features and building the text prompt as two separate steps (features spliced into embeddings manually in a custom forward()). HF's native LlavaForConditionalGeneration requires the raw image and prompt text to be passed to the processor together in one call, so it can correctly expand the single <image> placeholder token into the right number of tokens matching the vision tower's patch count — first attempt threw ValueError: Image features and image tokens do not match: tokens: 1, features 2359296 from trying to pre-extract features separately, as the old interface allowed. Rewrote __call__ to take the PIL image and query together in one processor(text=..., images=...) call.
- Verified working end-to-end, running directly in the existing vlm conda env/Jupyter kernel (no separate llava env needed — llava-1.5-7b-hf has no dependency on ConceptGraphs' old pinned stack): loaded the checkpoint, ran inference on a real image, received a sensible, on-topic caption ("The central object in the image is a person, possibly a man, who is sitting in a chair."). Confirms the free/local LLaVA substitution is viable for the captioning stage.
- Also confirmed (code inspection, not yet fixed): the two GPT-4 calls in build_scenegraph_cfslam.py (refine_node_captions, and the relation-extraction step in build-scenegraph mode) use the pre-1.0 OpenAI SDK style (openai.ChatCompletion.create(model="gpt-4", ...)), which respects a redirectable openai.api_base. This means both calls can likely be pointed at a free, local, OpenAI-API-compatible model server instead of paid GPT-4 — not yet implemented.
- Also confirmed (code inspection, not yet fixed): ReplicaDataset in datasets_common.py expects a single consolidated traj.txt pose file per scene (self.pose_path = os.path.join(self.input_folder, "traj.txt")), not the per-frame pose/<i>.txt files present in the Replica_OCC data actually downloaded — a format mismatch requiring a one-time conversion script before any ConceptGraphs run against this data.
- Resolved ReplicaDataset incompatibility — no traj.txt conversion needed. Read ReplicaDataset.get_filepaths()/load_poses() directly: it expects results/frame*.jpg, results/depth*.png, and one consolidated traj.txt — the Nice-SLAM-specific layout, which does not match Replica_OCC's structure at all (wrong subfolder name, wrong file-naming pattern, not just a pose-file format difference as first assumed). However, ScannetDataset in the same file expects exactly what Replica_OCC already provides: color/*.jpg, depth/*.png, pose/*.txt (one file per frame, natsorted, loaded directly via np.loadtxt), plus intrinsics read from intrinsic/intrinsic_color.txt — confirmed this exact subfolder/file exists in Replica_OCC (room0/intrinsic/{intrinsic,extrinsic}_{color,depth}.txt). Fix: use dataset_name: 'scannet' in the dataconfig instead of 'replica', keeping Replica's actual camera params/depth scale in the same YAML (new file: dataset/dataconfigs/replica/replica_occ.yaml) — zero file restructuring, no converter script needed. ScannetDataset.__init__ re-reads intrinsics from intrinsic/intrinsic_color.txt at runtime regardless of what's listed in the YAML, so accuracy there doesn't depend on the YAML being perfectly correct either.
- Verified end-to-end on real GPU hardware: ran scripts/run_slam_rgb.py (GradSLAM's PointFusion) against Replica_OCC/room0 using the new replica_occ.yaml config (dataset_name: scannet), stride=5 (400 of 2000 frames). Completed cleanly in ~110 seconds on a shard:20 H100 allocation, no errors, fused RGB point cloud saved to room0/rgb_cloud. This confirms the ScannetDataset substitution correctly reads Replica_OCC's color/depth/pose/ intrinsics with proper frame alignment — the dataset-compatibility question from earlier in this phase is now fully resolved and empirically verified, not just reasoned about from code inspection.
- generate_gsa_results.py completed successfully on room0: all 400/400 frames (stride 5) processed without error in ~1hr57min on a shard:20 allocation, submitted via sbatch after the interactive srun session hit its time limit at 228/400 frames. This is the first real end-to-end validation of the full detection/segmentation stage (Grounded-SAM + CLIP), and confirms every supervision/torchvision/transformers compatibility fix made during setup holds up across a complete run, not just a few test frames.
- Additional supervision API-drift bugs found and fixed in conceptgraph/utils/vis.py while getting this run working (same root cause as the earlier ColorPalette.default() fix — the installed supervision==0.30.1 has a substantially different API than what this 2023-era code was written against):
- sv.BoxAnnotator.__init__() no longer accepts text_scale/text_thickness/text_padding — these moved to a separate sv.LabelAnnotator class in current supervision versions.
- box_annotator.annotate(..., labels=labels) no longer accepts a labels kwarg — labeling is now a distinct .annotate() call via LabelAnnotator, applied after box drawing rather than combined into one call.
- Fixed by adding a label_annotator = sv.LabelAnnotator(...) alongside the existing box_annotator, and splitting the single annotate call into two sequential calls (box, then labels).
- Separately, root-caused and fixed a torchvision C++ extension load failure (RuntimeError: Couldn't load custom C++ ops) that had been silently present since the very first command in this env (visible as a suppressed warning: Failed to load image Python extension) but only became fatal once nms/batched_nms (used by SAM's automatic mask generator) was actually invoked. Fixed by reinstalling torchvision==0.15.2 explicitly matched to the pinned torch==2.0.1+cu118 build via the PyTorch wheel index, rather than relying on a plain pip install torchvision==0.15.2 (which had installed a build with mismatched/missing compiled ops). Also downgraded numpy<2 in this env, since the reinstalled torchvision's compiled extensions were built against NumPy 1.x and emitted _ARRAY_API not found warnings under NumPy 2.x (non-fatal here, but a known source of silent incorrect results if left unaddressed).
- Environment-setup notes accumulated getting to this point (same category of issue as throughout this project, listed for reference): conceptgraph package requires activating the conceptgraph conda env specifically (not vlm) since it was pip install -e .'d there; several dependencies were missing from the original Step 1 install list and surfaced one at a time on first real run — open3d, imageio, natsort, and gradslam (which had apparently not persisted correctly from its original Step 3 install and needed reinstalling with --no-build-isolation). Also re-learned the "must run on a GPU compute node, not the login node" lesson in this new env, same as with the LLaVA test — srun --partition=gpu --qos=normal --gres=shard:20 ... used to get GPU access for testing. Added persistent env vars (REPLICA_ROOT, REPLICA_CONFIG_PATH, GSA_PATH, LLAVA_CKPT_PATH) to ~/.bashrc so they survive across terminal sessions and env switches going forward.
- Also confirmed: the dataconfig's png_depth_scale: 6553.5 matches this Replica variant's actual depth encoding — not the /1000.0 (millimeters) scale factor used in the earlier Phase-E-adjacent GT-vs-VGGT AbsRel depth comparison code. That comparison's numbers, if already computed, need to be redone with the correct scale factor before being trusted/reported.

## Phase G — Remaining Replica scenes, pipeline scoping, architecture review

- **VGGT inference completed on all 5 Replica scenes** (`room0`, `room1`, `room2`, `office0`,
  `office1`) using the fully corrected pipeline (padding mask via `original_coords`, absolute
  confidence threshold, true unmirrored coordinates with camera-only orientation control, custom
  z-buffer renderer) — the same code that produced the verified-correct room0/pyramid results
  earlier in this phase. All outputs (`*_inputs.png`, `*_sweep.png`, `*_final.png`, per-scene `.ply`)
  saved to `outputs/pointclouds/`. This completes the qualitative portion of Stage 2 (VGGT
  reconstruction) for the full 5-scene set — dissertation figures for all scenes now exist and are
  trustworthy, no re-run needed.
- **Scoped two known gaps for later, explicitly not blocking current work:**
  - **VGGT-to-Replica-GT coordinate alignment is not implemented.** Confirmed this only matters for
    Stage 5's planned quantitative "3D reconstruction accuracy vs. ground truth" evaluation
    (Chamfer distance / point-to-point error would be meaningless without first rigidly aligning
    VGGT's arbitrary-scale, first-camera-frame coordinate system to Replica's GT frame, e.g. via
    Umeyama alignment). Does **not** block ConceptGraphs (Stage 3/4), which only needs internally
    self-consistent pose/depth/intrinsics per scene, not an external reference frame — confirmed by
    `run_slam_rgb.py`'s success on room0 without any such alignment. Deferred until Stage 5 is
    actually reached.
  - **Matplotlib/z-buffer visualization quality is a display-only concern, not a data-pipeline
    concern.** ConceptGraphs consumes VGGT's raw depth/pose/intrinsic tensors directly
    (`depth_map`, `extrinsic`, `intrinsic`) and never touches the custom renderer built for visual
    sanity-checking in Jupyter — confirmed no dependency between them. Improving render clarity is a
    dissertation-figure-polish task, decoupled from pipeline correctness.
  - **Reviewed the overall project architecture against a supervisor-facing flowchart** (5-stage
    pipeline: Data Prep → VGGT reconstruction → ConceptGraphs scene graph → Scene LLM reasoning →
    Applications/Evaluation) and identified one structural gap worth flagging: the "Scene LLM"
    stage (natural-language Q&A/planning over the finished graph) is **not a component ConceptGraphs
    provides out of the box** — its own LLM calls (`refine-node-captions`, `build-scenegraph`) are
    embedded inside the graph-*construction* stage, not a separate queryable interface. Building a
    conversational/task-planning layer on top of the exported graph JSON will be custom work, to be
    scoped separately when reaching that stage.

## Phase H — GPT-4 → free local model redirect (Ollama)

- **Installed Ollama as a standalone binary inside the `vlm` conda env's own `bin`/`lib` folders**
  (not system-wide, not a separate env) — since Ollama has no Python dependencies, this just makes
  it available on PATH only when `vlm` is active. Official install script's direct `.tgz` download
  URL is dead (Ollama switched to `.tar.zst` format at some point) — fixed by querying GitHub's
  releases API directly for the current real asset filename rather than guessing, then extracting
  with `tar --zstd`.
- **Ollama's model registry blob storage (`*.r2.cloudflarestorage.com`, Cloudflare R2) is
  unreachable from this network** — confirmed not HPC-specific: the exact same timeout occurred
  from a personal Windows laptop on a different network, ruling out an HPC firewall as the cause
  and pointing to a broader block/routing issue with that specific CDN endpoint. `ollama pull`
  therefore cannot be used at all in this environment.
- **Fix: bypassed Ollama's registry entirely.** Downloaded a GGUF-format checkpoint
  (`bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`, `Q4_K_M` quantization, ~4.9GB) directly from
  Hugging Face — a domain already confirmed reliable throughout this project — then registered it
  as a local Ollama model via a minimal `Modelfile` (`FROM <path-to-gguf>`) and `ollama create`,
  which never touches the R2 blob-storage path. Verified with a real chat completion request
  (`curl .../v1/chat/completions`) returning a coherent response.
- **Patched `conceptgraph/scenegraph/build_scenegraph_cfslam.py`** to redirect both GPT-4 call
  sites (`refine_node_captions`, and the relation-extraction step in `build-scenegraph` mode) to
  the local Ollama server: set `openai.api_base = "http://localhost:11434/v1"`, changed both
  `model="gpt-4"` to `model="llama3.1-gguf"`, and gave `OPENAI_API_KEY` a placeholder fallback
  value (the SDK only checks the key is non-empty when hitting a custom `api_base`, never
  validates it). Confirmed feasible in advance by inspecting the code: both calls use the
  pre-1.0 `openai` SDK's `ChatCompletion.create()` syntax, which respects a redirectable
  `api_base` — installed `openai==0.28.1` in `conceptgraph` to match, since this env had no
  `openai` package at all until now and the current (1.x) SDK removed that calling style entirely.
- **This closes out all three prerequisite blockers identified for the ConceptGraphs pipeline**:
  dataset-format compatibility (Phase G), the detection/segmentation stage validated at scale
  (Phase G — 400/400 frames, room0), and now the GPT-4 dependency, at zero API cost. Local model
  currently runs on CPU (~24 tokens/sec observed) rather than GPU — acceptable for now given
  correctness was the priority; worth revisiting only if per-object captioning across 5 scenes
  turns out too slow in practice.
