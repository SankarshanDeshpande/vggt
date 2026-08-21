# Project Flow — Status & Roadmap

## Kaam ho chuka hai (Done)

1. **HPC environment fully working** — conda env, VGGT repo + training deps installed, all the
   `conda activate` / compute-node-no-internet / SSH-tunnel-instability issues diagnosed and fixed.
   This alone is a real chunk of "implementation challenges" content for the report.
2. **VGGT training pipeline validated end-to-end** on CO3D `apple` (single-sequence subset) —
   checkpoint loads, dataloader works (after patching a real bug in `co3d.py`), forward + backward
   pass runs, loss decreases sensibly, checkpoint saves. This proves the released training code,
   your environment, and your data all work together correctly.
3. **VGGT inference pipeline built from scratch** in Jupyter (not using the demo scripts directly,
   since they hit the same subprocess issues) — image loading, model forward pass, depth/camera
   prediction, point-cloud unprojection, confidence + mask filtering, outlier removal, and a custom
   z-buffer renderer (built because matplotlib's default 3D scatter doesn't depth-sort correctly).
4. **Tested inference on 5 different scenes**: CO3D apple (dense orbit), kitchen (dense indoor),
   room/no-overlap (sparse indoor — correctly shows fragmentation, an expected result not a bug),
   oil painting (matches paper Fig. 3 row 1), and a video (pyramid, after fixing frame-extraction).
5. **Multiple real bugs found, diagnosed, and fixed** — reflection vs rotation (inverted convexity),
   unmasked black padding polluting the point cloud, mask/image misalignment, lack of depth-sorting
   in visualisation. Each of these is genuinely reusable "lessons learned" content.
6. **Dataset coverage gap identified**: only CO3D and VKitti have ready dataset loaders in the
   official code; ScanNet/Replica/HyperSim etc. would need new loaders written.
7. **Replica subset download started** (`room0`, `room1`, `room2`, `office0`, `office1`) for the
   room-scale phase, with a planned architecture (VGGT as RGB→depth+pose front-end → ConceptGraphs)
   already sketched out.

## Ab aage kya karna hai (Next)

**Immediate (Phase D continuation):**
- Confirm Replica download completed correctly (RGB + depth + poses, not just RGB — check this
  explicitly, since Phase 1 below needs ground-truth depth).
- Adapt the existing inference pipeline (same cells, different image folder) to run VGGT on Replica
  RGB and produce point clouds — this is mostly copy-paste from what's already built.

**Phase 1 — Replica with ground-truth depth:**
- Replica RGB + Replica GT depth + Replica GT poses → ConceptGraphs. Goal: confirm ConceptGraphs
  stage itself works correctly, independent of VGGT. Not yet started — ConceptGraphs isn't
  installed/tested at all yet.

**Phase 2 — Replica with VGGT-predicted depth (the important ablation):**
- Replica RGB only → VGGT → predicted depth + poses → ConceptGraphs. Compare scene-graph output
  against Phase 1. This is the number that answers "how much does using VGGT instead of real depth
  cost you" — a genuinely useful, presentable result for the dissertation.

**Phase 3 — ScanNet:**
- Real captured indoor RGB-D, not rendered. Tests whether the pipeline holds up on real-world
  imagery. Requires writing a ScanNet `Dataset`/loading path (none exists in the released VGGT
  training code, per deviations.md, but for *inference* you don't need the training-side Dataset
  class — you just need images).

**Phase 4 — Your own room:**
- Ordinary phone video → frame extraction (already built, reused from the pyramid video work) →
  VGGT → depth + poses → ConceptGraphs → scene reasoning. This is the actual novel end-to-end demo
  for the dissertation — everything before this phase exists to justify and validate it.

**Not started at all yet, needed before Phase 1 can begin:**
- ConceptGraphs installation and setup on the HPC (same network-access caveats as VGGT/CO3D likely
  apply — plan to download any weights/deps from the login node).
- Scene-LLM stage (final step in the planned architecture) — not evaluated at all yet.

## Report / Paper / PPT ke liye directly useful hai

- `research_log.md` → seedha "Methodology" + "Implementation Challenges" section ban sakta hai.
- `deviations.md` → "Limitations" / "Scope" section, aur agar examiner poochhe "why apple only, why
  single sequence" — is document mein har answer already likha hai with reasoning.
- `table.md` → results section ka starting point; jab quantitative paper-table reproduction ya
  Phase 2 ablation ready ho, usi format mein add karte jaana.
- Figures: abhi tak jo bhi `.png`/`.ply` bane hain (apple, kitchen, room, pyramid) — sab ek
  `results/figures/` folder mein daal dena HPC par, taaki thesis ke figures seedha wahi se aayen.

## Suggested next action

Sabse pehle Replica download confirm karo aur uska folder structure check karo (RGB + depth + pose
teeno hain ya sirf RGB) — usi se pata chalega Phase 1 abhi start ho sakta hai ya nahi.
