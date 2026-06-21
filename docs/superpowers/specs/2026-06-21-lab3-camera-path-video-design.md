# Lab3 Continuous Camera-Path Video Rendering Design

## Context

Current Lab 3 runs under `outputs/lab3/<run>` already contain:

- a canonical prepared image set under `prepared/`
- shared COLMAP poses under `prepared/sparse/0` when `share_poses=true`
- trained 3DGS, Nerfacto, and NeuS artifacts under `results/`
- existing evaluation render entrypoints for Graphdeco 3DGS and nerfstudio-backed methods

The missing piece is a single post-processing script that consumes one completed run, derives a continuous camera path from the discrete source poses, re-renders the same path for all three learned methods, emits three separate videos, and records per-method rendering time on that exact workload.

The user will assemble comparison layouts separately, so this work stops at per-method video generation plus timing summaries.

## Confirmed Requirements

- Input is an existing Lab 3 run directory, passed mainly by `--run-dir`.
- The script may keep editable in-file defaults for convenience, but every important value must also be overridable from the CLI.
- The script must not be named around `fps4`; it should be generic even though the intended run currently comes from the fps4 experiment.
- The rendered path must be shared across 3DGS, Nerfacto, and NeuS.
- The path is fit from the original discrete poses.
- Pose fitting uses:
  - position: continuous interpolation of camera centers
  - orientation: smooth interpolation driven by neighboring discrete orientations
- Default output target is about 30 seconds at 24 fps, but both must be adjustable.
- Output resolution must support CLI override via `--width/--height`, while defaulting to the run config’s `eval_size` when present.
- The script must produce three independent videos, one per method.
- The script must measure and save how long each method takes to render the same video segment.
- This task only implements the script; it does not execute rendering now.

## Recommended Approach

Implement one orchestration script that:

1. inspects a completed run directory and resolves all required trained artifacts;
2. loads the shared discrete camera poses from COLMAP text or binary model files;
3. builds a sampled continuous camera trajectory from those poses;
4. exports backend-specific camera-path inputs as needed;
5. renders the trajectory separately for 3DGS, Nerfacto, and NeuS;
6. encodes per-frame outputs into one video per method;
7. writes a machine-readable timing and output summary.

This keeps all fairness-sensitive decisions in one place: path sampling, frame count, resolution, and timing semantics.

## Alternatives Considered

### Option A: Separate trajectory exporter plus three renderer scripts

Pros:

- each backend remains isolated
- easier to rerun one backend manually

Cons:

- splits the fairness logic across multiple entrypoints
- increases file count and interface surface
- makes timing aggregation less reliable

### Option B: Command generator only

Pros:

- smallest implementation

Cons:

- leaves execution semantics to manual steps
- does not guarantee same path, same frame count, or same timing scope
- does not satisfy the “complete one script” requirement well

### Chosen Option

Option A is rejected as unnecessary decomposition for the current scope. Option B is too weak. A single orchestration script is the right boundary.

## Design

### Script placement and scope

Add a new script under:

- `scripts/lab3/render_camera_path_videos.py`

The name stays generic and does not encode the current fps4 run.

The script is a post-training tool. It must only read an existing run directory and write new render/video artifacts inside that run, without retraining or mutating source datasets.

### Configuration model

The script exposes all operational parameters in two ways:

1. editable top-of-file defaults collected in one small config block;
2. argparse flags overriding those defaults.

Required high-value knobs:

- `--run-dir`
- `--fps`
- `--duration-sec`
- `--width`
- `--height`
- `--pose-source` or equivalent selector if multiple pose sources are later supported
- `--position-spline` or equivalent interpolation mode flag for future extensibility
- `--output-dirname`
- `--ffmpeg-bin`
- `--methods` with default `3dgs nerf neus`
- backend-specific binary overrides where current wrappers already need them

The in-file defaults are a usability requirement, not a replacement for CLI configuration.

### Artifact discovery

The script resolves method artifacts from the run directory using the same conventions already used by Lab 3:

- 3DGS model root: `results/3dgs`
- Nerfacto training config: latest `config.yml` under `results/nerf/train`
- NeuS training config: latest `config.yml` under `results/neus/train`
- NeuS patched checkpoint / patched load dir: reuse the existing `patch_checkpoint_for_eval` logic or the same helper path layout instead of duplicating an incompatible workaround
- Run config: `configs/run_config.json`

Resolution defaulting:

- first choice: `eval_size` from `configs/run_config.json`
- fallback: infer from one prepared image
- CLI `--width/--height` always wins

### Pose loading

Primary pose source:

- shared COLMAP reconstruction under `prepared/sparse/0`

The script should load ordered image poses and image names from the COLMAP model. The order must be deterministic and should align with the prepared image sequence rather than arbitrary file iteration.

The trajectory source should prefer all available original capture poses, not only held-out evaluation poses, because the stated requirement is to fit a continuous path from the original discrete video poses.

### Continuous trajectory fitting

For this task, the fitting model is intentionally simple and explicit:

- camera centers are interpolated continuously from discrete centers
- orientations are smoothed/interpolated separately from the center fit

Recommended concrete implementation:

- convert COLMAP world-to-camera poses into camera-to-world matrices
- extract camera centers and forward/up vectors
- parameterize the path by cumulative arc length or normalized sample index
- fit camera centers with a cubic spline when enough points exist, with a safe fallback to linear interpolation for short sequences
- interpolate orientation using quaternion slerp between neighboring keyframes chosen by the sampled path parameter
- reconstruct a camera-to-world pose per output frame from interpolated center + orientation

This deliberately avoids claiming full SE(3) smoothing. The design should state that orientation and translation are treated separately by choice.

### Path sampling

Default output size:

- duration: 30 seconds
- fps: 24
- frame count: `round(duration_sec * fps)`

Sampling must use a single canonical frame count shared by all methods.

The script should write the sampled path to an intermediate manifest, for example:

- `renders/camera_path_videos/camera_path.json`

That manifest should contain at minimum:

- frame index
- timestamp
- camera-to-world transform
- image width / height
- fx / fy / cx / cy if required by downstream renderers

This manifest becomes the truth source for all backends and for debugging fairness issues later.

### Backend render strategy

#### 3DGS

Reuse the Graphdeco repository already referenced by the Lab 3 wrapper. The script should not reimplement the renderer itself. It should generate whatever camera-path input Graphdeco’s render path expects, or invoke a small local adapter if the upstream renderer only supports dataset/test renders.

If a local adapter is needed, keep it narrow:

- load the trained 3DGS model
- consume the sampled camera-path manifest
- render RGB frames to a designated directory

#### Nerfacto

Prefer reusing `ns-render` against the trained config when it supports external camera-path rendering. If nerfstudio’s existing `dataset` mode is insufficient, use the minimal compatible path-render mode or a local adapter that still consumes the saved model/config rather than rebuilding rendering logic from scratch.

#### NeuS

Use the same nerfstudio-family rendering path as Nerfacto, but preserve the existing patched-checkpoint workaround so rendering does not fail on appearance-embedding shape mismatch.

The script must not assume NeuS can render successfully from raw training artifacts without that compatibility step.

### Output layout

Write outputs under a dedicated subdirectory inside the run:

- `renders/camera_path_videos/`

Recommended structure:

- `renders/camera_path_videos/camera_path.json`
- `renders/camera_path_videos/3dgs/frames/`
- `renders/camera_path_videos/3dgs/video.mp4`
- `renders/camera_path_videos/nerf/frames/`
- `renders/camera_path_videos/nerf/video.mp4`
- `renders/camera_path_videos/neus/frames/`
- `renders/camera_path_videos/neus/video.mp4`
- `renders/camera_path_videos/timing_summary.json`
- `renders/camera_path_videos/timing_summary.csv`

This keeps the new artifact family separate from held-out evaluation renders.

### Timing semantics

Timing must be comparable across methods, so the script needs one explicit policy:

- per-method timer starts immediately before frame rendering begins for the sampled path
- per-method timer ends after the final frame image is written
- video encoding time should be tracked separately from frame rendering time

The summary should therefore include at least:

- `render_time_sec`
- `encode_time_sec`
- `total_time_sec`
- `frame_count`
- `target_fps`
- `effective_render_fps = frame_count / render_time_sec`
- output resolution
- output video path

This separation prevents misleading comparisons where one backend is penalized mostly by ffmpeg rather than rendering.

### Error handling

The script should fail loudly and specifically when:

- `--run-dir` does not exist
- the run config is missing or malformed
- no shared COLMAP pose source can be found
- a method’s final config/checkpoint cannot be resolved
- the requested resolution is invalid
- the trajectory has fewer than two valid poses
- ffmpeg is missing when video encoding is requested

Method failures should not silently pass. By default the script should stop on the first backend failure because the point of the tool is fair three-way rendering. A future `--keep-going` flag can be added later if needed, but it is not required now.

## Files and Responsibilities

### New file

- `scripts/lab3/render_camera_path_videos.py`
  - parses defaults + CLI
  - resolves run artifacts
  - loads and fits poses
  - exports canonical path manifest
  - orchestrates method-specific renders
  - times rendering and encoding
  - writes summary files

### Existing files to reuse, not duplicate

- `src/lab3/reconstruction/dgs.py`
  - existing knowledge of 3DGS repo layout and binaries
- `src/lab3/reconstruction/nerfstudio.py`
  - existing config discovery and nerfstudio command conventions
- `src/lab3/reconstruction/neus.py`
  - existing NeuS compatibility workaround logic
- `src/lab3/common.py`
  - subprocess helpers, errors, timing helpers where appropriate

### Optional refactor if duplication becomes obvious

If the script starts copying too much logic from reconstruction adapters, a small helper module under `src/lab3/` can be introduced for:

- config/artifact discovery
- COLMAP pose loading
- camera-path manifest writing

That refactor is allowed only if it reduces duplication materially. It is not required up front.

## Testing Strategy

Testing should focus on deterministic logic and artifact resolution rather than actual heavy rendering.

Required coverage:

1. pose loading from a tiny synthetic or fixture COLMAP model
2. continuous frame-count generation from `fps * duration`
3. resolution defaulting and CLI override precedence
4. timing-summary schema generation
5. backend artifact discovery for 3DGS / Nerfacto / NeuS
6. NeuS patched-checkpoint path selection without executing CUDA rendering
7. ffmpeg command generation

Heavy external rendering should be covered only by command construction tests and temporary-directory fixtures, not by launching Graphdeco or nerfstudio in unit tests.

## Non-Goals

- training or retraining any model
- building a combined comparison montage video
- changing Lab 3 pipeline output semantics
- implementing mathematically stricter SE(3) smoothing in this task
- redesigning the existing evaluation pipeline around this path renderer

## Opened Extension Points

The script should be written so later changes remain local:

- alternate path duration/fps/resolution
- alternate interpolation modes
- optional keep-going behavior
- optional depth/normal video export
- optional direct stitched comparison video

These are extension points only; they should not expand the first implementation.
