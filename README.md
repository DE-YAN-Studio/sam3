# SAM3 (Windows Fork)

Fork of [facebookresearch/sam3](https://github.com/facebookresearch/sam3) — Meta's open-vocabulary segmentation model (Segment Anything with Concepts).

## What changed

`sam3/model/edt.py` wraps the hard `import triton` in a `try/except` with a `scipy` CPU fallback. Triton is Linux-only; this makes SAM3 importable on Windows. The fallback only affects the video-tracking path (`fill_holes_in_mask_scores`) — image inference is unaffected.

## TouchDesigner integration

See [DE-YAN-Studio/sam3-td](https://github.com/DE-YAN-Studio/sam3-td) — a FastAPI server and TouchDesigner client scripts that expose SAM3 inference as live TOPs, using this fork as a submodule.

## Original project

All other code is Meta's. For model weights, architecture details, and training instructions see the [original SAM3 repo](https://github.com/facebookresearch/sam3) and [paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/).
