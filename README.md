# Faithful Seam Carving

Content-aware resizing that never silently deletes or fuses objects.

Seam carving (Avidan & Shamir 2007) shrinks an image by removing low-energy
paths of pixels. It protects *pixels*, not *objects*: a person in the
background can be carved away, and the background between two people can be
removed until they fuse into one shape. In the RetargetMe user study
(Rubinstein et al. 2010) plain seam carving ranked last of 8 methods, mostly
for deformed people and objects, and no metric we found checks whether objects
survive. This project measures that failure and prevents it, with no training.

## What is new (and what is not)

| Part | Status |
|---|---|
| Forward-energy seam carving, seam insertion, dual-axis resizing | Re-implementation of Rubinstein et al. 2008 / Avidan & Shamir 2007 (`fsc/engine.py`) |
| Line protection, saliency protection, cost-knee stop | Known ideas (Kiess 2010; Achanta 2009, Hou & Zhang 2007; Dong 2009), kept as ablation components (`fsc/fixes.py`) |
| Object removal | In the original 2007 paper; here a demo of the same machinery (`fsc/removal.py`) |
| **Object-integrity metrics**: destroyed, vanished, fused, worse-than-scaling | **New** (`fsc/metrics.py`) |
| **Per-object budget** (cost grows as an object loses pixels) | **New** (`fsc/integrity.py`) |
| **Gap protection** between neighbouring objects | **New** (`fsc/integrity.py`) |
| **Integrity guard**: refuse a seam that breaks or fuses an object, then finish by scaling or cropping | **New trigger** for the stop of Dong et al. 2009 (`fsc/integrity.py`) |

## Method

Given an instance label map (0 = background, 1..N = objects), before every
seam each pixel of object *i* gets the extra cost

    λ · exp(γ · d_i)      d_i = 1 − (pixels of i left) / (pixels of i at start)

plus `μ` on background pixels lying in a gap of at most `g` pixels between two
different objects (along the carving direction). γ = 0 is the usual static
mask. Before a seam is removed, the guard refuses it if any object would lose
more than `max_loss` of its pixels, or if two objects that did not touch in the
original would become 4-adjacent (checked exactly). Carving then stops and the
rest of the size change is done by scaling (or an object-aware crop).

Defaults: λ = 1, γ = 8, μ = 50, g = 3 px, max_loss = 0.5 (`fsc/conditions.py`, `Params`).

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                          # 81 tests
python scripts/pilot_coins.py                # pilot table + figure on the coins photo
python scripts/get_data.py                   # COCO128-seg, ~7 MB, into data/
python scripts/run_ablation.py --out results # full ablation (~12 min on 2 cores at 320 px)
python scripts/make_figures.py --auto 4 --ratio 0.5 --tiles
```

Your own photo (masks from a pretrained YOLO model: `pip install ultralytics`):

```bash
python scripts/demo.py photo.jpg --width 50% -o out.png --overlay out_objects.png
python scripts/demo.py photo.jpg --width 50% --method sc_forward -o plain.png
python scripts/demo.py photo.jpg --labels my_mask.png --remove 2 -o removed.png   # object removal
```

User study material (paired comparisons, RetargetMe protocol):

```bash
python scripts/make_user_study.py --images 000000000073 000000000368 --ratio 0.5
```

## Conditions in the ablation

| Group | Name | What changes |
|---|---|---|
| Baselines | `scale`, `crop`, `sc_backward`, `sc_forward` | scaling, centre crop, vanilla and forward-energy seam carving |
| Original fixes | `lines`, `saliency`, `knee`, `fixes_all` | forward energy + each fix, then all three |
| Integrity (ours) | `static_mask`, `budget`, `budget_gap`, `faithful`, `faithful_crop`, `faithful_all` | static mask; budget; + gaps; + guard (scale finish); crop finish; + lines and saliency |

## Metrics (per object, then pooled)

* **kept**: share of the object's original pixels still in the output
* **destroyed**: kept < 50%; **vanished**: object entirely gone
* **fused**: touches another object in the output but not in the original
* **intact**: none of the above
* **worse than scaling**: object area shrank more than uniform scaling would have shrunk it
* **aspect distortion**: |log(aspect out / aspect in)| of its bounding box (scaling to 50% width = 0.69)
* **raggedness**: (perimeter² / area) out ÷ in; > 1 means a more jagged outline
* **carved %**: share of the size change done by seams before any finishing step

SSIM against the original is deliberately absent: it needs equal sizes, and
upscaling the result back favours plain scaling.

## Results

70 COCO128-seg photos with at least 3 objects (413 objects), downscaled to
320 px wide, ground-truth masks, width reduced to 50%. Full tables for 75% and
50%: [`results/summary.md`](results/summary.md); every run: `results/ablation_rows.csv`.

| Method | Intact % | Crushed % | Fused % | Aspect distortion | Carved % | s / image |
|---|---|---|---|---|---|---|
| Scaling | 97.1 | 0.0 | 2.9 | 0.693 | 0 | 0.00 |
| Centre crop | 57.6 | 42.4 | 0.0 | 0.361 | 0 | 0.00 |
| Seam carving, forward energy | 64.9 | 29.5 | 8.5 | 0.500 | 100 | 0.90 |
| + all three original fixes | 64.4 | 25.7 | 14.5 | 0.477 | 79 | 0.74 |
| + static masks | 51.3 | 14.8 | 41.4 | 0.307 | 100 | 0.92 |
| + budget | 56.9 | 0.5 | 42.6 | 0.241 | 100 | 1.02 |
| + budget + gaps | 94.4 | 0.5 | 5.1 | 0.252 | 100 | 1.25 |
| **Faithful (budget + gaps + guard)** | **99.0** | **0.0** | **1.0** | **0.263** | **98** | **1.33** |

At 75% width: seam carving 93.9% intact (21% of images damaged), Faithful 100%.

Reading it: static masks trade crushing for fusing; the budget stops crushing;
gap protection stops most fusing; the guard closes the rest while still
carving 98% of the way. Line protection and saliency make integrity worse
(48.7% and 57.4% intact). Every remaining fusion of Faithful happens in images
where the guard stopped and the scaling finish merged 1-2 px gaps (scaling
alone shows the same 2.9%).

Pilot (`scripts/pilot_coins.py`, 24 coins, 50% width): seam carving crushes 6,
static masks fuse 20 pairs, Faithful 0 and 0.

## Honest limits

* With `max_loss = 0.5` the guard enforces the "destroyed" criterion by
  construction. The informative comparisons are `budget` / `budget_gap` (no
  guard) and how much carving the guard still allows (carved %).
* Ground-truth COCO masks; objects under 0.5% of the image are ignored, so
  small background objects are unprotected. Predicted masks will be worse.
* Small set: the COCO128 images with at least 3 objects, downscaled to 320 px
  wide for speed. The paper needs COCO val2017 and RetargetMe at full size.
* Forward energy has many exact ties in flat regions; tiny floating-point
  differences (e.g. a grey image vs the same image as BGR) change which seam
  wins. Report results over many images, not one.
* At strong reductions protected objects get jagged outlines (see raggedness).
* Parameters are not tuned yet; run a sensitivity study on γ, μ and max_loss.

## Layout

```
fsc/engine.py       forward/backward energy, DP, removal, insertion, hooks, finishes
fsc/fixes.py        line mask, spectral-residual saliency, cost-knee stop
fsc/integrity.py    per-object budget, gap protection, integrity guard
fsc/metrics.py      object-integrity metrics
fsc/baselines.py    scaling and cropping in the same result format
fsc/conditions.py   every ablation condition behind one function
fsc/removal.py      object removal demo
fsc/datasets.py     YOLO-seg loader (COCO128-seg / COCO), optional YOLO masks
fsc/viz.py          overlays and figure panels
scripts/            pilot, ablation runner, figures, demo, user study, data download
tests/              81 tests, including brute-force checks of the dynamic program
```

## References

Avidan & Shamir, *Seam Carving for Content-Aware Image Resizing*, SIGGRAPH 2007 ·
Rubinstein, Shamir & Avidan, *Improved Seam Carving for Video Retargeting*, SIGGRAPH 2008 ·
Dong et al., *Optimized Image Resizing Using Seam Carving and Scaling*, SIGGRAPH Asia 2009 ·
Rubinstein et al., *A Comparative Study of Image Retargeting*, SIGGRAPH Asia 2010 ·
Kiess et al., *Seam Carving with Improved Edge Preservation*, SPIE 2010 ·
Achanta & Süsstrunk, *Saliency Detection for Content-Aware Image Resizing*, ICIP 2009 ·
Hou & Zhang, *Saliency Detection: A Spectral Residual Approach*, CVPR 2007 ·
Dickman & Diefenbach, *Directorial Editing*, Electronics 2024 ·
Shen et al., *Prune and Repaint*, NeurIPS 2024 ·
Elsner et al., *Retargeting Visual Data with Deformation Fields*, ECCV 2024.

Data: COCO128-seg (Ultralytics, GPL-3.0), a subset of MS COCO (CC BY 4.0).
