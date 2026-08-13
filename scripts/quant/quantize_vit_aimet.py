#!/usr/bin/env python
"""AIMET PTQ W8A16 for the Qwen3-VL vision tower (ExportQwen3VLViT).

WHY THE ViT MUST BE QUANTIZED AT ALL
------------------------------------
The FP16 vision tower cannot be driven by a stock GeniePipeline on QAIRT
2.48.40.260702, for two independent reasons in the shipped qualla sources:

  - QnnNspImageModel::setupInputFP16 is an empty stub -- it discards the pixel
    data and returns success (nsp-image-model.cpp:526-530).
  - The requantization dispatch table has no Float16 entries at all, so moving
    image features into the embedding accumulator throws
    (quantization/src/Quantization.cpp:163-192).

Both gaps close when the tower's graph I/O is FIXED-POINT:
QNN_DATATYPE_UFIXED_POINT_16 dispatches to a real setupInput<uint16_t> copy
(nsp-image-model.cpp:541-543), and {UFixed16, Float32} IS in the requant map.
Qualcomm's own reference vision encoder (siglip, examples/Genie/configs/glm-4v)
is likewise quantized rather than FP16. Hence W8A16, matching the text tower.

The deliverable that makes the converter emit UFIXED_16 I/O is an encodings
file carrying entries for `pixel_values` and all four outputs; that is asserted
at the end of every run, not eyeballed.

CALIBRATION DATA
----------------
W8A16 activation encodings are whatever calibration says they are, so the
calibration set has to look like production. modeling_vit_export.make_pixel_values
generates a UNIFORM RANDOM NOISE image; that is correct for the parity gates
(they compare two implementations of identical math, so image semantics are
irrelevant) and actively wrong here -- noise has flat-spectrum statistics
nothing like a photograph: no edges, no smooth regions, no spatial correlation.

So calibration reads REAL IMAGES from --images-dir (repeatable). The pool the
shipped encodings were drawn from was 44 photographs:

  $LLMDEPLOY_DATA/work/calib/images-coco-val2017   40 COCO val2017 photographs
                                                   (41 on disk; one is below
                                                    --min-edge and is skipped)
  $QAIRT_SDK/examples/Models/InceptionV3/data       4 SDK sample photographs

$QAIRT_SDK/examples/QAIRT/python/images is also worth passing but contributed
nothing here -- all three are 224px, under the --min-edge floor.

COCO images are fetched by --fetch-coco (needs the proxy, see the Gotchas in
CLAUDE.md); the SDK ones are always on disk. Images are deterministically
shuffled with --shuffle-seed and split, so the calibration and --eval sets are
disjoint BY CONSTRUCTION rather than by a seed convention.

--synthetic N appends N structured synthetic images (gradients, hard-edged
solids, 1/f "pink" noise, gratings, document-like layouts, exposure/contrast
extremes). That is a FALLBACK for a box with no real images, and the script
falls back to it automatically -- loudly -- if --images-dir yields too few.
It spans realistic image statistics far better than uniform noise, but it is
not photographs and should be reported as a quality risk when it is used.

Every image is centre-cropped to a square and routed through the REAL HF image
processor, so patch ordering and normalisation match production exactly, and
each one is asserted to land on the static 32x32 patch grid.

QUALITY GATE
------------
--eval runs the held-out images through both the FP32 wrapper and the quantsim
model and reports per-output max|d| and cosine. There is no pre-established
numerical bar for a quantized ViT -- the text tower's 3/4 last-token argmax is
a language-model metric and does not transfer -- so nothing here asserts on a
threshold. Exit status depends only on what is unambiguously broken: non-finite
values, shape mismatches, an output that is all-zero or constant.

Run:
  $PY_DEPLOY scripts/quant/quantize_vit_aimet.py \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
      --out   $LLMDEPLOY_DATA/work/quant/qwen3vl-4b-vit \
      --images-dir $LLMDEPLOY_DATA/work/calib/images-coco-val2017 \
      --images-dir $QAIRT_SDK/examples/Models/InceptionV3/data \
      --images-dir $QAIRT_SDK/examples/QAIRT/python/images \
      --eval --lean-export
"""
import argparse
import gc
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "export"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modeling_vit_export import ExportQwen3VLViT, output_names  # noqa: E402
# The AIMET export workarounds are hard-won and live in the text tower's script;
# import them rather than re-deriving them. See quantize_aimet.py for the full
# rationale of each (model.pth suppression, all-markers scratch cleanup,
# weight clamp to the symmetric INT8 range).
from quantize_aimet import (  # noqa: E402
    clip_weights_to_7f7f,
    drop_export_scratch,
    suppress_torch_model_dump,
)

DATA = Path(os.environ.get("LLMDEPLOY_DATA", "/home/vinc/llm-local"))
EDGE = 512                      # square input edge; 512/16 = 32 patches a side
GRID = [[1, 32, 32]]            # the ONE grid the exported graph is specialised to
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OPSET = 17                      # matches export_qwen3vl_vit.py

# COCO val2017 image ids, fetched by --fetch-coco. Real photographs, public,
# and reachable over plain http from images.cocodataset.org, which matters
# because the HF proxy path is unreliable (CLAUDE.md Gotchas).
COCO_VAL2017_IDS = [
    139, 285, 632, 724, 776, 785, 802, 872, 885, 1000, 1268, 1296, 1353, 1425,
    1490, 1503, 1532, 1584, 1675, 1761, 1818, 1993, 2006, 2149, 2153, 2157,
    2261, 2299, 2431, 2473, 2532, 2587, 2592, 2685, 2923, 3156, 3255, 3501,
    3553, 3661, 39769,
]


# --------------------------------------------------------------------------
# housekeeping
# --------------------------------------------------------------------------
def disk_guard(need_gb):
    """Shell out to the one disk_guard that exists, in scripts/env.sh.

    Not reimplemented here on purpose: the WSL rule it encodes (check Windows
    C:, not the guest's own df, because a failed vhdx grow delivers SIGBUS
    rather than ENOSPC) is exactly the kind of thing that rots in a second copy.
    """
    env_sh = Path(__file__).resolve().parents[1] / "env.sh"
    r = subprocess.run(["bash", "-c", f'source "{env_sh}" && disk_guard {need_gb}'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(r.stderr.strip() or f"disk_guard {need_gb} refused to proceed")
    print(f"disk_guard {need_gb} GB: ok")


def peak_rss_gib():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


# --------------------------------------------------------------------------
# calibration images
# --------------------------------------------------------------------------
def fetch_coco(dest):
    """Download the COCO val2017 subset into `dest` (idempotent, skips existing).

    Kept in-script so the calibration set is reproducible: encodings that cannot
    be regenerated because the images vanished are encodings nobody can audit.
    """
    import urllib.request
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    got = 0
    for img_id in COCO_VAL2017_IDS:
        name = f"{img_id:012d}.jpg"
        path = dest / name
        if path.exists() and path.stat().st_size > 0:
            got += 1
            continue
        url = f"http://images.cocodataset.org/val2017/{name}"
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                blob = r.read()
            path.write_bytes(blob)
            got += 1
        except Exception as e:                                   # noqa: BLE001
            print(f"  fetch {name}: {type(e).__name__}: {e}")
    print(f"fetch-coco: {got}/{len(COCO_VAL2017_IDS)} images in {dest}")
    return got


def list_images(dirs, min_edge):
    """Every image file under `dirs` with both sides >= min_edge, path-sorted.

    The size floor drops icons, logos and UI sprites, which are mostly flat
    colour and would contribute nothing but a narrow activation range.
    """
    from PIL import Image
    out = []
    for d in dirs:
        d = Path(os.path.expandvars(str(d)))
        if not d.is_dir():
            print(f"  images-dir {d}: not a directory, skipped")
            continue
        n = 0
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() not in IMG_EXT or not p.is_file():
                continue
            try:
                with Image.open(p) as im:      # header only, no decode
                    w, h = im.size
            except Exception:                                    # noqa: BLE001
                continue
            if min(w, h) >= min_edge:
                out.append(p)
                n += 1
        print(f"  images-dir {d}: {n} usable images")
    return out


def to_square(img, edge=EDGE):
    """Shorter side to `edge`, then centre crop. Aspect ratio is preserved, so
    the crop keeps real image statistics instead of stretching them."""
    from PIL import Image
    img = img.convert("RGB")
    w, h = img.size
    s = edge / min(w, h)
    if s != 1.0:
        img = img.resize((max(edge, round(w * s)), max(edge, round(h * s))),
                         Image.LANCZOS)
        w, h = img.size
    left, top = (w - edge) // 2, (h - edge) // 2
    return img.crop((left, top, left + edge, top + edge))


def pixel_values_of(proc, img):
    """Square PIL image -> pixel_values [1024, 1536] via the REAL HF processor.

    The grid assertion is the load-bearing one: the exported graph is
    specialised to exactly one grid_thw, so an image that lands anywhere else
    would silently calibrate a different shape than the one that ships.
    """
    out = proc.image_processor(images=img, return_tensors="pt")
    grid = out["image_grid_thw"]
    assert grid.tolist() == GRID, f"image_grid_thw {grid.tolist()} != {GRID}"
    pv = out["pixel_values"].to(torch.float32)
    n_patches = int(grid[0, 0] * grid[0, 1] * grid[0, 2])
    assert pv.shape[0] == n_patches, f"pixel_values {tuple(pv.shape)} vs {n_patches} patches"
    return pv


# ---- structured synthetic fallback ---------------------------------------
def _pink(rng, edge, beta):
    """1/f**beta noise, per channel. Natural images have ~1/f amplitude spectra;
    uniform noise has a flat one, which is the whole reason this file exists."""
    f = np.fft.fftfreq(edge)
    fx, fy = np.meshgrid(f, f)
    r = np.sqrt(fx ** 2 + fy ** 2)
    r[0, 0] = 1.0 / edge
    amp = r ** (-beta)
    ch = []
    for _ in range(3):
        spec = np.fft.fft2(rng.normal(size=(edge, edge))) * amp
        c = np.fft.ifft2(spec).real
        c -= c.min()
        c /= (c.max() + 1e-9)
        ch.append(c)
    return np.stack(ch, -1)


def synthetic_images(n, seed=0, edge=EDGE):
    """FALLBACK calibration set: structured synthetic images.

    Not photographs. Used only when --images-dir yields too few real images (or
    --synthetic asks for extras). It spans the statistics that matter for
    activation ranges -- smooth gradients, hard edges against flat regions,
    high-frequency texture, 1/f spectra, gratings, document-like layouts, and
    exposure/contrast/colour-balance extremes -- which uniform noise does not.
    """
    from PIL import Image
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:edge, 0:edge] / (edge - 1.0)
    out = []
    styles = ["gradient", "radial", "blocks", "texture", "pink", "grating",
              "document", "extreme"]
    for i in range(n):
        style = styles[i % len(styles)]
        if style == "gradient":                    # smooth, low frequency
            a = rng.uniform(0, 2 * np.pi)
            t = np.cos(a) * xx + np.sin(a) * yy
            t = (t - t.min()) / (t.max() - t.min() + 1e-9)
            c0, c1 = rng.uniform(0, 1, 3), rng.uniform(0, 1, 3)
            im = c0 + t[..., None] * (c1 - c0)
        elif style == "radial":
            cx, cy = rng.uniform(0.2, 0.8, 2)
            r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            r = r / (r.max() + 1e-9)
            im = np.stack([r, 1 - r, np.abs(0.5 - r) * 2], -1)
        elif style == "blocks":                    # hard edges, flat regions
            k = int(rng.integers(2, 17))
            tile = rng.uniform(0, 1, (k, k, 3))
            im = np.kron(tile, np.ones((edge // k + 1, edge // k + 1, 1)))[:edge, :edge]
        elif style == "texture":                   # high frequency, correlated
            base = rng.uniform(0, 1, (edge // 2, edge // 2, 3)).repeat(2, 0).repeat(2, 1)
            im = 0.5 * base + 0.5 * rng.uniform(0, 1, (edge, edge, 3))
        elif style == "pink":
            im = _pink(rng, edge, beta=rng.uniform(0.8, 1.6))
        elif style == "grating":
            fq = rng.uniform(2, 40)
            a = rng.uniform(0, np.pi)
            g = 0.5 + 0.5 * np.sin(2 * np.pi * fq * (np.cos(a) * xx + np.sin(a) * yy))
            im = np.stack([g, np.roll(g, edge // 8, 0), np.roll(g, edge // 8, 1)], -1)
        elif style == "document":                  # flat page + dark text bars
            im = np.full((edge, edge, 3), rng.uniform(0.85, 1.0))
            for _ in range(int(rng.integers(20, 60))):
                y = int(rng.integers(0, edge - 12))
                x = int(rng.integers(0, edge - 40))
                w = int(rng.integers(30, min(300, edge - x)))
                im[y:y + int(rng.integers(4, 12)), x:x + w] = rng.uniform(0, 0.25)
        else:                                      # exposure / contrast extremes
            base = _pink(rng, edge, beta=1.2)
            gain = rng.choice([0.12, 0.25, 3.0, 6.0])
            bias = rng.choice([0.0, 0.0, 0.75, -0.35])
            im = np.clip(base * gain + bias, 0, 1)
        arr = np.clip(im, 0, 1)
        out.append((Image.fromarray((arr * 255).astype(np.uint8)), f"synthetic:{style}{i}"))
    return out


def build_image_sets(args, proc):
    """(calib, held_out) lists of (pixel_values, label), disjoint by construction."""
    paths = list_images(args.images_dir, args.min_edge) if args.images_dir else []
    random.Random(args.shuffle_seed).shuffle(paths)
    need = args.n_calib + args.n_eval
    n_synth = args.synthetic
    if len(paths) < need:
        n_synth = max(n_synth, need - len(paths))
        print(f"WARNING: only {len(paths)} real images for {need} slots -- falling back "
              f"to {n_synth} STRUCTURED SYNTHETIC images. These are not photographs; "
              "treat the resulting activation encodings as a quality risk.")
    from PIL import Image
    items = []
    for p in paths[:need]:
        with Image.open(p) as im:
            items.append((to_square(im), str(p)))
    items += synthetic_images(n_synth, seed=args.shuffle_seed)
    # interleave so a synthetic top-up cannot land entirely in one split
    random.Random(args.shuffle_seed + 1).shuffle(items)
    assert len(items) >= need, f"only {len(items)} calibration images for {need} slots"

    def prep(chunk):
        return [(pixel_values_of(proc, img), label) for img, label in chunk]

    calib = prep(items[:args.n_calib])
    held = prep(items[args.n_calib:need])
    return calib, held


# --------------------------------------------------------------------------
# quality gate
# --------------------------------------------------------------------------
def compare(name, a, b):
    """max|d| and cosine for one output, parity_vit.py's house style.

    Finiteness is asserted BEFORE any reduction: max(0.0, nan) is 0.0 and
    min(1.0, nan) is 1.0, so a NaN reaching the reduction is silently swallowed
    and the gate exits 0 on a graph emitting garbage. Documented past bug.
    """
    assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
    assert np.isfinite(a).all(), f"{name}: quantsim output has non-finite values"
    assert np.isfinite(b).all(), f"{name}: FP32 reference has non-finite values"
    assert float(np.abs(a).max()) > 0.0, f"{name}: quantsim output is all zeros"
    assert float(a.std()) > 0.0, f"{name}: quantsim output is constant"
    assert float(b.std()) > 0.0, f"{name}: FP32 reference is constant"
    d = float(np.abs(a - b).max())
    # float64 for the reduction: in float32 over 655k elements the dot product
    # returns values like 1.00000012, which is impossible for a cosine.
    af, bf = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    cos = float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf)))
    return d, cos, float(np.abs(b).max())


def run_eval(sim, model, held, names):
    print(f"== eval: {len(held)} held-out images, quantsim vs FP32 ==", flush=True)
    stats = {n: {"d": [], "cos": [], "ref": []} for n in names}
    for pv, label in held:
        with torch.no_grad():
            q = sim.model(pv)
            f = model(pv)
        print(f"[eval] {label}")
        for name, qa, fa in zip(names, q, f):
            d, cos, ref = compare(name, qa.cpu().numpy(), fa.cpu().numpy())
            stats[name]["d"].append(d)
            stats[name]["cos"].append(cos)
            stats[name]["ref"].append(ref)
            print(f"    {name:26s} max|d|={d:.4e}  cos={cos:.6f}  "
                  f"(fp32 absmax {ref:.3f}, rel {d / ref:.3%})")
    print("== eval summary (worst over held-out images) ==")
    for name in names:
        s = stats[name]
        print(f"  {name:26s} max|d|={max(s['d']):.4e}  min cos={min(s['cos']):.6f}  "
              f"mean cos={sum(s['cos']) / len(s['cos']):.6f}  "
              f"worst rel={max(d / r for d, r in zip(s['d'], s['ref'])):.3%}")
    return stats


# --------------------------------------------------------------------------
# export post-checks
# --------------------------------------------------------------------------
def ensure_io_names(out, names_out):
    """Make sure model.onnx / model.encodings use the canonical I/O names.

    AIMET's quantsim.export re-exports the graph and has historically mangled
    I/O names (onnx::Cast_0, t.10601, bare numerics) -- see rename_aimet_io.py.
    onnx_export_args normally prevents that; this is the belt-and-braces path,
    positional exactly like rename_aimet_io.py (torch.onnx.export preserves
    forward-arg and return order) and a no-op when the names already match.
    """
    import onnx
    out = Path(out)
    m = onnx.load(str(out / "model.onnx"), load_external_data=False)
    g = m.graph
    assert len(g.input) == 1, (
        f"graph has {len(g.input)} inputs {[t.name for t in g.input]}, expected 1 "
        "-- the cu_seqlens/position constant fold did not happen")
    assert len(g.output) == len(names_out), (
        f"graph has {len(g.output)} outputs, expected {len(names_out)}")
    rename = {}
    for vi, new in zip(list(g.input) + list(g.output), ["pixel_values"] + names_out):
        if vi.name != new:
            rename[vi.name] = new
    if not rename:
        print("graph I/O names already canonical")
        return
    print(f"renaming mangled AIMET I/O names: {rename}")

    def fix(n):
        return rename.get(n, n)

    for vi in list(g.input) + list(g.output) + list(g.value_info):
        vi.name = fix(vi.name)
    for node in g.node:
        node.input[:] = [fix(x) for x in node.input]
        node.output[:] = [fix(x) for x in node.output]
    # initializers keep data_location=EXTERNAL, so this rewrites the proto only
    onnx.save(m, str(out / "model.onnx"))
    for fn in ("model.encodings", "model_torch.encodings"):
        p = out / fn
        if not p.exists():
            continue
        enc = json.loads(p.read_text())
        for section in ("activation_encodings", "param_encodings"):
            sec = enc.get(section)
            if isinstance(sec, dict):
                enc[section] = {fix(k): v for k, v in sec.items()}
            elif isinstance(sec, list):
                for e in sec:
                    e["name"] = fix(e.get("name"))
        p.write_text(json.dumps(enc, indent=2))
    print("rewrote model.onnx and the encodings files with canonical names")


def report_encodings(out, names_out):
    """Assert and print the encodings that make the converter emit UFIXED_16 I/O.

    This is the entire point of the task: without an activation encoding on
    `pixel_values` and on each output, qairt-converter has nothing to override
    with and the graph keeps float I/O -- which the Genie image path cannot
    drive (see the module docstring). So it asserts rather than warns.
    """
    enc = json.loads((Path(out) / "model.encodings").read_text())
    acts = enc.get("activation_encodings")
    if isinstance(acts, list):
        by_name = {e.get("name"): e for e in acts}
    else:
        by_name = dict(acts or {})
    print(f"== model.encodings: {len(by_name)} activation, "
          f"{len(enc.get('param_encodings') or [])} param entries "
          f"(version {enc.get('version')}) ==")
    missing = []
    for name in ["pixel_values"] + names_out:
        e = by_name.get(name)
        if e is None:
            missing.append(name)
            print(f"  {name:26s} MISSING")
            continue
        if isinstance(e, list):        # 0.6.1 schema: list of per-channel dicts
            e = e[0]
        scale = e.get("scale", e.get("scale_"))
        offset = e.get("offset")
        print(f"  {name:26s} dtype={e.get('dtype')} bw={e.get('bw')} "
              f"is_sym={e.get('is_sym')} enc_type={e.get('enc_type')} "
              f"scale={scale} offset={offset}")
    assert not missing, (
        f"model.encodings has no activation encoding for {missing} -- the "
        "converter cannot emit UFIXED_16 I/O without them, and the Genie image "
        "path cannot drive a float-IO vision tower")
    return by_name


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DATA / "models/Qwen3-VL-4B-Instruct"))
    ap.add_argument("--out", default=str(DATA / "work/quant/qwen3vl-4b-vit"))
    ap.add_argument("--images-dir", action="append", default=[],
                    help="directory of real calibration images (repeatable)")
    ap.add_argument("--fetch-coco", metavar="DIR", nargs="?", const=str(
        DATA / "work/calib/images-coco-val2017"),
        help="download the COCO val2017 subset into DIR and add it to --images-dir")
    ap.add_argument("--n-calib", type=int, default=24)
    ap.add_argument("--n-eval", type=int, default=6,
                    help="held-out images for --eval; disjoint from calibration")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="append N structured synthetic images (fallback set)")
    ap.add_argument("--min-edge", type=int, default=256,
                    help="skip images whose shorter side is below this (icons/logos)")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--weight-bw", type=int, default=8, choices=(4, 8))
    ap.add_argument("--act-bw", type=int, default=16, choices=(8, 16))
    ap.add_argument("--no-clip", action="store_true",
                    help="skip clip_weights_to_7f7f (the text tower applies it)")
    # aimet-torch 2.36 ships the exact config the SA8797P summary names
    ap.add_argument("--config", default=str(
        Path(torch.__file__).parent.parent /
        "aimet_torch/common/quantsim_config/htp_quantsim_config_v81_per_channel_linear.json"))
    # CPU by default: CLAUDE.md mandates QUANT_DEVICE=cpu above 0.6B, and the
    # quantsim graph is far larger than the tower's own weights on an 8 GB card.
    ap.add_argument("--device", default=os.environ.get("QUANT_DEVICE", "cpu"))
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--lean-export", action="store_true",
                    help="drop quantsim's two disposable full-size copies "
                         "(model.pth, all-markers scratch). Safe here: the "
                         "corruption it causes is cross-stage encodings "
                         "adoption, and the ViT is a single-graph build.")
    args = ap.parse_args()

    if args.fetch_coco:
        fetch_coco(args.fetch_coco)
        args.images_dir.append(args.fetch_coco)

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from aimet_torch.common.defs import QuantScheme
    from aimet_torch.quantsim import QuantizationSimModel
    from aimet_torch.onnx_utils import OnnxExportApiArgs

    print("== calibration images ==", flush=True)
    proc = AutoProcessor.from_pretrained(args.model, min_pixels=EDGE * EDGE,
                                         max_pixels=EDGE * EDGE)
    calib, held = build_image_sets(args, proc)
    print(f"calibration: {len(calib)} images; held-out: {len(held)}")
    for pv, label in calib:
        print(f"  [calib] {label}  pixel_values{tuple(pv.shape)} "
              f"[{float(pv.min()):+.3f},{float(pv.max()):+.3f}]")
    for pv, label in held:
        print(f"  [eval ] {label}")

    # eager is mandatory: wrapper parity was established against the eager path,
    # and an sdpa-loaded model traces a different attention subgraph.
    print("== loading checkpoint (fp32, eager) ==", flush=True)
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager").eval()
    vision = hf.model.visual
    n_deepstack = len(vision.deepstack_visual_indexes)
    names_out = output_names(n_deepstack)

    grid_thw = torch.tensor(GRID, dtype=torch.long)
    model = ExportQwen3VLViT(vision, grid_thw).eval().to(args.device)
    del hf, vision
    gc.collect()      # the VL checkpoint is ~17.6 GB fp32; do not wait for cycles

    dummy = (calib[0][0].to(args.device),)
    calib = [(pv.to(args.device), lb) for pv, lb in calib]
    held = [(pv.to(args.device), lb) for pv, lb in held]

    print(f"== quantsim W{args.weight_bw}A{args.act_bw} on {args.device} ==", flush=True)
    sim = QuantizationSimModel(
        model,
        dummy_input=dummy,
        quant_scheme=QuantScheme.post_training_tf_enhanced,
        default_param_bw=args.weight_bw,
        default_output_bw=args.act_bw,
        config_file=args.config,
    )

    def calibrate(m, _):
        with torch.no_grad():
            for i, (pv, label) in enumerate(calib):
                m(pv)
                print(f"  calib {i + 1}/{len(calib)}  {label}", flush=True)

    sim.compute_encodings(calibrate, None)
    if not args.no_clip:
        # Symmetric weight quantizers only -- the ViT is what exposed the bug
        # where this clamped asymmetric nn.LayerNorm gains 1.0 -> 0.498; see
        # quantize_aimet.clip_weights_to_7f7f.
        clip_weights_to_7f7f(sim)

    if args.eval:
        run_eval(sim, model, held, names_out)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # The export writes one full-size ONNX (~1.6 GB) plus, unless --lean-export,
    # a model.pth and an all-markers scratch export of the same size each.
    disk_guard(6 if args.lean_export else 12)

    # aimet-torch 2.36.0 bug: quantsim.export references nn.lora.QuantizedLora,
    # which this build doesn't define. Provide an inert shim class.
    import aimet_torch.nn.lora as _lora
    if not hasattr(_lora, "QuantizedLora"):
        _lora.QuantizedLora = type("QuantizedLoraShim", (), {})
    # aimet-torch 2.36.0 bug #2: _onnx_model_size_larger_than_max_protobuf calls
    # proto.ByteSize(), which itself raises EncodeError past 2 GB. Force the
    # external-data path unconditionally. (Both patches: see quantize_aimet.py.)
    import aimet_torch.onnx_utils as _onnx_utils
    _onnx_utils._onnx_model_size_larger_than_max_protobuf = lambda _m: True
    if args.lean_export:
        suppress_torch_model_dump()

    dummy = tuple(t.cpu() for t in dummy)
    del model, calib, held
    gc.collect()
    import ctypes
    try:                       # glibc keeps freed arenas mapped; ask for them back
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass

    print("== export ==", flush=True)
    sim.export(str(out), "model", dummy_input=dummy,
               onnx_export_args=OnnxExportApiArgs(
                   opset_version=OPSET,
                   input_names=["pixel_values"],
                   output_names=names_out))
    if args.lean_export:
        drop_export_scratch(out)
    ensure_io_names(out, names_out)
    report_encodings(out, names_out)
    print(f"exported quantsim to {out} (model.onnx + model.encodings)")
    print(f"peak RSS {peak_rss_gib():.2f} GiB")


if __name__ == "__main__":
    main()
