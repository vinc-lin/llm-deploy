#!/usr/bin/env python
"""Assemble the device test kit from the selected images.

Per selected image this writes, all FLAT (the bundle is flat by design --
Genie resolves config-referenced paths from the bundle root):

  wx_<scene>.jpg     the exact 512x512 RGB the device sees, for human eyes
  wx_<scene>.raw     UFixed16 pixel_values quantized against the SHIPPED ViT
                     ctx-bin's own encoding -- Genie does no preprocessing,
                     it feeds this file as an opaque blob
  wx_<scene>.json    sidecar: grid, encoding, clip statistics
  wx_<scene>.script  the pipeline script with only the image path changed

and TEST_IMAGES.md, the table the device team reads.

The clip gate is the interesting one: a real photograph whose pixels fall
more than CLIP_EXCESS_LSB_MAX outside the graph's calibrated input range is
NOT a kit image, it is a finding -- the ViT was calibrated on COCO
photographs and the device would clamp anything outside that. It is reported,
not silently accepted.

Expected captions are NOT computed here. They come from the e2e gate itself
(`parity_e2e_vl.py --image`), so the kit's references and the gate's numerics
can never drift apart; this script splices them in when given --captions.

  $PY_DEPLOY scripts/pipeline/build_test_kit.py \
      --selection work/kit-candidates/selection.json \
      --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct \
      --encodings <bundle>/qwen3vl-4b-vit-w8a16_ctx.info.json \
      --script configs/genie_pipeline_qwen3vl.script --out work/kit
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_image import (  # noqa: E402
    EDGE, RAW_BYTES, load_encoding, preprocess, quantize, CLIP_EXCESS_LSB_MAX)

SAMPLE_LINE = "sample_image.raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--encodings", required=True,
                    help="the SHIPPED ViT ctx-bin info.json")
    ap.add_argument("--script", required=True, help="base pipeline script")
    ap.add_argument("--out", required=True)
    ap.add_argument("--captions", default=None,
                    help="json {kit_stem: device_caption} from the e2e gate")
    args = ap.parse_args()

    sel = json.loads(Path(args.selection).read_text())
    cdir = Path(args.selection).parent
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.script).read_text()
    assert SAMPLE_LINE in base, f"{args.script} does not reference {SAMPLE_LINE}"
    devcaps = json.loads(Path(args.captions).read_text()) if args.captions else {}

    scale, offset, bw = load_encoding(args.encodings)
    print(f"ViT pixel_values encoding: bw={bw} scale={scale:.9e} offset={offset}")

    from PIL import Image
    rows, problems = [], []
    for e in sel:
        stem = e["kit_stem"]
        src = cdir / e["file"]
        jpg = out / f"{stem}.jpg"
        Image.open(src).convert("RGB").resize((EDGE, EDGE),
                                              Image.BICUBIC).save(jpg, quality=95)

        pv, grid = preprocess(args.model, jpg)
        q, clipped, excess, err = quantize(pv, scale, offset, bw)
        status = "ok"
        if excess > CLIP_EXCESS_LSB_MAX:
            status = f"OUT OF CALIBRATION RANGE ({excess:.1f} LSB)"
            problems.append(f"{stem}: pixels fall up to {excess:.1f} LSB outside "
                            "the ViT's calibrated range")
        raw = out / f"{stem}.raw"
        raw.write_bytes(q.tobytes())
        assert raw.stat().st_size == RAW_BYTES, raw.stat().st_size
        (out / f"{stem}.json").write_text(json.dumps(
            {"shape": list(pv.shape), "dtype": "uint16", "bytes": RAW_BYTES,
             "grid_thw": grid.tolist(), "edge": EDGE,
             "encoding": {"scale": scale, "offset": offset, "bitwidth": bw},
             "clipped": clipped, "clip_excess_lsb": excess,
             "max_roundtrip_err": err,
             "source": {k: e.get(k) for k in
                        ("title", "url", "licence", "author", "descurl",
                         "source", "scene")}}, indent=1) + "\n")
        (out / f"{stem}.script").write_text(base.replace(SAMPLE_LINE, f"{stem}.raw"))

        print(f"  {stem:22s} clip {excess:5.2f} LSB ({clipped}/{q.size})  {status}")
        rows.append({
            "stem": stem, "scenes": e.get("scenes", []),
            "hf": e.get("caption_hf_fp32", ""), "device": devcaps.get(stem, ""),
            "licence": e.get("licence", "?"), "author": e.get("author", "?"),
            "title": e.get("title", ""), "url": e.get("descurl") or e.get("url", ""),
            "excess": excess, "status": status,
        })

    md = ["# Weather / road test images",
          "",
          "Real outdoor photographs covering the deployment's scenes: the view",
          "outside a vehicle -- road, surroundings, weather. Each ships the exact",
          "512x512 JPEG the device sees, the preprocessed `UFixed16` blob, a",
          "sidecar, and its own pipeline script.",
          "",
          "```",
          "LD_LIBRARY_PATH=. ./genie-app -s wx_<scene>.script",
          "```",
          "",
          "## What to compare against",
          "",
          "Two references per image, and they are NOT the same thing:",
          "",
          "* **device-faithful** -- the e2e parity gate's own chain with deepstack",
          "  zeroed, which is exactly what a stock Genie pipeline runs (no",
          "  deepstack path exists). This is the one to diff against.",
          "* **HF fp32** -- the full model with deepstack, for context only. The",
          "  device cannot reach this; the gap between the two columns IS the",
          "  deepstack-by-zeros degradation.",
          "",
          "Neither will match token-for-token: the device is W8A16. **The bar is",
          "semantic agreement on the weather and the scene contents**, not wording.",
          "Flag an image only if the caption describes a different scene, or",
          "degenerates into repetition.",
          "",
          "| image | scenes | expected (device-faithful) | HF fp32 reference |",
          "|---|---|---|---|"]
    for r in rows:
        md.append(f"| `{r['stem']}` | {', '.join(r['scenes'])} | "
                  f"{r['device'] or '_(pending)_'} | {r['hf']} |")
    md += ["",
           "## Preprocessing",
           "",
           "Every `.raw` is quantized against the **shipped ViT ctx-bin's own**",
           f"`pixel_values` encoding (bw={bw}, scale={scale:.9e}, offset={offset}),",
           "not against `model.encodings`. Genie does no preprocessing -- it feeds",
           "the file straight to the graph -- so these bytes must already be what",
           "the tensor expects.",
           "",
           "`clip_excess_lsb` in each sidecar says how far outside the graph's",
           "calibrated input range that image's pixels fell. The ViT was calibrated",
           "on COCO photographs; a large value means the device would clamp, and",
           "that is a recalibration finding rather than a bad image.",
           "",
           "| image | clip excess (LSB) | status |",
           "|---|---:|---|"]
    for r in rows:
        md.append(f"| `{r['stem']}` | {r['excess']:.2f} | {r['status']} |")
    md += ["",
           "## Provenance and licensing",
           "",
           "| image | source | licence | author |",
           "|---|---|---|---|"]
    for r in rows:
        t = f"[{r['title']}]({r['url']})" if r["url"] else r["title"]
        md.append(f"| `{r['stem']}` | {t} | {r['licence']} | {r['author']} |")
    md.append("")
    (out / "TEST_IMAGES.md").write_text("\n".join(md) + "\n")

    print(f"\nwrote {len(rows)} image(s) + TEST_IMAGES.md to {out}")
    if problems:
        print("\nCALIBRATION FINDINGS (reported, not hidden):")
        for p in problems:
            print(f"  - {p}")
    missing = [r["stem"] for r in rows if not r["device"]]
    if missing:
        print(f"\nNOTE: no device-faithful caption yet for {missing}. Run "
              "parity_e2e_vl.py --image per image and re-run with --captions.")


if __name__ == "__main__":
    main()
