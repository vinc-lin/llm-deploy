#!/usr/bin/env python
"""Pick the kit's final images by asking the model what it actually sees.

Filenames and search terms are claims about content; a caption is evidence.
This runs every candidate through HF `Qwen3VLForConditionalGeneration.generate`
with the kit's own general-purpose prompt and the kit's own 512x512
preprocessing, then selects a set whose captions jointly cover the scenes the
deployment cares about -- rain, fog/overcast, snow, clear, and a road with
vehicles.

Selection is on the CAPTION TEXT, so it cannot drift from what the model
perceives, and the recorded caption doubles as the fp32 reference the device
output is judged against.

This is the fp32-with-deepstack reference. It is NOT what the device produces:
the device runs W8A16 and has no deepstack path, so its captions are generated
separately by the e2e gate's device-faithful chain. Both go in the kit README.

  $PY_DEPLOY scripts/pipeline/select_test_images.py \
      --candidates work/kit-candidates --model $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct
"""
import argparse
import json
import re
import time
from pathlib import Path

EDGE = 512
PROMPT = "Describe this image in one sentence."

# scene -> words whose presence in a caption is evidence FOR that scene.
SCENE_WORDS = {
    "rain": ("rain", "rainy", "wet", "puddle", "umbrella", "downpour", "drizzl"),
    "fog": ("fog", "foggy", "mist", "misty", "haze", "hazy"),
    "snow": ("snow", "snowy", "snowing", "winter", "blizzard", "icy", "ice"),
    "clear": ("sunny", "clear", "blue sky", "sunlight", "sunshine", "bright"),
    "overcast": ("overcast", "cloudy", "grey sky", "gray sky", "clouds", "gloomy"),
    "vehicles": ("car", "cars", "truck", "bus", "vehicle", "traffic", "van",
                 "motorcycle", "parked"),
    "road": ("road", "street", "highway", "roadway", "intersection", "lane",
             "avenue", "sidewalk"),
}
# The kit must cover at least these. "vehicles"/"road" are the deployment
# context (a camera outside a vehicle), the rest are the weather axis.
REQUIRED = ("rain", "snow", "road")
WEATHER_AXIS = ("rain", "fog", "snow", "clear", "overcast")


def scenes_in(caption):
    c = caption.lower()
    return {s for s, words in SCENE_WORDS.items() if any(w in c for w in words)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None, help="default <candidates>/selection.json")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--want", type=int, default=6)
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    cdir = Path(args.candidates)
    raw = json.loads((cdir / "candidates.json").read_text())
    # Defensive: one photo can match several scene searches, and captioning the
    # same bytes twice would let a single image occupy two kit slots.
    manifest, seen = [], set()
    for e in raw:
        key = e.get("file")
        if key in seen:
            print(f"  dropping duplicate manifest entry for {key}")
            continue
        seen.add(key)
        if not (cdir / key).is_file():
            print(f"  dropping {key}: file missing")
            continue
        manifest.append(e)
    print(f"{len(manifest)} candidates from {cdir} ({len(raw)} manifest entries)")

    proc = AutoProcessor.from_pretrained(
        args.model, min_pixels=EDGE * EDGE, max_pixels=EDGE * EDGE)
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager").eval()

    t0 = time.time()
    for i, entry in enumerate(manifest):
        p = cdir / entry["file"]
        img = Image.open(p).convert("RGB").resize((EDGE, EDGE), Image.BICUBIC)
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": args.prompt}]}]
        text = proc.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True)
        batch = proc(text=[text], images=[img], return_tensors="pt")
        with torch.no_grad():
            gen = hf.generate(**batch, max_new_tokens=args.max_new_tokens,
                              do_sample=False, temperature=None, top_p=None,
                              top_k=None)
        cap = proc.batch_decode(gen[:, batch["input_ids"].shape[1]:],
                                skip_special_tokens=True)[0].strip()
        cap = re.sub(r"\s+", " ", cap)
        entry["caption_hf_fp32"] = cap
        entry["scenes"] = sorted(scenes_in(cap))
        print(f"  [{i + 1}/{len(manifest)}] {entry['file'][:44]:46s} "
              f"{entry['scenes']}\n        {cap[:150]}", flush=True)
    print(f"  captioning wall {time.time() - t0:.0f}s")

    # -- greedy cover: take whichever candidate adds the most missing scenes --
    chosen, covered = [], set()
    pool = list(manifest)
    while pool and len(chosen) < args.want:
        pool.sort(key=lambda e: (-len(set(e["scenes"]) - covered),
                                 -len(e["scenes"])))
        best = pool.pop(0)
        gain = set(best["scenes"]) - covered
        if not gain and len(chosen) >= len(REQUIRED):
            break
        chosen.append(best)
        covered |= set(best["scenes"])

    missing_req = [s for s in REQUIRED if s not in covered]
    weather = [s for s in WEATHER_AXIS if s in covered]
    print(f"\nselected {len(chosen)}; scenes covered {sorted(covered)}")
    print(f"weather axis covered: {weather}")

    for e in chosen:
        stem = "wx_" + "_".join(sorted(set(e["scenes"]) & set(WEATHER_AXIS))
                                or ["scene"])
        e["kit_stem"] = stem
    # disambiguate collisions deterministically
    seen = {}
    for e in chosen:
        n = seen.get(e["kit_stem"], 0)
        seen[e["kit_stem"]] = n + 1
        if n:
            e["kit_stem"] = f"{e['kit_stem']}{n + 1}"

    out = Path(args.out) if args.out else cdir / "selection.json"
    out.write_text(json.dumps(chosen, indent=2) + "\n")
    print(f"wrote {out}")
    for e in chosen:
        print(f"  {e['kit_stem']:16s} <- {e['file']}")

    if missing_req:
        raise SystemExit(
            f"FAIL: required scene(s) {missing_req} not covered by any caption. "
            "Widen the candidate pool (fetch_test_images.py --per-scene) rather "
            "than relabelling an image the model does not actually see that way.")
    if len(chosen) < 5:
        raise SystemExit(f"FAIL: only {len(chosen)} image(s) selected, want >= 5")
    print("PASS: kit selection covers the required scenes")


if __name__ == "__main__":
    main()
