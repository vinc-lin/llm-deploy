#!/usr/bin/env python
"""What does the REAL model answer, in one word, about each test image?

Test K Stage K2 asked the device for a one-word answer per image and judged the
first word by eye. Four images gave a clean word (Red / Rain / Snow / Snow) and
three gave an `S`-initial fragment that was recorded as WRONG or DEGENERATE.

But `Sunny` tokenizes as ['S', 'unny'] -- two tokens -- while `Snow`, `Rain` and
`Red` are ONE token each. So an `S` followed by garbage is exactly what a correct
first token plus a broken decode step 1 looks like, and the three "failures" may
be nothing of the kind. That is an inference; this script measures it.

For each image it prints the greedy continuation of the SAME chat-templated
prompt the bundle ships, as token ids, so the device's first token can be
compared against ground truth rather than against a guess.

The images are resized to 512x512 to reproduce the pipeline's 1024 patches ->
256 image tokens after the 2x2 merge, i.e. the same 3-chunk prefill the device
runs.

    $PY_DEPLOY scripts/validate/host_first_token_vl.py [--n 6]
"""
import argparse
import os
from pathlib import Path

import torch
from PIL import Image

MODEL = os.environ.get("VL_MODEL", "/home/vinc/llm-local/models/Qwen3-VL-4B-Instruct")
IMGDIR = Path(os.environ.get(
    "VL_IMAGES", "/home/vinc/llm-local/bundles/qwen3vl_v5_session/03_vl4b_v5"))

# Exactly the two prompts shipped in deliverables/qwen3vl_testk_session/prompts/
WEATHER = "Answer with one word: what is the weather in this photo?"
SHAPE = "Answer with one word: what colour is the largest shape in this image?"

CASES = [
    ("sample_image.png", SHAPE),
    ("wx_clear.jpg", WEATHER),
    ("wx_clear2.jpg", WEATHER),
    ("wx_clear_snow.jpg", WEATHER),
    ("wx_fog_overcast_rain.jpg", WEATHER),
    ("wx_snow.jpg", WEATHER),
    ("wx_snow2.jpg", WEATHER),
]

# What the device produced, for side-by-side reading (Test K report, 2026-08-21).
DEVICE = {
    "sample_image.png": "Red",
    "wx_clear.jpg": "S + 000...",
    "wx_clear2.jpg": "Sally...",
    "wx_clear_snow.jpg": "Ser [...]",
    "wx_fog_overcast_rain.jpg": "Rain",
    "wx_snow.jpg": "Snow",
    "wx_snow2.jpg": "Snow [er...]",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="tokens to generate per image")
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForImageTextToText

    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.float32, device_map="cpu").eval()

    print(f"model  {MODEL}")
    print(f"images {IMGDIR}\n")
    hdr = f"{'image':26s} {'device first word':16s} {'HOST greedy':34s} first ids"
    print(hdr)
    print("-" * len(hdr))

    for name, prompt in CASES:
        path = IMGDIR / name
        if not path.exists():
            print(f"{name:26s} -- MISSING --")
            continue
        # 512x512 -> 32x32 patches -> 1024 patches -> 256 image tokens after merge
        img = Image.open(path).convert("RGB").resize((512, 512), Image.BICUBIC)
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt")
        n_prompt = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.n,
                                 do_sample=False, num_beams=1)
        new = out[0][n_prompt:].tolist()
        toks = proc.tokenizer.convert_ids_to_tokens(new)
        txt = proc.tokenizer.decode(new).replace("\n", "\\n")
        print(f"{name:26s} {DEVICE.get(name,'?'):16s} {txt[:34]:34s} {new}")
        print(f"{'':26s} {'':16s} {'':34s} {toks}")
        print(f"{'':26s} prompt tokens: {n_prompt}")

    print("\nRead: if the HOST's first token equals the device's first token, the "
          "device's PREFILL is right for that image and only the tokens after it "
          "are the known decode defect.")


if __name__ == "__main__":
    main()
