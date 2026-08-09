#!/usr/bin/env python
"""Reference outputs from HF Qwen3 for parity checking of the export pipeline."""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = Path(os.environ.get("LLMDEPLOY_DATA", "/home/vinc/llm-local"))

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "解释一下什么是注意力机制。",
    "1+2+3+...+100 =",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DATA / "models/Qwen3-0.6B"))
    ap.add_argument("--out", default=str(DATA / "work/reference"))
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()

    texts = {}
    for i, p in enumerate(PROMPTS):
        ids = tok(p, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = model(ids).logits
            gen = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        torch.save({"prompt": p, "input_ids": ids, "logits": logits, "greedy_ids": gen}, out / f"ref_{i}.pt")
        texts[p] = tok.decode(gen[0], skip_special_tokens=True)

    with open(out / "greedy_texts.json", "w") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    print(json.dumps(texts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
