#!/usr/bin/env python
"""Generate the test kit's expected-output references.

WHY THE KIT NEEDS THESE
-----------------------
A throughput number from a broken graph is worse than no number: W8A16 defects
on this pipeline have historically produced output that is *fluent but wrong*
rather than obvious garbage (the 2026-08-11 last-token-only logits head shipped
and read out of bounds without ever looking broken). The device team cannot
judge that by eye, so they need a reference to diff against.

These are greedy continuations from the HF model in fp32 -- the same reference
`quantize_aimet.py --eval` scores against. An INT8 device run will not match
token-for-token forever; what matters is that it tracks for the first tens of
tokens and stays on-topic and well-formed after that. The threshold is recorded
in the kit README rather than enforced here, because the call is a human one.

Run:
  gen_kit_expected.py [--max-new-tokens 48]
"""
import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = Path(os.environ.get("LLMDEPLOY_DATA", "/home/vinc/llm-local"))
ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "kit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DATA / "models/Qwen3-0.6B"))
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", default=str(KIT / "expected"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()

    index = []
    for prompt_file in sorted((KIT / "prompts").glob("*.txt")):
        text = prompt_file.read_text()
        ids = tok(text, return_tensors="pt").input_ids
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None,
                                 top_k=None, pad_token_id=tok.eos_token_id)
        cont = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=False)
        dst = out / f"{prompt_file.stem}.txt"
        dst.write_text(cont)
        index.append((prompt_file.stem, ids.shape[1], cont))
        print(f"{prompt_file.stem}: prompt={ids.shape[1]} tokens -> {dst.name}")

    readme = ["# Expected outputs (greedy, HF fp32 reference)",
              "",
              "Diff the first ~30 tokens of each arm's `stdout_r1.txt` against these.",
              "An INT8 device run will not match forever -- early divergence is the",
              "signal, not eventual divergence. Flag an arm if it diverges in the",
              "first few tokens, or if it degenerates into repetition.",
              "",
              "| prompt | prompt tokens |",
              "|---|---:|"]
    for name, n, _ in index:
        readme.append(f"| `{name}` | {n} |")
    readme += ["",
               "The `technical` prompt is 56 tokens, matching the 2026-08-13 report's",
               "protocol exactly, so its rates are directly comparable to the",
               "11.72 / 9.18 tok/s baselines."]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    print(f"\nwrote {len(index)} references + README to {out}")


if __name__ == "__main__":
    main()
