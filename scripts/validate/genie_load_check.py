#!/usr/bin/env python
"""Run the Genie load simulation against raw ctx-bin info.jsons.

Same engine as `lint_pipeline_bundle.py` check 9, but without needing an
assembled bundle -- so the ctx-bin build can gate on it immediately after
generation, and Phase 6 can re-run it against a re-downloaded info.json to
prove the UPLOADED bytes load, not just the local ones.

Shard order is the ctx-bin load order and matters: the variant map is
populated from the first split only. Pass the info.jsons in the same order
the node config lists its `ctx-bins`.

  $PY_DEPLOY scripts/validate/genie_load_check.py \
      --info a_1_of_2.info.json a_2_of_2.info.json \
      --config configs/genie_text_generator_qwen3vl_4b.json

  # or without a config:
      --ctx-size 2048 --kv-dim 128 --rope-dim 64
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lint_pipeline_bundle import load_graphs, simulate_genie_load    # noqa: E402


class _Rep:
    def fail(self, msg):
        print(f"  FAIL {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", nargs="+", required=True,
                    help="ctx-bin info.json files, in ctx-bin load order")
    ap.add_argument("--config", default=None,
                    help="text-generator node config to read the params from")
    ap.add_argument("--ctx-size", type=int, default=None)
    ap.add_argument("--kv-dim", type=int, default=None)
    ap.add_argument("--rope-dim", type=int, default=None)
    args = ap.parse_args()

    ctx_size, kv_dim, rope_dim = args.ctx_size, args.kv_dim, args.rope_dim
    if args.config:
        doc = json.loads(Path(args.config).read_text())
        body = doc[next(iter(doc))]
        eng = body.get("engine") or {}
        backend = eng.get("backend") or {}
        sub = backend.get(backend.get("type")) or {}
        pe = (eng.get("model") or {}).get("positional-encoding") or {}
        ctx_size = ctx_size or (body.get("context") or {}).get("size")
        kv_dim = kv_dim or sub.get("kv-dim")
        rope_dim = rope_dim or pe.get("rope-dim")
    if kv_dim is None or rope_dim is None:
        sys.exit(f"need kv-dim and rope-dim (got {kv_dim}, {rope_dim})")

    rep = _Rep()
    shards = []
    for p in args.info:
        g = load_graphs(Path(p), rep)
        if g is None:
            sys.exit(f"unreadable: {p}")
        shards.append(g)

    print(f"shards (load order): {[Path(p).name for p in args.info]}")
    print(f"context.size={ctx_size} kv-dim={kv_dim} rope-dim={rope_dim}")
    errors, facts = simulate_genie_load(shards, ctx_size, kv_dim, rope_dim)
    for name, shard, ar, cl, gtype in facts.get("graphs", []):
        print(f"  {name}: shard {shard}, AR={ar}, CL={cl}, {gtype}")
    if "cache_group_ctx_size" in facts:
        print(f"  cache group 'past_': ctx={facts['cache_group_ctx_size']}, "
              f"{'scatter' if facts['use_scatter'] else 'concat'}, "
              f"variant map {facts.get('variant_map')}")
    n_enc = facts.get("encoding_names_checked", 0)
    if facts.get("encoding_check_applicable"):
        print(f"  cross-graph encodings: {n_enc} tensor name(s) compared")
    else:
        print("  cross-graph encodings: N/A -- every cross-graph tensor is "
              "FLOAT_16 (unquantized), so Check 4 has nothing to compare. "
              "It becomes live only for a KV-INT8 build.")
    if errors:
        print(f"\nGENIE LOAD WOULD FAIL ({len(errors)} error(s)):")
        for gname, tname, msg in errors:
            print(f"  {gname} : {tname} - {msg}")
        sys.exit(1)
    print("\nPASS: validateModel replay clean -- these ctx-bins would load")


if __name__ == "__main__":
    main()
