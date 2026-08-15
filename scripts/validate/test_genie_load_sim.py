#!/usr/bin/env python
"""Mutation test for lint check 9 (`simulate_genie_load`).

A load simulation that cannot reproduce the failure it exists to prevent is
vacuous. This asserts both directions against real bytes:

  MUTATION  the info.jsons of the bundle that actually died on device
            (2026-08-14) must fail with the device's exact message,
            `Expected [ 1, 128, 2176] Found [ 1, 128, 128]`, on BOTH shards --
            because the variant map is keyed by (AR, CL) and populated from
            shard 0 only, so shard 1 inherits shard 0's rewritten CL.

  CONTROL   the same tower rebuilt to the ground-truth past-KV shapes must
            pass cleanly.

  PROBES    each individual rule is broken in isolation and must be caught,
            so a future refactor cannot silently stop checking one tensor.

Run:
  $PY_DEPLOY scripts/validate/test_genie_load_sim.py \
      [--old $LLMDEPLOY_DATA/bundles/qwen3vl_4b_e2e_pipeline]
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lint_pipeline_bundle import simulate_genie_load          # noqa: E402

CTX_SIZE, KV_DIM, ROPE_DIM = 2048, 128, 64
N_LAYERS, SPLIT_AT, EMBD, VOCAB = 36, 18, 2560, 151936
AR, PAST, TOTAL = 128, 2048, 2176           # ground truth: PAST+AR == TOTAL


def _t(name, dims):
    return {"name": name, "dimensions": list(dims)}


def _kv(lo, hi, ar, past, out_ar):
    """past-KV in/out tensor pairs for layers [lo, hi)."""
    ins, outs = [], []
    for n in range(lo, hi):
        if past:
            ins.append(_t(f"past_key_{n}_in", [1, 8, KV_DIM, past]))
            ins.append(_t(f"past_value_{n}_in", [1, 8, past, KV_DIM]))
        outs.append(_t(f"past_key_{n}_out", [1, 8, KV_DIM, out_ar]))
        outs.append(_t(f"past_value_{n}_out", [1, 8, out_ar, KV_DIM]))
    return ins, outs


def build_rebuilt():
    """The two ctx-bins the plan's ground-truth table specifies."""
    shards = []
    for si, (lo, hi) in enumerate(((0, SPLIT_AT), (SPLIT_AT, N_LAYERS))):
        gmap = {}
        for gname, ar, past in ((f"prefill_{si}", AR, PAST),
                                (f"decode_{si}", 1, TOTAL - 1)):
            kin, kout = _kv(lo, hi, ar, past, ar)
            ins = [_t("attention_mask", [1, ar, TOTAL]),
                   _t("position_ids_cos", [1, ar, ROPE_DIM]),
                   _t("position_ids_sin", [1, ar, ROPE_DIM])] + kin
            outs = list(kout)
            if si == 0:
                ins.insert(0, _t("inputs_embeds", [1, 1, ar, EMBD]))
                suffix = "_p" if gname.startswith("prefill") else ""
                ins += [_t(f"deepstack_visual_embed_{k}{suffix}", [1, 1, 256, EMBD])
                        for k in range(3)]
                outs.append(_t("last_hidden_states", [1, 1, ar, EMBD]))
            else:
                ins.insert(0, _t("last_hidden_states", [1, 1, ar, EMBD]))
                outs.append(_t("logits", [1, ar, VOCAB]))
            gmap[gname] = {"graphName": gname, "graphInputs": ins,
                           "graphOutputs": outs}
        shards.append(gmap)
    return shards


def load_old(bundle):
    shards = []
    for i in (1, 2):
        p = bundle / f"qwen3vl-4b-w8a16_{i}_of_2.info.json"
        if not p.is_file():
            return None
        doc = json.loads(p.read_text())
        info = doc.get("info", doc)
        gmap = {}
        for g in info["graphs"]:
            gi = g.get("info", g)
            gmap[gi["graphName"]] = gi
        shards.append(gmap)
    return shards


def run(name, shards, expect_pass, must_contain=None, want_count=None):
    errors, facts = simulate_genie_load(shards, CTX_SIZE, KV_DIM, ROPE_DIM)
    ok = True
    if expect_pass and errors:
        print(f"  FAIL {name}: expected clean, got {len(errors)} error(s)")
        for e in errors[:6]:
            print(f"        {e}")
        ok = False
    elif not expect_pass and not errors:
        print(f"  FAIL {name}: expected failure, got none")
        ok = False
    if must_contain:
        hits = [e for e in errors if must_contain in e[2]]
        if want_count is not None and len(hits) != want_count:
            print(f"  FAIL {name}: expected {want_count} error(s) containing "
                  f"{must_contain!r}, got {len(hits)}")
            for e in errors[:8]:
                print(f"        {e}")
            ok = False
        elif not hits:
            print(f"  FAIL {name}: no error contained {must_contain!r}")
            ok = False
    if ok:
        detail = f" ({len(errors)} error(s))" if errors else ""
        print(f"  OK   {name}{detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=None,
                    help="bundle dir holding the 2026-08-14 info.jsons")
    args = ap.parse_args()
    good = build_rebuilt()
    passed = []

    print("\n[control] rebuilt past-KV tower must load")
    passed.append(run("ground-truth shapes", good, expect_pass=True))

    print("\n[mutation] the 2026-08-14 device failure must reproduce")
    old = None
    if args.old:
        old = load_old(Path(args.old))
        if old is None:
            print(f"  FAIL --old given but info.jsons not found in {args.old}")
            passed.append(False)
    if old is None and args.old is None:
        # Reconstruct the shipped bertcache prefill: mask [1,AR,AR], no past-KV
        # inputs. Same mutation, no dependency on a local bundle.
        old = copy.deepcopy(good)
        for si in (0, 1):
            g = old[si][f"prefill_{si}"]
            g["graphInputs"] = [t for t in g["graphInputs"]
                                if not t["name"].startswith("past_")]
            for t in g["graphInputs"]:
                if t["name"] == "attention_mask":
                    t["dimensions"] = [1, AR, AR]
        print("       (synthesised: no --old bundle given)")
    if old is not None:
        passed.append(run("bertcache prefill in a split tower", old,
                          expect_pass=False,
                          must_contain=f"Expected [ 1, {AR}, {TOTAL}] "
                                       f"Found [ 1, {AR}, {AR}]",
                          want_count=2))

    print("\n[probes] each rule must fire in isolation")
    cases = [
        ("mask trailing dim", 0, "prefill_0", "graphInputs", "attention_mask",
         [1, AR, TOTAL - 8]),
        ("past_key_in past_dim", 0, "prefill_0", "graphInputs", "past_key_0_in",
         [1, 8, KV_DIM, PAST - 32]),
        ("past_value_in transpose", 0, "prefill_0", "graphInputs",
         "past_value_0_in", [1, 8, KV_DIM, PAST]),
        ("past_key_out AR", 0, "prefill_0", "graphOutputs", "past_key_0_out",
         [1, 8, KV_DIM, AR // 2]),
        ("rope dim", 1, "prefill_1", "graphInputs", "position_ids_cos",
         [1, AR, ROPE_DIM * 2]),
    ]
    for label, si, gname, key, tname, dims in cases:
        m = copy.deepcopy(good)
        for t in m[si][gname][key]:
            if t["name"] == tname:
                t["dimensions"] = dims
        passed.append(run(f"broken {label}", m, expect_pass=False))

    # No logits anywhere in the last split -> m_logit_variants empty.
    m = copy.deepcopy(good)
    for gname in ("prefill_1", "decode_1"):
        m[1][gname]["graphOutputs"] = [
            t for t in m[1][gname]["graphOutputs"] if t["name"] != "logits"]
    passed.append(run("no logits anywhere in last split", m, expect_pass=False,
                      must_contain="m_logit_variants would be empty"))

    # Stripping logits from decode_1 ALONE is deliberately NOT a load error:
    # the graph reclassifies DEFAULT -> DECODER_PREFILL, which Check 2 exempts
    # (nsp-model.cpp:787), and prefill_1 still registers a logit variant. Assert
    # the reclassification so this stays a documented decision rather than a
    # hole someone rediscovers on device.
    m = copy.deepcopy(good)
    m[1]["decode_1"]["graphOutputs"] = [
        t for t in m[1]["decode_1"]["graphOutputs"] if t["name"] != "logits"]
    errs, facts = simulate_genie_load(m, CTX_SIZE, KV_DIM, ROPE_DIM)
    types = {n: t for n, _, _, _, t in facts.get("graphs", [])}
    if errs or types.get("decode_1") != "DECODER_PREFILL":
        print(f"  FAIL logits-less decode_1 reclassifies: {len(errs)} error(s), "
              f"decode_1 typed {types.get('decode_1')!r} (want DECODER_PREFILL)")
        passed.append(False)
    else:
        print("  OK   logits-less decode_1 reclassifies to DECODER_PREFILL "
              "(exempt by :787, not a load error)")
        passed.append(True)

    n_bad = passed.count(False)
    print(f"\n{len(passed) - n_bad}/{len(passed)} checks passed")
    if n_bad:
        print("FAIL: the load simulation is not trustworthy")
        sys.exit(1)
    print("PASS: load simulation reproduces the incident and clears the rebuild")


if __name__ == "__main__":
    main()
