"""What does the model ACTUALLY generate? Ground truth for "is this output wrong?"

MEASURED 2026-08-21, templated "What is 2+2? Answer with one number.":

    prefill 20 tokens -> first generated id 19        # '4'
      step 1: EOS (151645) -- generation stops here   # '<|im_end|>'
    generated ids: [19, 151645]                       # "4<|im_end|>"

TWO TOKENS. That is the whole correct output. The device produced `4` then a
repetition loop to the context limit, so the continuation is a defect -- and
this script is what settled it, against a prior note in
docs/ISSUE_qwen3vl_4b_text_numerics.md claiming "4 then repeats is expected:
greedy with no EOS". It is not: eos-token is configured [151645, 151643] and
the graphs emit it immediately. That note is withdrawn.

It also proves the graphs and the KV recurrence are correct end to end -- prefill
both shards, then greedy decode with the cache re-laid across the prefill/decode
width change and written back per step, exactly as parity_e2e_vl.Decoder does.
So a device that degenerates is failing in the ctx-bin decode path or in Genie's
feed, not in the model.

Run it before calling any device output "garbage": knowing the correct answer is
two tokens changes what the symptom means.

The device gives `4` then a repetition loop. The question that decides where to
look next is whether the graphs themselves do that. This runs the real split
fp32 ONNX -- prefill both shards, then N greedy decode steps with the KV
recurrence exactly as parity_e2e_vl.Decoder does it -- and prints the text.

  host coherent  -> the graphs and the KV contract are fine; the defect is in
                    the ctx-bin decode path or Genie's recurrence
  host degenerate-> the defect is in the model/quantization, and every probe so
                    far has been looking in the wrong place
"""
import gc, os, sys
import numpy as np, onnxruntime as ort, torch
from pathlib import Path
sys.path.insert(0, str(Path(os.environ["LLMDEPLOY_ROOT"]) / "scripts/export"))
from modeling_export import MASK_VALUE, rope_tables

D = Path(os.environ["LLMDEPLOY_DATA"])
SPLIT = D / "work/onnx/qwen3vl-4b-gqa-aimet-splitkv"
LUT = D / "work/lut/qwen3vl-4b"
THETA, NSTEP = 5_000_000.0, int(sys.argv[1]) if len(sys.argv) > 1 else 12
IDS = [151644,872,198,3838,374,220,17,10,17,30,21806,448,825,1372,13,151645,198,151644,77091,198]

import json
p = json.loads((LUT / "embedding_lut_params.json").read_text())
def lut_row(t):
    with (LUT / p["lut-path"]).open("rb") as fh:
        fh.seek(t * p["size"] * 4); return np.frombuffer(fh.read(p["size"]*4), "<f4").astype(np.float32)

def sess(name):
    so = ort.SessionOptions(); so.intra_op_num_threads = 40
    return ort.InferenceSession(str(SPLIT/name/f"{name}.onnx"), so, providers=["CPUExecutionProvider"])

def meta(s):
    sh = {i.name: i.shape for i in s.get_inputs()}
    pk = next(n for n in sh if n.startswith("past_key_") and n.endswith("_in"))
    _, nkv, hd, past = sh[pk]
    idx = sorted(int(n.split("_")[2]) for n in sh if n.startswith("past_key_") and n.endswith("_in"))
    ar = sh["attention_mask"][-2] if len(sh["attention_mask"]) == 3 else 1
    deep = sorted(n for n in sh if n.startswith("deepstack_"))
    return dict(sh=sh, nkv=nkv, hd=hd, past=past, ar=ar, idx=idx, deep=deep,
                total=sh["attention_mask"][-1], H=sh.get("inputs_embeds",[0,0,0,2560])[-1])

n = len(IDS)
rows = np.stack([lut_row(t) for t in IDS])
cache = {}

# ---- prefill both shards -------------------------------------------------
s0, s1 = sess("prefill_0"), sess("prefill_1")
m0, m1 = meta(s0), meta(s1)
H, AR, PAST, TOTAL = m0["H"], m0["ar"], m0["past"], m0["total"]
mask = np.full((1, AR, TOTAL), MASK_VALUE, np.float32)
for i in range(n): mask[0, i, PAST:PAST+i+1] = 0.0
c, sn = rope_tables(torch.arange(n), m0["hd"], THETA)
cos = np.zeros((1, AR, m0["hd"]//2), np.float32); sin = np.zeros_like(cos)
cos[0,:n], sin[0,:n] = c.numpy()[0], sn.numpy()[0]
emb = np.zeros((1,1,AR,H), np.float32); emb[0,0,:n] = rows
for tag, s, m in (("s0", s0, m0), ("s1", s1, m1)):
    cache[tag] = {}
    for i in m["idx"]:
        cache[tag][f"past_key_{i}_in"] = np.zeros((1,m["nkv"],m["hd"],m["past"]), np.float32)
        cache[tag][f"past_value_{i}_in"] = np.zeros((1,m["nkv"],m["past"],m["hd"]), np.float32)
kv0 = [f"past_{x}_{i}_out" for i in m0["idx"] for x in ("key","value")]
kv1 = [f"past_{x}_{i}_out" for i in m1["idx"] for x in ("key","value")]
f0 = {"inputs_embeds":emb, "attention_mask":mask, "position_ids_cos":cos,
      "position_ids_sin":sin, **{d: np.zeros((1,1,AR,H),np.float32) for d in m0["deep"]}, **cache["s0"]}
r0 = dict(zip(["last_hidden_states"]+kv0, s0.run(["last_hidden_states"]+kv0, f0)))
hid = np.asarray(r0["last_hidden_states"], np.float32)
f1 = {"last_hidden_states":hid.reshape(1,AR,H), "attention_mask":mask,
      "position_ids_cos":cos, "position_ids_sin":sin, **cache["s1"]}
r1 = dict(zip(["logits"]+kv1, s1.run(["logits"]+kv1, f1)))
lg = np.asarray(r1["logits"], np.float32).reshape(AR,-1)
for tag, m, r in (("s0",m0,r0), ("s1",m1,r1)):
    for i in m["idx"]:
        cache[tag][f"past_key_{i}_in"][:,:,:,:n] = r[f"past_key_{i}_out"][:,:,:,:n]
        cache[tag][f"past_value_{i}_in"][:,:n,:] if False else None
        cache[tag][f"past_value_{i}_in"][:,:,:n,:] = r[f"past_value_{i}_out"][:,:,:n,:]
nxt = int(np.argmax(lg[n-1]))
print(f"prefill {n} tokens -> first generated id {nxt}", flush=True)
del s0, s1, r0, r1; gc.collect()

# ---- decode loop ---------------------------------------------------------
d0, d1 = sess("decode_0"), sess("decode_1")
n0, n1 = meta(d0), meta(d1)
DPAST, DTOTAL = n0["past"], n0["total"]
# re-lay the prefill cache into the decode graph's width
dc = {}
for tag, m in (("s0",n0), ("s1",n1)):
    dc[tag] = {}
    for i in m["idx"]:
        k = np.zeros((1,m["nkv"],m["hd"],DPAST), np.float32)
        v = np.zeros((1,m["nkv"],DPAST,m["hd"]), np.float32)
        k[:,:,:,:n] = cache[tag][f"past_key_{i}_in"][:,:,:,:n]
        v[:,:,:n,:] = cache[tag][f"past_value_{i}_in"][:,:,:n,:]
        dc[tag][f"past_key_{i}_in"], dc[tag][f"past_value_{i}_in"] = k, v
dkv0 = [f"past_{x}_{i}_out" for i in n0["idx"] for x in ("key","value")]
dkv1 = [f"past_{x}_{i}_out" for i in n1["idx"] for x in ("key","value")]
out_ids, clen = [nxt], n
for step in range(NSTEP):
    e = lut_row(out_ids[-1]).reshape(1,1,1,H)
    mk = np.full((1,1,DTOTAL), MASK_VALUE, np.float32)
    mk[0,0,:clen] = 0.0; mk[0,0,DPAST] = 0.0
    cc, ss = rope_tables(torch.arange(clen, clen+1), n0["hd"], THETA)
    g0 = {"inputs_embeds":e, "attention_mask":mk, "position_ids_cos":cc.numpy().astype(np.float32),
          "position_ids_sin":ss.numpy().astype(np.float32),
          **{d: np.zeros((1,1,1,H),np.float32) for d in n0["deep"]}, **dc["s0"]}
    q0 = dict(zip(["last_hidden_states"]+dkv0, d0.run(["last_hidden_states"]+dkv0, g0)))
    h = np.asarray(q0["last_hidden_states"], np.float32)
    g1 = {"last_hidden_states":h.reshape(1,1,H), "attention_mask":mk,
          "position_ids_cos":cc.numpy().astype(np.float32),
          "position_ids_sin":ss.numpy().astype(np.float32), **dc["s1"]}
    q1 = dict(zip(["logits"]+dkv1, d1.run(["logits"]+dkv1, g1)))
    for tag, m, q in (("s0",n0,q0), ("s1",n1,q1)):
        for i in m["idx"]:
            dc[tag][f"past_key_{i}_in"][:,:,:,clen:clen+1] = q[f"past_key_{i}_out"]
            dc[tag][f"past_value_{i}_in"][:,:,clen:clen+1,:] = q[f"past_value_{i}_out"]
    clen += 1
    nid = int(np.argmax(np.asarray(q1["logits"], np.float32).reshape(-1)))
    out_ids.append(nid)
    if nid in (151645, 151643):
        print(f"  step {step+1}: EOS ({nid}) -- generation stops here", flush=True); break
print("\ngenerated ids:", out_ids)
