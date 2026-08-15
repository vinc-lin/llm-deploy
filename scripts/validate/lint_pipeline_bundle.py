#!/usr/bin/env python
"""Gate 2 (Stage 3): static contract lint of the assembled e2e pipeline bundle.

Nothing in this bundle can be executed off-device -- HTP context binaries do not
run on x86, and this SDK ships no x86 W8A16 path either. So every cross-artifact
agreement that the runtime assumes but never verifies has to be proven here, on
the finalised bytes, or it is proven on silicon by a SIGSEGV.

Each check maps to a specific silent-failure mode with a known precedent:

  1. reference closure -- every file named by the pipeline script and by the
     three node configs is actually in the bundle. A config naming a file that
     is not there is a load failure on a device with no debugger attached.
  2. graph binding -- every graph inside every ctx-bin is listed in the
     `graph_names` of the htp extensions file that ITS OWN node config points
     at. An unbound graph does not error: it silently compiles with backend
     defaults (O=0, 4 MB VTCM, 24 MB spill). That shipped once in this repo and
     ended in a device SIGSEGV (BUILD_GUIDE 5.4b).
  3. schema clash -- `positional-encoding` together with a backend-level
     `pos-id-dim`/`rope-theta` is a hard GENIE_STATUS_ERROR_JSON_SCHEMA at load
     (`Engine.cpp:159-161`, `677-680`). The Stage 2 bundle shipped exactly that.
     Also `sum(mrope-section) == rope-dim` (`nsp-model.cpp:3856-3863`); omitting
     the section defaults to Qwen2-VL's {16,24,24}, which also sums to 64 and so
     passes the runtime guard while producing a wrong ownership map.
  4. vision-param -- `engine.model.vision-param.{height,width}` in PATCH units
     (pre-merge), with (h/2)*(w/2) equal to the ViT's own output row count.
     `ImageEncoder.cpp:46-47,136-138` -> `nsp-model.cpp:3813-3838`. Nothing
     cross-checks this at runtime; a mismatch silently corrupts every position
     id after the image, and a missing vision-param drops MRoPE to plain rope.
  5. ViT IO dtype -- `pixel_values` and all four outputs must be UFIXED_16.
     `setupInputFP16` is an empty stub that discards the pixels and returns
     success (`nsp-image-model.cpp:526-530`), so an fp16 ViT is not a load
     error, it is a caption that ignores the image.
  6. LUT -- exact byte count and a float32/2560 declaration. A fixed-point LUT
     silently no-ops against this graph's input.
  7. sample image -- exact byte count, grid, dtype, and an encoding that matches
     the shipped ViT ctx-bin's own `pixel_values` quantizeParams. Genie does no
     preprocessing (`node set image` is an opaque blob straight into
     GenieNode_setData), so host and device agree here or nowhere.
  8. chat template -- the two segment files, plus 256 `<|image_pad|>` rows,
     must re-tokenize EXACTLY to the processor's own chat template for the same
     prompt and the same shipped image. The prompt string is recovered FROM the
     shipped segment file, so this check cannot drift away from the bundle.
     Also: no trailing newline (genie-app reads segment files with rdbuf(),
     `main.cpp:1151-1157`, so a trailing newline lands in the prompt) and no
     literal backslash-n (genie-app never unescapes, `main.cpp:655-658,1140-1142`).
  9. Genie load simulation -- replays libGenie's own `validateModel`
     (`nsp-model.cpp:620-964`) against the shipped info.jsons: graph
     classification (`nsp-graph.cpp:194-262`), AR/CL derivation (`:72-187`),
     the cache-group scatter/concat + ctx detection (`nsp-model.cpp:524-575`),
     the `(AR,CL)`-keyed variant map with its **`DECODER_PREFILL` CL rewrite**
     (`:580-614`), then every `checkShape` (`:844-917`). This is the check the
     2026-08-14 bundle needed and did not have: it passed 1-8 and then died at
     load with `ShapeError : attention_mask - Expected [ 1, 128, 2176] Found
     [ 1, 128, 128]`, twice, before emitting a token. Failures are printed in
     the device's own vocabulary so they diff against logcat directly.
 10. decode-only fallback -- the emergency config must be ONE file swap. The
     whole document is compared against the primary with the two select-graphs
     keys removed; anything else differing (context size, sampler, ctx-bins)
     means "try the fallback" silently changes something else and a failure
     there proves nothing. Also: no selected name may be absent from the
     shipped graphs, none may be a prefill graph, and the subset it does load
     must itself pass the load simulation.
 11. test kit closure -- every `wx_*.script` must be runnable as shipped: all
     referenced files present, each script feeding its OWN blob, each `.raw`
     exactly 1024*1536*2 bytes, each sidecar's encoding equal to the shipped
     ViT's `pixel_values` quantizeParams, and no orphan blobs.

Checks 9-11 are mutation-tested by `test_genie_load_sim.py` and by the
procedure in the plan doc: dropping a kit blob, reverting a prefill mask to
the v1 shape, drifting the fallback's context size, and pointing the fallback
at a prefill graph must each fire. The context-size mutation is why check 10
compares whole documents -- an earlier version diffed only the QnnHtp block
and let that drift through.

Every failure is reported; the exit code is the violation count.

Run:
  $PY_DEPLOY scripts/validate/lint_pipeline_bundle.py \
      --bundle $LLMDEPLOY_DATA/bundles/qwen3vl_4b_e2e_pipeline \
      --model  $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct
"""
import argparse
import copy
import json
import re
import sys
from pathlib import Path

LUT_ROWS, LUT_DIM = 151936, 2560
LUT_BYTES = LUT_ROWS * LUT_DIM * 4
N_PATCH, N_FEAT = 1024, 1536
RAW_BYTES = N_PATCH * N_FEAT * 2
IMG_ROWS = 256                      # merged ViT output rows == <|image_pad|> count
EDGE = 512
SPATIAL_MERGE = 2
VIT_DTYPE = "QNN_DATATYPE_UFIXED_POINT_16"
VIT_IN = "pixel_values"
VIT_OUT = ("image_features", "deepstack_visual_embed_0",
           "deepstack_visual_embed_1", "deepstack_visual_embed_2")
CLASH_KEYS = ("pos-id-dim", "rope-theta")
IMAGE_PAD = "<|image_pad|>"

# genie-app's script language. Comments are stripped first: the shipped script
# quotes `node set text` and `node set textFile` inside its own comments, and a
# regex that does not strip them "finds" a segment file called "reads".
CFG_RE = re.compile(r"^\s*node\s+config\s+create\s+(\S+)\s+(\S+)\s*$", re.M)
SET_RE = re.compile(
    r"^\s*node\s+set\s+(textFile|text|image)\s+(\S+)\s+(\S+)\s+(.+?)\s*$", re.M)


class Report:
    """Collect every violation instead of dying on the first one -- a bundle
    with three problems should take one run to diagnose, not three."""

    def __init__(self):
        self.problems = []

    def head(self, n, title):
        print(f"\n[{n}] {title}")

    def ok(self, msg):
        print(f"  OK   {msg}")

    def info(self, msg):
        print(f"       {msg}")

    def fail(self, msg):
        print(f"  FAIL {msg}")
        self.problems.append(msg)


def strip_comments(text):
    """genie-app treats '#' as a comment introducer; no shipped command quotes one."""
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def unwrap(node):
    """qnn-context-binary-utility wraps records as {"version":..,"info":{..}}.
    Tolerate an already-unwrapped record so the lint fails on the contract
    rather than on a utility version bump."""
    if isinstance(node, dict) and isinstance(node.get("info"), dict):
        return node["info"]
    return node


def info_sidecar(bundle, ctxbin):
    """The info.json shipped alongside a ctx-bin. The bundle is flat, so three
    files cannot all be called info.json; each carries its binary's stem."""
    stem = ctxbin[:-4] if ctxbin.endswith(".bin") else ctxbin
    return bundle / f"{stem}.info.json"


def load_graphs(path, rep):
    """ctx-bin info.json -> {graphName: graph info dict}, or None on failure."""
    try:
        doc = unwrap(json.loads(Path(path).read_text()))
    except Exception as exc:                              # noqa: BLE001
        rep.fail(f"{path.name}: unreadable ctx-bin info.json ({exc})")
        return None
    graphs = {}
    for g in doc.get("graphs") or []:
        gi = unwrap(g)
        name = gi.get("graphName")
        if name is None:
            rep.fail(f"{path.name}: a graph record has no graphName")
            continue
        graphs[name] = gi
    if not graphs:
        rep.fail(f"{path.name}: no graphs found")
    return graphs


def tensor_map(graph, key):
    out = {}
    for t in graph.get(key) or []:
        ti = unwrap(t)
        if ti.get("name") is not None:
            out[ti["name"]] = ti
    return out


def scale_offset(tensor):
    """quantizeParams -> (scale, offset). 2.48.40 names the member "scaleOffset";
    other releases have emitted "scaleOffsetEncoding". Never default: a silently
    wrong pixel scale is indistinguishable from a bad image until it hits silicon."""
    qp = tensor.get("quantizeParams") or {}
    q = qp.get("scaleOffset") or qp.get("scaleOffsetEncoding")
    if not q:
        return None
    return float(q["scale"]), int(q["offset"])


# --------------------------------------------------------------------------- #
# node configs
# --------------------------------------------------------------------------- #
class Node:
    """One genie-app node config: its single top-level kind plus the sub-blocks
    the runtime actually reads."""

    def __init__(self, alias, file, doc):
        self.alias, self.file = alias, file
        (self.kind,) = doc.keys()
        self.body = doc[self.kind]
        self.engine = self.body.get("engine") or {}
        self.model = self.engine.get("model") or {}
        self.backend = self.engine.get("backend") or {}
        self.ctxbins = list((self.model.get("binary") or {}).get("ctx-bins") or [])
        self.extensions = self.backend.get("extensions")
        self.tokenizer = (self.body.get("tokenizer") or {}).get("path")
        # the text-generator names it "embedding", the text-encoder "lut"
        self.lut = self.body.get("embedding") or self.body.get("lut")

    def refs(self):
        r = list(self.ctxbins)
        for x in (self.extensions, self.tokenizer):
            if x:
                r.append(x)
        if self.lut and self.lut.get("lut-path"):
            r.append(self.lut["lut-path"])
        return r

    def backend_dicts(self):
        """The backend block and its type-specific sub-block (e.g. QnnHtp).
        `pos-id-dim` lived in the latter in the Stage 2 bundle."""
        out = [self.backend]
        sub = self.backend.get(self.backend.get("type"))
        if isinstance(sub, dict):
            out.append(sub)
        return out


def parse_nodes(bundle, cfg_files, rep):
    nodes = []
    for alias, name in cfg_files:
        p = bundle / name
        if not p.is_file():
            continue                                   # check 1 reports the miss
        try:
            doc = json.loads(p.read_text())
        except Exception as exc:                       # noqa: BLE001
            rep.fail(f"{name}: unparseable node config ({exc})")
            continue
        if len(doc) != 1:
            rep.fail(f"{name}: expected exactly 1 top-level key, got {sorted(doc)}")
            continue
        nodes.append(Node(alias, name, doc))
    return nodes


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def check_refs(bundle, refs, rep):
    rep.head(1, "reference closure (script + node configs -> bundle files)")
    for src, r in sorted(refs):
        if (bundle / r).is_file():
            rep.ok(f"{r:42s} <- {src}")
        else:
            rep.fail(f"referenced file missing from bundle: {r} (from {src})")


def check_graph_binding(bundle, nodes, rep):
    rep.head(2, "graph binding (every ctx-bin graph listed in its node's "
                "htp extensions)")
    graphs_by_bin, all_graphs = {}, set()
    for node in nodes:
        if not node.ctxbins:
            continue
        listed = set()
        if node.extensions and (bundle / node.extensions).is_file():
            try:
                ext = json.loads((bundle / node.extensions).read_text())
                listed = {n for g in ext.get("graphs", [])
                          for n in g.get("graph_names", [])}
            except Exception as exc:                    # noqa: BLE001
                rep.fail(f"{node.extensions}: unparseable htp extensions ({exc})")
        have = set()
        for cb in node.ctxbins:
            side = info_sidecar(bundle, cb)
            if not side.is_file():
                rep.fail(f"{cb}: no shipped info.json ({side.name}) -- graph "
                         "binding cannot be verified against the actual binary")
                continue
            g = load_graphs(side, rep)
            if g is None:
                continue
            graphs_by_bin[cb] = g          # name -> graph info, for checks 4/5
            have |= set(g)
        all_graphs |= have
        rep.info(f"{node.kind}: graphs {sorted(have)} / {node.extensions} "
                 f"binds {sorted(listed)}")
        unbound = have - listed
        if unbound:
            rep.fail(f"{node.kind}: unbound graph(s) {sorted(unbound)} -- would "
                     "silently compile with HTP defaults (O=0, 4 MB VTCM)")
        else:
            rep.ok(f"{node.kind}: all {len(have)} graph(s) bound")
    # A graph_names entry matching nothing is a stale config: the tuning block
    # it carries is applied to no graph at all.
    listed_all = set()
    for ext in sorted(bundle.glob("htp_backend_ext_config*.json")):
        try:
            e = json.loads(ext.read_text())
        except Exception:                               # noqa: BLE001
            continue                                    # already reported above
        listed_all |= {n for g in e.get("graphs", []) for n in g.get("graph_names", [])}
    stale = listed_all - all_graphs
    if stale:
        rep.fail(f"graph_names entries matching no shipped graph: {sorted(stale)}")
    else:
        rep.ok(f"no stale graph_names entries ({len(listed_all)} name(s) all resolve)")
    return graphs_by_bin


def check_schema(nodes, rep):
    rep.head(3, "schema clash (positional-encoding vs backend pos-id-dim/rope-theta)")
    for node in nodes:
        pe = node.model.get("positional-encoding")
        clash = sorted({k for d in node.backend_dicts() for k in CLASH_KEYS if k in d})
        if pe and clash:
            rep.fail(f"{node.file}: declares positional-encoding AND backend "
                     f"{clash} -- hard GENIE_STATUS_ERROR_JSON_SCHEMA at load "
                     "(Engine.cpp:159-161,677-680)")
        elif clash:
            rep.ok(f"{node.file}: backend {clash}, no positional-encoding block")
        elif pe:
            rep.ok(f"{node.file}: positional-encoding only, no backend {list(CLASH_KEYS)}")
        else:
            rep.ok(f"{node.file}: neither positional-encoding nor backend "
                   f"{list(CLASH_KEYS)}")
        if not pe:
            continue
        dim = pe.get("rope-dim")
        sec = (pe.get("rope-scaling") or {}).get("mrope-section")
        if dim is None:
            rep.fail(f"{node.file}: positional-encoding has no rope-dim")
        elif sec is None:
            rep.fail(f"{node.file}: no mrope-section -- defaults to Qwen2-VL's "
                     "{16,24,24}, which also sums to 64 and passes the runtime "
                     "guard while owning the wrong axes")
        elif sum(sec) != dim:
            rep.fail(f"{node.file}: sum(mrope-section)={sum(sec)} != rope-dim={dim} "
                     "(nsp-model.cpp:3856-3863)")
        else:
            rep.ok(f"{node.file}: sum(mrope-section)={sum(sec)} == rope-dim={dim}")


def check_vision_param(bundle, nodes, graphs_by_bin, rep):
    rep.head(4, "vision-param (patch units) vs the ViT's own output row count")
    enc = [n for n in nodes if n.kind == "image-encoder"]
    if not enc:
        rep.fail("no image-encoder node config in the pipeline")
        return None
    node = enc[0]
    vp = node.model.get("vision-param")
    if not isinstance(vp, dict):
        rep.fail(f"{node.file}: no engine.model.vision-param -- setVisionParam is "
                 "never called and image rows fall back to plain rope "
                 "(ImageEncoder.cpp:46-47,136-138)")
        return node
    missing = [k for k in ("height", "width") if k not in vp]
    if missing:
        rep.fail(f"{node.file}: vision-param missing {missing}")
        return node
    h, w = vp["height"], vp["width"]
    if h % SPATIAL_MERGE or w % SPATIAL_MERGE:
        rep.fail(f"{node.file}: vision-param {h}x{w} not divisible by "
                 f"spatial-merge-size {SPATIAL_MERGE}")
        return node
    rows = (h // SPATIAL_MERGE) * (w // SPATIAL_MERGE)
    # Cross-check against the shipped graph, not against a constant: this is the
    # number the encoder actually appends (ImageEncoder.cpp:144-146).
    vit_rows = None
    for cb in node.ctxbins:
        g = graphs_by_bin.get(cb)
        if not g:
            continue
        for gi in g.values():
            t = tensor_map(gi, "graphOutputs").get("image_features")
            if t and isinstance(t.get("dimensions"), list) and t["dimensions"]:
                vit_rows = t["dimensions"][0]
    if rows != IMG_ROWS:
        rep.fail(f"{node.file}: vision-param {h}x{w} patches -> {rows} merged rows "
                 f"!= {IMG_ROWS}; every position id after the image would be "
                 "silently wrong (nsp-model.cpp:3813-3838)")
    elif vit_rows is not None and rows != vit_rows:
        rep.fail(f"{node.file}: vision-param -> {rows} rows but the ViT graph "
                 f"emits image_features[{vit_rows}]")
    else:
        rep.ok(f"{node.file}: vision-param {h}x{w} patches -> ({h}/2)*({w}/2) = "
               f"{rows} merged rows == image_features rows "
               f"({vit_rows if vit_rows is not None else 'n/a'})")
    if h * w != N_PATCH:
        rep.fail(f"{node.file}: vision-param {h}*{w} = {h * w} patches != "
                 f"{N_PATCH} (nsp-image-model.cpp:373 validates this against "
                 "the pixel_values shape)")
    return node


def check_vit_io(bundle, enc_node, graphs_by_bin, rep):
    rep.head(5, f"ViT IO dtype (all {VIT_DTYPE})")
    if enc_node is None:
        rep.fail("no image-encoder node config; ViT IO not checked")
        return None
    pixel = None
    for cb in enc_node.ctxbins:
        g = graphs_by_bin.get(cb)
        if not g:
            rep.fail(f"{cb}: no graph info; ViT IO not checked")
            continue
        for name, gi in sorted(g.items()):
            ins = tensor_map(gi, "graphInputs")
            outs = tensor_map(gi, "graphOutputs")
            if VIT_IN not in ins:
                rep.fail(f"{cb}:{name}: no input named {VIT_IN!r} (Genie's image "
                         "model binds by exact name)")
            for tname in (VIT_IN,) + VIT_OUT:
                t = ins.get(tname) or outs.get(tname)
                if t is None:
                    rep.fail(f"{cb}:{name}: missing tensor {tname!r}")
                    continue
                dt = t.get("dataType")
                if dt != VIT_DTYPE:
                    rep.fail(f"{cb}:{name}: {tname} is {dt}, not {VIT_DTYPE} -- a "
                             "stock pipeline cannot drive this graph "
                             "(nsp-image-model.cpp:526-530)")
                else:
                    rep.ok(f"{cb}:{name}: {tname:26s} {t.get('dimensions')} {dt}")
            if VIT_IN in ins:
                pixel = scale_offset(ins[VIT_IN])
                if pixel is None:
                    rep.fail(f"{cb}:{name}: {VIT_IN} carries no scale/offset")
    return pixel


def check_lut(bundle, nodes, rep):
    rep.head(6, "embedding LUT (bytes and declared dtype/size)")
    seen = set()
    for node in nodes:
        if not node.lut:
            continue
        path = node.lut.get("lut-path")
        dt, size = node.lut.get("datatype"), node.lut.get("size")
        if dt != "float32":
            rep.fail(f"{node.file}: LUT datatype {dt!r} != 'float32' -- a "
                     "fixed-point LUT silently no-ops against this graph's input")
        elif size != LUT_DIM:
            rep.fail(f"{node.file}: LUT size {size} != {LUT_DIM}")
        else:
            rep.ok(f"{node.file}: declares {path} float32/{size}")
        if path in seen or not path:
            continue
        seen.add(path)
        p = bundle / path
        if not p.is_file():
            continue                                    # check 1 reports the miss
        got = p.stat().st_size
        if got != LUT_BYTES:
            rep.fail(f"{path}: {got} bytes != {LUT_ROWS}*{LUT_DIM}*4 = {LUT_BYTES}")
        else:
            rep.ok(f"{path}: {got} bytes == {LUT_ROWS}*{LUT_DIM}*4")


def check_sample_image(bundle, raw_name, pixel_enc, rep):
    rep.head(7, "sample image (bytes, grid, dtype, host<->device encoding)")
    if raw_name is None:
        rep.fail("the pipeline script sets no image input")
        return None
    raw = bundle / raw_name
    if not raw.is_file():
        return None                                     # check 1 reports the miss
    got = raw.stat().st_size
    if got != RAW_BYTES:
        rep.fail(f"{raw_name}: {got} bytes != {N_PATCH}*{N_FEAT}*2 = {RAW_BYTES} "
                 "(node set image is an opaque blob; the size is never checked "
                 "against the tensor)")
    else:
        rep.ok(f"{raw_name}: {got} bytes == {N_PATCH}*{N_FEAT}*2")
    meta_path = raw.with_suffix(".json")
    if not meta_path.is_file():
        rep.fail(f"{meta_path.name}: missing sidecar for {raw_name}")
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("grid_thw") != [[1, 32, 32]]:
        rep.fail(f"{meta_path.name}: grid_thw {meta.get('grid_thw')} != [[1,32,32]]")
    else:
        rep.ok(f"{meta_path.name}: grid_thw [[1,32,32]]")
    if meta.get("dtype") != "uint16":
        rep.fail(f"{meta_path.name}: dtype {meta.get('dtype')!r} != 'uint16'")
    else:
        rep.ok(f"{meta_path.name}: dtype uint16")
    enc = meta.get("encoding") or {}
    if pixel_enc is None:
        rep.fail("ViT pixel_values encoding unavailable; host<->device agreement "
                 "not checked")
    elif "scale" not in enc or "offset" not in enc:
        rep.fail(f"{meta_path.name}: no encoding scale/offset")
    elif (float(enc["scale"]), int(enc["offset"])) != pixel_enc:
        rep.fail(f"{meta_path.name}: encoding scale/offset "
                 f"({enc['scale']!r}, {enc['offset']!r}) != the shipped ViT "
                 f"ctx-bin's pixel_values {pixel_enc} -- the device would "
                 "dequantize the image with different parameters than the host "
                 "quantized it with")
    else:
        rep.ok(f"{meta_path.name}: encoding scale={enc['scale']:.9e} "
               f"offset={enc['offset']} == ViT pixel_values quantizeParams")
    return meta


def check_chat_template(bundle, model, seg_files, meta, rep):
    rep.head(8, "chat template equivalence (segments + 256 image_pad == processor)")
    if len(seg_files) != 2:
        rep.fail(f"expected exactly 2 `node set textFile` segments in the script, "
                 f"got {len(seg_files)}")
        return
    segs = []
    for name in seg_files:
        p = bundle / name
        if not p.is_file():
            return                                      # check 1 reports the miss
        blob = p.read_bytes()
        try:
            s = blob.decode("utf-8")
        except UnicodeDecodeError as exc:
            rep.fail(f"{name}: not valid UTF-8 ({exc})")
            return
        if "\\n" in s:
            rep.fail(f"{name}: contains a literal backslash-n -- genie-app never "
                     "unescapes (main.cpp:655-658,1140-1142); write a real newline")
        rep.info(f"{name}: {len(blob)} bytes {s!r}")
        segs.append(s)
    seg1, seg2 = segs

    # The prompt is recovered FROM the shipped bundle, so this check can never
    # drift away from what is actually going to run.
    if not seg2.startswith("<|vision_end|>"):
        rep.fail(f"{seg_files[1]}: does not start with <|vision_end|>")
        return
    tail = seg2[len("<|vision_end|>"):]
    if "<|im_end|>" not in tail:
        rep.fail(f"{seg_files[1]}: no <|im_end|> after the prompt")
        return
    prompt = tail.split("<|im_end|>", 1)[0]
    rep.info(f"prompt recovered from {seg_files[1]}: {prompt!r}")

    from PIL import Image
    from transformers import AutoProcessor
    edge = int((meta or {}).get("edge") or EDGE)
    proc = AutoProcessor.from_pretrained(model, min_pixels=edge * edge,
                                         max_pixels=edge * edge)
    tok = proc.tokenizer
    pad_id = tok.convert_tokens_to_ids(IMAGE_PAD)
    if pad_id is None or pad_id == tok.unk_token_id:
        rep.fail(f"tokenizer has no {IMAGE_PAD} token")
        return
    png = bundle / "sample_image.png"
    if not png.is_file():
        rep.fail("sample_image.png missing -- the equivalence must be proven "
                 "against the image this bundle actually ships")
        return
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}]}]
    want_text = proc.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)

    # Byte-exactness against the template's own halves. The rule is "nothing
    # beyond the segment", NOT "no newline at the end": seg2's last character is
    # legitimately a newline, because the template ends "<|im_start|>assistant\n"
    # and the model is primed on that token. What must never happen is ONE MORE
    # -- rdbuf() would feed it straight into the prompt.
    want_parts = want_text.split(IMAGE_PAD)
    if len(want_parts) != 2:
        rep.fail(f"the chat template contains {len(want_parts) - 1} {IMAGE_PAD} "
                 "markers, expected exactly 1")
        return
    for name, got_s, want_s in zip(seg_files, segs, want_parts):
        if got_s == want_s:
            rep.ok(f"{name}: byte-exact against the template segment "
                   f"({len(got_s.encode('utf-8'))} bytes)")
        elif got_s.rstrip("\n") == want_s.rstrip("\n"):
            rep.fail(f"{name}: trailing-newline drift -- file ends "
                     f"{got_s[-24:]!r}, template segment ends {want_s[-24:]!r}. "
                     "genie-app reads segment files with rdbuf() "
                     "(main.cpp:1151-1157), so the difference lands in the prompt")
        else:
            rep.fail(f"{name}: {got_s!r} != the template segment {want_s!r}")

    with Image.open(png) as im:
        expect = proc(text=[want_text], images=[im.convert("RGB")],
                      return_tensors="np")["input_ids"][0].tolist()
    got = (tok(seg1, add_special_tokens=False)["input_ids"]
           + [pad_id] * IMG_ROWS
           + tok(seg2, add_special_tokens=False)["input_ids"])
    n_pad = expect.count(pad_id)
    if n_pad != IMG_ROWS:
        rep.fail(f"the processor expands sample_image.png to {n_pad} {IMAGE_PAD} "
                 f"rows, not {IMG_ROWS} -- the ViT graph emits {IMG_ROWS}")
    if got == expect:
        rep.ok(f"segments + {IMG_ROWS}*{IMAGE_PAD} == processor chat template, "
               f"token for token ({len(expect)} tokens, {n_pad} image_pad)")
        return
    rep.fail(f"segments + {IMG_ROWS}*{IMAGE_PAD} != the processor's chat template "
             f"({len(got)} tokens vs {len(expect)})")
    rep.info(f"script : {(seg1 + IMAGE_PAD * IMG_ROWS + seg2)[:90]!r}...")
    rep.info(f"HF     : {want_text[:90]!r}...")
    for i, (a, e) in enumerate(zip(got, expect)):
        if a != e:
            rep.info(f"first divergence at token {i}: "
                     f"{a} {tok.decode([a])!r} != {e} {tok.decode([e])!r}")
            break


# --------------------------------------------------------------------------- #
# check 9 -- Genie load simulation (replays validateModel off-device)
# --------------------------------------------------------------------------- #
KV_PREFIX = "past_"


def _is_kv(name):
    """QnnUtils::isKVTensor (qnn-utils.hpp:234-238)."""
    return (name.endswith("_in") or name.endswith("_out")) and \
           ("key" in name or "value" in name)


def _dims4(dims):
    """qualla's Dims axis mapping (qnn-utils.cpp:113-122 feeding :66-76).

    Dims are RIGHT-ALIGNED into a 4-vector padded with 1s, then
    (batch,height,width,channel) = (v0,v1,v2,v3). So [1,128,2176] is
    h=1,w=128,c=2176 and [1,8,128,2048] is h=8,w=128,c=2048 -- which is why
    the device prints a 3-tuple for a 4-D tensor.

    Rank > 4 takes a different path (Dims still reads at(1..3), i.e. NOT the
    trailing dims). No shipped graph has one; refuse rather than guess.
    """
    v = [int(x) for x in dims]
    if len(v) > 4:
        return None
    v = [1] * (4 - len(v)) + v
    h, w, c = v[1], v[2], v[3]
    # "Hack to mix batch dimension" (qnn-utils.cpp:68-75), batchSize == 1
    if v[0] != 1 and v[1] == 1:
        h = v[0]
    batch = v[0] if (v[0] > 1 and v[1] != 1) else 1
    return batch, h, w, c


class _T:
    """One tensor as validateModel sees it."""

    def __init__(self, name, dims):
        self.name, self.raw = name, [int(x) for x in dims]
        d = _dims4(dims)
        self.ok = d is not None
        self.batch, self.h, self.w, self.c = d if d else (1, 1, 1, 1)

    numel = property(lambda s: s.h * s.w * s.c)
    maxdim = property(lambda s: max(s.h, s.w, s.c))
    hwc = property(lambda s: [s.h, s.w, s.c])


class _G:
    """One graph variant, plus the (AR, CL) and GraphType qualla derives."""

    def __init__(self, name, gi, shard):
        self.name, self.shard = name, shard
        self.ins = {n: _T(n, t["dimensions"])
                    for n, t in tensor_map(gi, "graphInputs").items()
                    if t.get("dimensions")}
        self.outs = {n: _T(n, t["dimensions"])
                     for n, t in tensor_map(gi, "graphOutputs").items()
                     if t.get("dimensions")}
        self.type = self._classify()
        self.ar = self._ar()
        self.cl = self._cl()

    def _classify(self):
        """GraphVariant::determineGraphType (nsp-graph.cpp:194-262).

        NOTE m_layerNames[INPUT] is still its default "input_ids" here
        (nsp-model.hpp:47) -- validateModel only rewrites it to
        "inputs_embeds" later (nsp-model.cpp:669). So an embeddings-input
        decoder has inputIDExists == false and lands on the DECODER_PREFILL
        branch. That is exactly how shard 0 got classified on device.
        """
        input_id = "input_ids" in self.ins
        kv_in = any(n.startswith(KV_PREFIX) and _is_kv(n) for n in self.ins)
        matched, past_kv, img = set(), False, False
        for n in self.outs:
            if n.startswith(KV_PREFIX) and _is_kv(n):
                past_kv = True
                matched.add(n)
            if n.startswith(("image_features", "vision_embedding",
                             "cross_attention_states")):
                img = True
                matched.add(n)
                break                      # nsp-graph.cpp:223
        logits = "logits" in self.outs
        if logits:
            matched.add("logits")
        if "last_hidden_states" in self.outs:
            matched.add("last_hidden_states")   # :231-233 -- the split handoff
        matched_all = matched == set(self.outs)

        if input_id and not past_kv and not logits:
            return "LUT"
        if not input_id and past_kv and not logits:
            return "DECODER_PREFILL" if matched_all else "DECODER"
        if kv_in and not past_kv:
            return "KV_SHARE_NO_KV_OUTPUT"
        if not input_id and not past_kv and logits:
            return "LMHEAD"
        if img:
            return "IMAGE_ENCODER"
        return "DEFAULT"

    def _ar(self):
        """determineGraphInputSize (nsp-graph.cpp:72-142), in its own order."""
        t = self.ins.get("input_ids")
        if t:
            return t.numel // t.batch
        for n in ("inputs_embeds",
                  "_model_embed_tokens_Gather_Gather_output_0",
                  "_embed_tokens_Gather_Gather_output_0"):
            t = self.ins.get(n)
            if t:
                return t.numel // t.maxdim
        t = self.ins.get("attention_mask")
        if t:
            return t.numel // t.maxdim
        for n, t in self.outs.items():
            if n.startswith(KV_PREFIX) and "key" in n:
                return t.c
        t = self.outs.get("logits")
        if t:
            return t.numel // t.c
        return None

    def _cl(self):
        """determineGraphContextSize (nsp-graph.cpp:146-187)."""
        t = self.ins.get("attention_mask")
        if t:
            return t.c
        for n, t in self.outs.items():
            if n.startswith(KV_PREFIX) and "key" in n:
                tin = self.ins.get(n[:n.rfind("_")] + "_in")
                if tin:
                    return t.c + tin.c
        return None


def _shape_err(tensor, exp):
    """checkShape's own message (nsp-model.cpp:486-493). -1 prints as '*'."""
    e = ", ".join("*" if d == -1 else str(d) for d in exp)
    f = ", ".join(str(d) for d in tensor.hwc)
    return f"Expected [ {e}] Found [ {f}]"


def _cmp(graph, tensor, exp, errors):
    if tensor is None:
        return
    if all(e == -1 or e == g for e, g in zip(exp, tensor.hwc)):
        return
    errors.append((graph.name, tensor.name, _shape_err(tensor, exp)))


def simulate_genie_load(shards, ctx_size, kv_dim, rope_dim):
    """Replay Genie's load-time validation. `shards` is a list of
    {graphName: graph-info} in ctx-bin load order (== m_nsp_graphs order).

    Returns (errors, facts). Each error is (graph, tensor, message) using the
    device's own vocabulary, so a failure here is directly comparable to an
    adb logcat line.
    """
    graphs, facts = [], {}
    for i, gmap in enumerate(shards):
        for name, gi in sorted(gmap.items()):
            graphs.append(_G(name, gi, i))
    errors = []
    for g in graphs:
        if g.ar is None or g.cl is None:
            errors.append((g.name, "-", "could not determine AR/CL from graph IO"))
    if errors:
        return errors, facts
    facts["graphs"] = [(g.name, g.shard, g.ar, g.cl, g.type) for g in graphs]

    # -- 1. cache-group scatter/concat + ctx size (nsp-model.cpp:524-575) ----
    # nsp_graph_count is a std::map keyed by (n_tokens, ctx_size): ascending.
    specs = sorted({(g.ar, g.cl) for g in graphs})
    shard_type = {}
    for g in graphs:
        shard_type.setdefault(g.shard, g.type)
    scatter, cache_ctx, detected = False, 0, False
    for ar, cl in specs:
        kv_ctx, mask = 0, None
        for si in range(len(shards)):
            if shard_type.get(si) == "KV_SHARE_NO_KV_OUTPUT":
                continue
            v = next((g for g in graphs
                      if g.shard == si and g.ar == ar and g.cl == cl), None)
            if v is None:
                continue
            if kv_ctx == 0:
                for n, t in v.ins.items():
                    if n.startswith(KV_PREFIX) and "key" in n:
                        kv_ctx = t.c
                        break
            if mask is None:
                mask = v.ins.get("attention_mask")
            if kv_ctx != 0 and mask is not None:
                group_ctx = mask.maxdim
                if not detected:
                    if kv_ctx == group_ctx:
                        scatter = True
                    elif kv_ctx == group_ctx - ar:
                        scatter = False
                    else:
                        errors.append((
                            v.name, "past_key_*_in",
                            "Could not determine whether KV$ uses Scatter or "
                            f"Concat. KV$ has input dimension {kv_ctx}. "
                            f"Expected CL={cl} or CL - AR-n={cl - ar}"))
                        return errors, facts
                    cache_ctx = group_ctx
                else:
                    cache_ctx = max(cache_ctx, group_ctx)
                detected = True
                break
    facts["use_scatter"], facts["cache_group_ctx_size"] = scatter, cache_ctx

    # -- 2. variant map, with the DECODER_PREFILL CL rewrite (:580-614) ------
    # Populated from the FIRST shard owning a past_*key* output, then keyed by
    # (AR, CL) only -- so every shard sharing that key inherits shard 0's
    # rewritten CL. That is why the device logged two IDENTICAL ShapeErrors.
    vmap = {}
    for si in range(len(shards)):
        owner = [g for g in graphs if g.shard == si]
        if not owner:
            continue
        rep_out = next((n for n in sorted(owner[0].outs)
                        if n.startswith(KV_PREFIX) and "key" in n), None)
        if rep_out is None:
            continue
        keyin = rep_out[:rep_out.rfind("_")] + "_in"
        for g in owner:
            kout = g.outs.get(rep_out)
            if kout is None:
                continue
            arn = kout.c
            kin = g.ins.get(keyin)
            ctx = arn if kin is None else kin.c + (0 if scatter else arn)
            if g.type == "DECODER_PREFILL":
                ctx = cache_ctx                       # nsp-model.cpp:604-605
            vmap[(g.ar, g.cl)] = (arn, ctx)
        break
    facts["variant_map"] = dict(vmap)

    # -- 3. Check 3: shapes for all named tensors (:844-917) -----------------
    for g in graphs:
        if (g.ar, g.cl) not in vmap:
            errors.append((g.name, "-", f"no cache-group mapping for "
                                        f"(AR={g.ar}, CL={g.cl})"))
            continue
        arn, gctx = vmap[(g.ar, g.cl)]
        past_dim = gctx if scatter else gctx - arn

        _cmp(g, g.ins.get("attention_mask"), [1, arn, gctx], errors)   # :858
        for n in ("position_ids_cos", "position_ids_sin"):             # :871-873
            _cmp(g, g.ins.get(n), [1, g.ar, rope_dim], errors)

        if g.type != "KV_SHARE_NO_KV_OUTPUT":                          # :889-890
            for n, t in g.ins.items():
                if not n.startswith(KV_PREFIX):
                    continue
                if "key" in n:
                    _cmp(g, t, [-1, kv_dim, past_dim], errors)         # :903
                elif "value" in n:
                    _cmp(g, t, [-1, past_dim, kv_dim], errors)         # :905
            for n, t in g.outs.items():
                if not n.startswith(KV_PREFIX):
                    continue
                if "key" in n:
                    _cmp(g, t, [-1, kv_dim, arn], errors)              # :911
                elif "value" in n:
                    _cmp(g, t, [-1, arn, kv_dim], errors)              # :913

    # -- 4. Check 1 (first split) and Check 2 (last split) -------------------
    last = len(shards) - 1
    for g in graphs:
        if g.shard == 0:
            t = g.ins.get("inputs_embeds")
            if t is None:
                errors.append((g.name, "inputs_embeds", "Tensor not found"))
            else:
                embd = t.maxdim
                if t.numel != g.ar * embd:                             # :709-716
                    errors.append((g.name, "inputs_embeds", "Wrong input shape"))
        if g.shard == last and g.type != "DECODER_PREFILL":            # :787
            t = g.outs.get("logits")
            if t is None:
                errors.append((g.name, "logits", "Tensor not found"))
            else:
                vocab = t.maxdim
                if t.numel not in (vocab, vocab * g.ar):               # :793-798
                    errors.append((g.name, "logits", "Wrong logits shape"))

    # -- at least one logit variant in the last split ------------------------
    # Removing `logits` does NOT trip Check 2: a logits-less graph reclassifies
    # to DECODER_PREFILL, which :787 exempts. The failure surfaces later and
    # quieter -- registerLogitVariants (:1198-1204) walks the last split only,
    # so m_logit_variants ends up empty and kvmanager.cpp:467-490 has nothing
    # to fall back to when the final step must produce a sample.
    if not any(g.shard == last and "logits" in g.outs for g in graphs):
        errors.append(("-", "logits",
                       "no variant in the last split produces logits -- "
                       "m_logit_variants would be empty and no inference step "
                       "could yield a sample"))

    # -- context.size must fit the cache group ------------------------------
    if ctx_size is not None and cache_ctx and ctx_size > cache_ctx:
        errors.append(("-", "context.size",
                       f"dialog context.size {ctx_size} exceeds the cache-group "
                       f"context {cache_ctx}"))
    return errors, facts


def check_genie_load(bundle, nodes, graphs_by_bin, rep):
    """Check 9 -- would libGenie's validateModel accept these ctx-bins?

    The 2026-08-14 device attempt died here, twice, before a single token:
      ShapeError : attention_mask - Expected [ 1, 128, 2176] Found [ 1, 128, 128]
      -> Failed to create the Genie Node (-1) -> SIGSEGV
    Nothing in the build chain models load-time validation, so a bundle that
    can never load still passes every other gate. This replays the validator
    against the finalised bytes and reports in the device's own vocabulary.
    """
    rep.head(9, "Genie load simulation (validateModel replay)")
    seen = False
    for node in nodes:
        if node.kind != "text-generator":
            continue
        seen = True
        shards = []
        missing = False
        for b in node.ctxbins:
            g = graphs_by_bin.get(b)
            if g is None:
                rep.fail(f"{node.file}: no graph info for ctx-bin {b}")
                missing = True
                break
            shards.append(g)
        if missing or not shards:
            continue

        ctx_size = ((node.body.get("context") or {}).get("size"))
        htp = node.backend.get(node.backend.get("type")) or {}
        kv_dim = htp.get("kv-dim")
        pe = (node.model.get("positional-encoding") or {})
        rope_dim = pe.get("rope-dim")
        if kv_dim is None or rope_dim is None:
            rep.fail(f"{node.file}: cannot simulate load -- "
                     f"kv-dim={kv_dim}, rope-dim={rope_dim}")
            continue

        errors, facts = simulate_genie_load(shards, ctx_size, kv_dim, rope_dim)
        for name, shard, ar, cl, gtype in facts.get("graphs", []):
            rep.info(f"{name}: shard {shard}, AR={ar}, CL={cl}, {gtype}")
        if "cache_group_ctx_size" in facts:
            rep.info(f"cache group '{KV_PREFIX}': ctx={facts['cache_group_ctx_size']}, "
                     f"{'scatter' if facts['use_scatter'] else 'concat'}, "
                     f"variant map {facts.get('variant_map')}")
        if errors:
            for gname, tname, msg in errors:
                rep.fail(f"{node.file}: {gname} : {tname} - {msg}")
        else:
            rep.ok(f"{node.file}: {len(facts.get('graphs', []))} graphs would "
                   f"pass validateModel")
    if not seen:
        rep.fail("no text-generator node found -- nothing to load-simulate")


# --------------------------------------------------------------------------- #
# check 10 -- decode-only fallback
# --------------------------------------------------------------------------- #
SELECT_KEYS = ("load-select-graphs", "execute-select-graphs")


def check_fallback(bundle, nodes, graphs_by_bin, rep):
    """The fallback must be one file swap, not a second bundle.

    Any drift between primary and fallback beyond the two select-graphs keys
    means the device team's "try the fallback" step silently changes something
    else -- context size, kv-dim, tuning -- and a failure then proves nothing.
    """
    rep.head(10, "decode-only fallback (one file swap, two keys)")
    prim = next((n for n in nodes if n.kind == "text-generator"), None)
    fb_files = sorted(bundle.glob("genie_text_generator_*_decodeonly.json"))
    fb_scripts = sorted(bundle.glob("genie_pipeline_*_decodeonly.script"))
    if not fb_files or not fb_scripts:
        rep.fail("no decode-only fallback shipped (config and/or script) -- the "
                 "single-attempt plan requires one")
        return
    if prim is None:
        rep.fail("no primary text-generator to compare the fallback against")
        return

    fb_path = fb_files[0]
    try:
        fb = json.loads(fb_path.read_text())
    except Exception as exc:                                  # noqa: BLE001
        rep.fail(f"{fb_path.name}: unparseable ({exc})")
        return
    prim_doc = json.loads((bundle / prim.file).read_text())
    a = prim_doc["text-generator"]["engine"]["backend"]["QnnHtp"]
    b = fb["text-generator"]["engine"]["backend"]["QnnHtp"]
    extra = sorted(set(b) - set(a))
    if extra != sorted(SELECT_KEYS):
        rep.fail(f"{fb_path.name}: QnnHtp adds {extra}, expected "
                 f"{sorted(SELECT_KEYS)}")

    # Compare the WHOLE document, not just the QnnHtp block. An earlier version
    # of this check only diffed QnnHtp, and a mutation that moved the
    # fallback's context.size from 2048 to 1024 sailed straight through -- the
    # exact class of silent behavioural drift this check exists to catch.
    def _strip(doc):
        d = copy.deepcopy(doc)
        htp = (((d.get("text-generator") or {}).get("engine") or {})
               .get("backend") or {}).get("QnnHtp")
        if isinstance(htp, dict):
            for k in SELECT_KEYS:
                htp.pop(k, None)
        return d

    def _paths(x, y, prefix=""):
        """Where two json documents disagree, as dotted paths."""
        if isinstance(x, dict) and isinstance(y, dict):
            out = []
            for k in sorted(set(x) | set(y)):
                out += _paths(x.get(k), y.get(k), f"{prefix}.{k}" if prefix else k)
            return out
        return [] if x == y else [f"{prefix} ({x!r} vs {y!r})"]

    diffs = _paths(prim_doc, _strip(fb))
    if diffs:
        rep.fail(f"{fb_path.name}: differs from the primary beyond the select "
                 f"keys, at {diffs[:6]}")
    else:
        rep.ok(f"{fb_path.name}: byte-for-byte the primary once "
               f"{sorted(SELECT_KEYS)} are removed (whole document compared)")

    selected = b.get("execute-select-graphs") or []
    all_graphs = {g for gm in graphs_by_bin.values() for g in gm}
    unknown = [g for g in selected if g not in all_graphs]
    if unknown:
        rep.fail(f"{fb_path.name}: execute-select-graphs names no shipped graph: "
                 f"{unknown}")
    if any(g.startswith("prefill") for g in selected):
        rep.fail(f"{fb_path.name}: selects a prefill graph, which defeats the "
                 "point of the fallback")

    # The fallback loads only its selected graphs, so validate that subset.
    shards = []
    for cb in prim.ctxbins:
        gm = graphs_by_bin.get(cb) or {}
        shards.append({k: v for k, v in gm.items() if k in set(selected)})
    if all(shards):
        ctx = (prim.body.get("context") or {}).get("size")
        htp = prim.backend.get(prim.backend.get("type")) or {}
        pe = prim.model.get("positional-encoding") or {}
        errs, _ = simulate_genie_load(shards, ctx, htp.get("kv-dim"),
                                     pe.get("rope-dim"))
        if errs:
            for gname, tname, msg in errs:
                rep.fail(f"{fb_path.name} (decode-only): {gname} : {tname} - {msg}")
        else:
            rep.ok(f"{fb_path.name}: the {len(selected)} selected graph(s) would "
                   "pass validateModel")
    else:
        rep.fail(f"{fb_path.name}: a ctx-bin contributes none of {selected}")

    fb_script = fb_scripts[0]
    body = strip_comments(fb_script.read_text())
    prim_script = next((p for p in bundle.glob("genie_pipeline_*.script")
                        if "decodeonly" not in p.name), None)
    if prim_script is None:
        rep.fail("no primary pipeline script to compare the fallback against")
        return
    pa = [l.strip() for l in strip_comments(prim_script.read_text()).splitlines()
          if l.strip()]
    pb = [l.strip() for l in body.splitlines() if l.strip()]
    if len(pa) != len(pb):
        rep.fail(f"{fb_script.name}: {len(pb)} statements vs the primary's "
                 f"{len(pa)} -- it must differ by ONE line")
    else:
        diffs = [(x, y) for x, y in zip(pa, pb) if x != y]
        if len(diffs) != 1 or fb_path.name not in diffs[0][1]:
            rep.fail(f"{fb_script.name}: expected exactly one differing line "
                     f"naming {fb_path.name}, got {diffs}")
        else:
            rep.ok(f"{fb_script.name}: differs from the primary by exactly the "
                   "textGenerator config line")


# --------------------------------------------------------------------------- #
# check 11 -- weather test-image kit
# --------------------------------------------------------------------------- #
def check_kit(bundle, raw_name, pixel_enc, rep):
    """Every kit script must be runnable as shipped.

    A kit image whose blob is missing, mis-sized, or quantized against a
    different encoding than the shipped ViT is not a test -- it is a device
    session wasted on a bad input.
    """
    rep.head(11, "weather test-image kit (closure, size, encoding)")
    scripts = sorted(bundle.glob("wx_*.script"))
    if not scripts:
        rep.fail("no wx_*.script in the bundle -- the test kit is missing")
        return
    before = len(rep.problems)
    seen_raw = set()
    for s in scripts:
        stem = s.stem
        body = strip_comments(s.read_text())
        refs = {v for _, _, _, v in SET_RE.findall(body)}
        refs |= {f for _, f in CFG_RE.findall(body)}
        missing = sorted(r for r in refs if not (bundle / r).is_file())
        if missing:
            rep.fail(f"{s.name}: references missing file(s) {missing}")
        own = [r for r in refs if r.endswith(".raw")]
        if own != [f"{stem}.raw"]:
            rep.fail(f"{s.name}: feeds {own}, expected ['{stem}.raw'] -- a kit "
                     "script that points at another image tests the wrong scene")
            continue
        seen_raw.add(f"{stem}.raw")
        blob = bundle / f"{stem}.raw"
        if blob.is_file() and blob.stat().st_size != RAW_BYTES:
            rep.fail(f"{blob.name}: {blob.stat().st_size} bytes != {RAW_BYTES}")
        side = bundle / f"{stem}.json"
        if not side.is_file():
            rep.fail(f"{stem}.json: sidecar missing")
            continue
        try:
            meta = json.loads(side.read_text())
        except Exception as exc:                              # noqa: BLE001
            rep.fail(f"{stem}.json: unparseable ({exc})")
            continue
        enc = meta.get("encoding") or {}
        got = (float(enc.get("scale", 0)), int(enc.get("offset", 0)))
        if pixel_enc and got != pixel_enc:
            rep.fail(f"{stem}.json: encoding {got} != the shipped ViT's "
                     f"pixel_values {pixel_enc} -- these bytes are not what the "
                     "graph expects")
        if meta.get("grid_thw") != [[1, 32, 32]]:
            rep.fail(f"{stem}.json: grid {meta.get('grid_thw')} != [[1,32,32]]")
    stray = sorted(p.name for p in bundle.glob("wx_*.raw")
                   if p.name not in seen_raw)
    if stray:
        rep.fail(f"kit blob(s) no script feeds: {stray}")
    if not (bundle / "TEST_IMAGES.md").is_file():
        rep.fail("TEST_IMAGES.md missing -- the kit has no expected outputs")
    if len(rep.problems) == before:
        rep.ok(f"{len(scripts)} kit script(s), blobs closed and encoding-matched")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="assembled bundle directory")
    ap.add_argument("--model", required=True, help="HF model dir (processor/tokenizer)")
    ap.add_argument("--script", default="genie_pipeline_qwen3vl.script",
                    help="pipeline script inside the bundle")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    rep = Report()
    print(f"bundle: {bundle}")

    script_path = bundle / args.script
    if not script_path.is_file():
        print(f"  FAIL pipeline script not in bundle: {args.script}")
        print("\n1 lint violation(s)")
        sys.exit(1)
    body = strip_comments(script_path.read_text())

    cfg_files = CFG_RE.findall(body)
    sets = SET_RE.findall(body)
    refs = {(args.script, f) for _, f in cfg_files}
    seg_files, raw_name, order = [], None, []
    for verb, node_alias, role, value in sets:
        order.append(verb)
        if verb == "text":
            rep_msg = (f"`node set text {node_alias} {role} ...` cannot carry a real "
                       "newline: genie-app's split() pushes the character after an "
                       "escape verbatim (main.cpp:655-658) and nothing unescapes "
                       "later (main.cpp:1140-1142). Use `node set textFile`.")
            rep.problems.append(rep_msg)
            print(f"  FAIL {rep_msg}")
            continue
        refs.add((args.script, value))
        if verb == "textFile":
            seg_files.append(value)
        else:
            raw_name = value

    nodes = parse_nodes(bundle, cfg_files, rep)
    for node in nodes:
        for r in node.refs():
            refs.add((node.file, r))

    check_refs(bundle, refs, rep)
    graphs_by_bin = check_graph_binding(bundle, nodes, rep)
    check_schema(nodes, rep)
    enc_node = check_vision_param(bundle, nodes, graphs_by_bin, rep)
    pixel_enc = check_vit_io(bundle, enc_node, graphs_by_bin, rep)
    check_lut(bundle, nodes, rep)
    meta = check_sample_image(bundle, raw_name, pixel_enc, rep)
    # Segment order builds the accumulator, so it is part of the contract.
    if order and order != ["textFile", "image", "textFile"]:
        rep.head(8, "chat template equivalence")
        rep.fail(f"script feeds the accumulator in order {order}, expected "
                 "['textFile', 'image', 'textFile'] (segment order IS the prompt)")
    else:
        check_chat_template(bundle, args.model, seg_files, meta, rep)
    check_genie_load(bundle, nodes, graphs_by_bin, rep)
    check_fallback(bundle, nodes, graphs_by_bin, rep)
    check_kit(bundle, raw_name, pixel_enc, rep)

    n = len(rep.problems)
    if n:
        print(f"\n{n} lint violation(s):")
        for p in rep.problems:
            print(f"  FAIL {p}")
        sys.exit(min(n, 125))                 # exit codes wrap at 256; never 0
    print("\nPASS: e2e pipeline bundle contract clean")


if __name__ == "__main__":
    main()
