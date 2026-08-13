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

Every failure is reported; the exit code is the violation count.

Run:
  $PY_DEPLOY scripts/validate/lint_pipeline_bundle.py \
      --bundle $LLMDEPLOY_DATA/bundles/qwen3vl_4b_e2e_pipeline \
      --model  $LLMDEPLOY_DATA/models/Qwen3-VL-4B-Instruct
"""
import argparse
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

    n = len(rep.problems)
    if n:
        print(f"\n{n} lint violation(s):")
        for p in rep.problems:
            print(f"  FAIL {p}")
        sys.exit(min(n, 125))                 # exit codes wrap at 256; never 0
    print("\nPASS: e2e pipeline bundle contract clean")


if __name__ == "__main__":
    main()
