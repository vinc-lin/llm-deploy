#!/usr/bin/env python
"""Refuse LUT/graph-input dtype pairs that Genie silently does nothing with.

THE FAILURE THIS EXISTS TO PREVENT
----------------------------------
Genie converts the embedding LUT into the graph input's encoding at runtime.
The raw copy is only a fast path taken when the two already agree
(qualla/dialog.cpp:631, :677, :844, :1109 -- all guarded identically):

    requantScale  = lutScale / inputScale                       (dialog.cpp:432)
    requantOffset = requantScale * lutOffset - inputOffset
    if (lutDataType == inputDataType && requantScale == 1 && requantOffset == 0)
        decoderInput = std::move(encoderOutput);     // raw copy
    else
        requantEmbedding(...);                       // convert

Scale mismatches are therefore fine -- Genie rescales.  DTYPE CLASS mismatches
are not.  Dialog::requantEmbedding (dialog.cpp:485-551) enumerates fixed->fixed
pairs only, and outside the ufixed4 case its if/else-if chain has NO trailing
else: an unsupported pair writes nothing at all, leaving the embedding buffer
untouched.  No exception, no log line, no failed load -- the tower just runs on
garbage.  That is the same class of silent-garbage bug as the 2026-08-11
last-token-only logits head, and it must not be possible to ship twice.

float32 LUTs bypass the dispatch entirely: dialog.cpp:684 passes the fp32 bytes
through, and nsp-model.cpp:3120 quantizeInput() converts per token into
FLOAT_16 / FLOAT_32 / UFIXED_8 / UFIXED_16.

Run (either form):
  lint_embedding_dtype.py --lut-params <lut>/embedding_lut_params.json \
                          --ctxbin-info <ctxbin>/info.json
  lint_embedding_dtype.py --lut-datatype ufixed16 --input-dtype QNN_DATATYPE_FLOAT_16
"""
import argparse
import json
import sys
from pathlib import Path

# Genie config spelling -> QNN datatype (Dialog.cpp:190-215 translateEmbeddingConfig)
LUT_NAME_TO_QNN = {
    "float32": "QNN_DATATYPE_FLOAT_32",
    "ufixed4": "QNN_DATATYPE_UFIXED_POINT_4",
    "ufixed8": "QNN_DATATYPE_UFIXED_POINT_8",
    "ufixed16": "QNN_DATATYPE_UFIXED_POINT_16",
    "sfixed8": "QNN_DATATYPE_SFIXED_POINT_8",
    "sfixed16": "QNN_DATATYPE_SFIXED_POINT_16",
}

# Exactly the branches present in Dialog::requantEmbedding (dialog.cpp:485-551).
# Transcribed from source, NOT from the datatype enum -- the enum lists many
# more types than the function actually implements.
REQUANT_TABLE = {
    "QNN_DATATYPE_UFIXED_POINT_4": {"QNN_DATATYPE_UFIXED_POINT_8",
                                    "QNN_DATATYPE_UFIXED_POINT_16"},
    "QNN_DATATYPE_UFIXED_POINT_8": {"QNN_DATATYPE_UFIXED_POINT_8",
                                    "QNN_DATATYPE_UFIXED_POINT_16"},
    "QNN_DATATYPE_UFIXED_POINT_16": {"QNN_DATATYPE_UFIXED_POINT_8",
                                     "QNN_DATATYPE_UFIXED_POINT_16"},
    "QNN_DATATYPE_SFIXED_POINT_8": {"QNN_DATATYPE_SFIXED_POINT_8",
                                    "QNN_DATATYPE_SFIXED_POINT_16",
                                    "QNN_DATATYPE_UFIXED_POINT_16"},
    "QNN_DATATYPE_SFIXED_POINT_16": {"QNN_DATATYPE_SFIXED_POINT_8",
                                     "QNN_DATATYPE_SFIXED_POINT_16"},
}
# quantizeInput's switch (nsp-model.cpp:3120-3155), i.e. the float32 fast path.
# FLOAT_16 is deliberately NOT here -- see FLOAT32_BROKEN_DESTINATIONS.
FLOAT32_DESTINATIONS = {"QNN_DATATYPE_UFIXED_POINT_8", "QNN_DATATYPE_UFIXED_POINT_16",
                        "QNN_DATATYPE_FLOAT_32"}
# quantizeInput HAS a FLOAT_16 case, so this pair looks supported and this lint
# used to pass it. It is not: the case advances its destination pointer by
# tensorOffset BYTES --
#     case QNN_DATATYPE_FLOAT_16:
#       float32ToFloat16(reinterpret_cast<uint8_t*>(getBuffer(t)) + tensorOffset, in, len);
# -- while the other three advance by ELEMENTS (uint8_t*/uint16_t*/float* + off),
# and setupInputEmbeddings (nsp-model.cpp:1807-1814) passes an ELEMENT count
# (i * m_embd_size) when padding a partially-filled chunk with the pad/EOS
# embedding. The pad write therefore starts at byte n_process*n_embd instead of
# n_process*n_embd*2 -- halfway into the real prompt -- and overwrites its back
# half. Fires only when variant > n_process, i.e. the LAST, partial prefill
# chunk: exactly the tokens the next prediction leans on hardest. Decode
# (variant == n_process == 1) is untouched, which is why a model can look fine
# on short single-token probes and still produce garbage from a real prompt.
FLOAT32_BROKEN_DESTINATIONS = {"QNN_DATATYPE_FLOAT_16"}
# ufixed4 is the one case that raises instead of falling through
THROWS_INSTEAD_OF_SILENCE = {"QNN_DATATYPE_UFIXED_POINT_4"}


def check_pair(lut_dtype, input_dtype):
    """(ok, detail). lut_dtype/input_dtype are QNN_DATATYPE_* strings."""
    if lut_dtype == "QNN_DATATYPE_FLOAT_32":
        if input_dtype in FLOAT32_BROKEN_DESTINATIONS:
            return False, (
                "float32 LUT -> FLOAT_16 input: quantizeInput's FLOAT_16 case advances "
                "its destination by tensorOffset BYTES while setupInputEmbeddings "
                "passes ELEMENTS, so padding a partially-filled prefill chunk "
                "overwrites the back half of the real prompt (nsp-model.cpp:3144 vs "
                ":1813). Graft a 16-bit INT activation encoding onto the tensor so the "
                "converter types it uFxp_16 -- scripts/quant/graft_input_encoding.py. "
                "Do NOT use --preserve_io_datatype: it pins the input to float32 and "
                "the converter emits a QNN_Convert that graph-prepare cannot create.")
        if input_dtype in FLOAT32_DESTINATIONS:
            return True, ("float32 LUT bypasses requantEmbedding; "
                          "quantizeInput (nsp-model.cpp:3120) handles this input dtype "
                          "with element-offset arithmetic")
        return False, (f"float32 LUT but quantizeInput has no case for {input_dtype} "
                       "(nsp-model.cpp:3120-3155)")
    supported = REQUANT_TABLE.get(lut_dtype)
    if supported is None:
        return False, (f"{lut_dtype} is not a source in Dialog::requantEmbedding "
                       "(dialog.cpp:485-551)")
    if input_dtype in supported:
        return True, "handled branch in Dialog::requantEmbedding"
    if lut_dtype in THROWS_INSTEAD_OF_SILENCE:
        return False, (f"{lut_dtype} -> {input_dtype} raises "
                       '"Unsupported requantization operation." at runtime')
    return False, (f"{lut_dtype} -> {input_dtype} has NO branch in "
                   "Dialog::requantEmbedding: the chain falls through and leaves the "
                   "embedding buffer UNTOUCHED (silent garbage, no error). "
                   f"Handled destinations for this LUT: {sorted(supported)}")


def input_dtype_from_ctxbin(info_json, tensor="inputs_embeds"):
    """Read the graph input tensor's dtype out of qnn-context-binary-utility JSON."""
    d = json.loads(Path(info_json).read_text())
    found = {}
    for g in d["info"]["graphs"]:
        gi = g["info"]
        for t in gi.get("graphInputs", []):
            ti = t.get("info", t)
            if ti.get("name") == tensor:
                found[gi.get("graphName", "?")] = ti.get("dataType")
    assert found, f"no input tensor named {tensor!r} in {info_json}"
    dtypes = set(found.values())
    assert len(dtypes) == 1, (
        f"{tensor} has different dtypes across graphs: {found} -- Genie reads it from "
        "the FIRST graph (dialog.cpp:430) and would mis-handle the others")
    return found, dtypes.pop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lut-params", help="embedding_lut_params.json")
    ap.add_argument("--lut-datatype", help="override: float32/ufixed8/ufixed16/...")
    ap.add_argument("--ctxbin-info", help="info.json from qnn-context-binary-utility")
    ap.add_argument("--input-dtype", help="override: QNN_DATATYPE_*")
    ap.add_argument("--tensor", default="inputs_embeds")
    args = ap.parse_args()

    if args.lut_datatype:
        lut_name = args.lut_datatype
    else:
        assert args.lut_params, "need --lut-params or --lut-datatype"
        lut_name = json.loads(Path(args.lut_params).read_text())["datatype"]
    assert lut_name in LUT_NAME_TO_QNN, f"unknown LUT datatype {lut_name!r}"
    lut_dtype = LUT_NAME_TO_QNN[lut_name]

    if args.input_dtype:
        per_graph, input_dtype = {"(override)": args.input_dtype}, args.input_dtype
    else:
        assert args.ctxbin_info, "need --ctxbin-info or --input-dtype"
        per_graph, input_dtype = input_dtype_from_ctxbin(args.ctxbin_info, args.tensor)

    print(f"LUT datatype      : {lut_name} -> {lut_dtype}")
    print(f"{args.tensor:<18s}: {input_dtype}")
    for g, t in per_graph.items():
        print(f"  graph {g}: {t}")

    ok, detail = check_pair(lut_dtype, input_dtype)
    print(f"\n{'PASS' if ok else 'FAIL'}: {detail}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
