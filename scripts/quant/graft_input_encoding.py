#!/usr/bin/env python
"""Graft a 16-bit activation encoding onto an embeddings-fed graph input.

WHY THIS EXISTS
---------------
Genie fills `inputs_embeds` from the float32 LUT and then calls
QnnNspModel::quantizeInput, whose `tensorOffset` argument is an ELEMENT offset
in three of its four cases and a BYTE offset in the fourth:

    UFIXED_POINT_8 :  reinterpret_cast<uint8_t*>(buf)  + tensorOffset   elements
    UFIXED_POINT_16:  reinterpret_cast<uint16_t*>(buf) + tensorOffset   elements
    FLOAT_32       :  reinterpret_cast<float*>(buf)    + tensorOffset   elements
    FLOAT_16       :  reinterpret_cast<uint8_t*>(buf)  + tensorOffset   BYTES  <--

setupInputEmbeddings passes an ELEMENT count (`i * m_embd_size`) when it pads a
partially-filled prefill chunk with the pad/EOS embedding. On a FLOAT_16
`inputs_embeds` that padding write therefore lands at byte n_process*n_embd
instead of n_process*n_embd*2 -- halfway INTO the real prompt -- and overwrites
its back half. It only fires when variant > n_process, i.e. the last, partial
prefill chunk.

AIMET never writes an encoding for this tensor: aimet-torch quantizes module
OUTPUTS, and `inputs_embeds` is a graph INPUT feeding a Squeeze. With no
encoding the converter types it FLOAT_16 (the --float_bitwidth default) and the
build lands on the broken branch. Giving it a 16-bit INT activation encoding
makes the converter type it uFxp_16, which is the correct element-offset branch
AND is native to HTP.

DO NOT reach for --preserve_io_datatype instead. It pins graph I/O to the SOURCE
model dtype (float32), and the converter then inserts a QNN_Convert the HTP
backend cannot create:

    graph_prepare.cc:212::ERROR:could not create op: q::QNN_Convert
    RouterX86 graph prepare failed 12

Measured 2026-08-19 on this exact build; both ctx-bins failed to finalize. The
same warning is already in vit_build_quant.sh's header -- the ViT gets its
UFIXED_16 I/O this way, and that path is device-proven.

RANGE
-----
The encoding has to cover every value that will ever arrive in the tensor. For a
text-only tower that is exactly the LUT (--lut). For a VL tower the same tensor
also carries spliced image embeddings, so union in the vision tower's own output
encoding as well: --cover-ctxbin-info <vit>/info.json reads it straight out of the
built ViT bin. That is not merely safe, it is the target -- when the two
encodings match exactly, Genie's image splice hits the requantScale==1 &&
requantOffset==0 fast path and copies instead of rescaling. Or state the range
outright with --min/--max.

Idempotent: re-running leaves an existing entry for the tensor alone unless
--force is given.

  graft_input_encoding.py --encodings model_filtered_renamed.encodings \
      --tensor inputs_embeds --lut $LLMDEPLOY_DATA/work/lut/qwen3-0.6b
"""
import argparse
import json
import sys
from pathlib import Path

BW = 16
QMAX = (1 << BW) - 1


def lut_range(lut_dir: Path):
    """Min/max over the whole LUT, read as the runtime reads it (flat <f4)."""
    import numpy as np
    p = json.loads((lut_dir / "embedding_lut_params.json").read_text())
    assert p["datatype"] == "float32", f"expected a float32 LUT, got {p['datatype']!r}"
    a = np.memmap(lut_dir / p["lut-path"], dtype="<f4", mode="r")
    return float(a.min()), float(a.max())


def ctxbin_range(info_path: Path, name: str):
    """Range of a tensor in a built ctx-bin, from its own scale/offset.

    For a VL tower this is how you cover the image features: point it at the
    vision ctx-bin's `image_features` output. Matching that encoding exactly is
    not just safe, it is desirable -- Genie's splice then hits the
    requantScale==1 && requantOffset==0 fast path and copies instead of
    rescaling.

    The key is `quantizeParams.scaleOffset`, NOT `scaleOffsetEncoding`; these
    names are not guessable and the wrong one silently reads None.
    """
    doc = json.loads(info_path.read_text())
    found = {}

    def walk(o):
        if isinstance(o, dict):
            i = o.get("info", o)
            if i.get("name") == name:
                so = (i.get("quantizeParams") or {}).get("scaleOffset") or {}
                if so.get("scale") is not None:
                    found["s"], found["o"] = so["scale"], so["offset"]
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(doc)
    if "s" not in found:
        raise SystemExit(f"no quantized tensor named {name!r} in {info_path}")
    s, o = float(found["s"]), int(found["o"])
    return o * s, (QMAX + o) * s


def encoding_range(enc_path: Path, name: str):
    """Reconstruct [min,max] from an existing asymmetric entry in an encodings file."""
    entries = json.loads(enc_path.read_text())
    if isinstance(entries, dict):
        entries = entries.get("activation_encodings", [])
    for e in entries:
        if e.get("name") == name:
            s, o = float(e["scale"][0]), int(e["offset"][0])
            return o * s, (QMAX + o) * s
    raise SystemExit(f"no encoding named {name!r} in {enc_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encodings", required=True, type=Path)
    ap.add_argument("--tensor", default="inputs_embeds")
    ap.add_argument("--lut", type=Path, help="LUT dir to take the range from")
    ap.add_argument("--cover-json", type=Path, help="also cover this encodings file's...")
    ap.add_argument("--cover-name", help="...entry for this tensor (e.g. the ViT output)")
    ap.add_argument("--cover-ctxbin-info", type=Path,
                    help="also cover a tensor in a built ctx-bin's info.json")
    ap.add_argument("--cover-ctxbin-tensor", default="image_features",
                    help="which tensor in --cover-ctxbin-info (default: image_features)")
    ap.add_argument("--min", type=float, dest="vmin")
    ap.add_argument("--max", type=float, dest="vmax")
    ap.add_argument("--out", type=Path, help="default: in place")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # AIMET writes {activation_encodings: [...], param_encodings: [...], ...};
    # some intermediate files are the bare activation list. Accept both and write
    # back in whichever shape came in -- rewriting a wrapped file as a bare list
    # would silently drop param_encodings and convert the whole model as float.
    doc = json.loads(args.encodings.read_text())
    wrapped = isinstance(doc, dict)
    if wrapped:
        if "activation_encodings" not in doc:
            raise SystemExit(f"{args.encodings} has no activation_encodings key")
        entries = doc["activation_encodings"]
    else:
        entries = doc
    if not isinstance(entries, list):
        raise SystemExit(f"{args.encodings}: activation_encodings is not a list")

    existing = next((e for e in entries if e.get("name") == args.tensor), None)
    if existing and not args.force:
        print(f"{args.tensor}: encoding already present, leaving it alone "
              f"(scale={existing['scale'][0]:.6g} offset={existing['offset'][0]}); "
              "--force to overwrite")
        return 0

    lo = hi = None
    if args.lut:
        lo, hi = lut_range(args.lut)
        print(f"LUT range          : [{lo:.6f}, {hi:.6f}]")
    if args.cover_json:
        if not args.cover_name:
            raise SystemExit("--cover-json needs --cover-name")
        clo, chi = encoding_range(args.cover_json, args.cover_name)
        print(f"covering {args.cover_name}: [{clo:.6f}, {chi:.6f}]")
        lo = clo if lo is None else min(lo, clo)
        hi = chi if hi is None else max(hi, chi)
    if args.cover_ctxbin_info:
        blo, bhi = ctxbin_range(args.cover_ctxbin_info, args.cover_ctxbin_tensor)
        print(f"covering {args.cover_ctxbin_tensor}: [{blo:.6f}, {bhi:.6f}]")
        lo = blo if lo is None else min(lo, blo)
        hi = bhi if hi is None else max(hi, bhi)
    if args.vmin is not None:
        lo = args.vmin if lo is None else min(lo, args.vmin)
    if args.vmax is not None:
        hi = args.vmax if hi is None else max(hi, args.vmax)
    if lo is None or hi is None:
        raise SystemExit("need --lut, --cover-json/--cover-name, or --min/--max")
    if not (lo < 0 < hi):
        raise SystemExit(
            f"range [{lo}, {hi}] does not straddle zero; an asymmetric uint16 grid "
            "that cannot represent 0 exactly would bias every padded/masked row")

    scale = (hi - lo) / QMAX
    offset = int(round(lo / scale))
    entry = {"bw": BW, "dtype": "INT", "enc_type": "PER_TENSOR", "is_sym": False,
             "name": args.tensor, "offset": [offset], "scale": [scale]}
    entries = [e for e in entries if e.get("name") != args.tensor] + [entry]
    if wrapped:
        doc["activation_encodings"] = entries
    else:
        doc = entries

    out = args.out or args.encodings
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"union range        : [{lo:.6f}, {hi:.6f}]")
    print(f"grafted {args.tensor}: scale={scale:.12g} offset={offset} "
          f"-> represents [{offset*scale:.6f}, {(QMAX+offset)*scale:.6f}]")
    print(f"wrote {out} ({len(entries)} encodings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
