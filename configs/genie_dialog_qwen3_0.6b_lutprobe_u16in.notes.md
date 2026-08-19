# Notes — `genie_dialog_qwen3_0.6b_lutprobe_u16in.json`

Moved out of the config's `_comment` key on 2026-08-19. libGenie 1.19
validates the config against a strict whitelist and rejects ANY unknown
top-level key:

    Unknown dialog config key: _comment / Failed to create the dialog config

The device team hit exactly this on the v5 session and had to strip the key
by hand before the probes would load. JSON has no comment syntax; the notes
belong beside the file, not inside it.

---

0.6B LUT PROBE -- UFIXED_16-INPUT variant (the fix). Identical to
genie_dialog_qwen3_0.6b_lutprobe.json in every respect -- same weights, same
encodings lineage, same LUT, same graph shapes, same HTP config -- EXCEPT that
its ctx-bin declares inputs_embeds as QNN_DATATYPE_UFIXED_POINT_16 rather than
QNN_DATATYPE_FLOAT_16, obtained by grafting a 16-bit INT activation encoding
onto that tensor (scripts/quant/graft_input_encoding.py).

Why it matters: quantizeInput's FLOAT_16 case advances its destination pointer
by tensorOffset BYTES, while setupInputEmbeddings passes an ELEMENT count when
padding a partially-filled prefill chunk -- so the pad write lands halfway into
the real prompt and overwrites its back half. UFIXED_16 uses correct element
arithmetic (uint16_t* + tensorOffset) and is native to HTP.

The LUT stays float32: embedding-datatype float32 is what selects
setupInputEmbeddings' quantizeInput branch, which quantizes fp32 -> uFxp_16 on
device using this graph's own scale/offset. Do NOT change datatype here.

Pair with the fp16-in bundle and run BOTH back to back on the same board.
