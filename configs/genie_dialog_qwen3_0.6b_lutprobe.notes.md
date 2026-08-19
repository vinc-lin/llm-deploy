# Notes — `genie_dialog_qwen3_0.6b_lutprobe.json`

Moved out of the config's `_comment` key on 2026-08-19. libGenie 1.19
validates the config against a strict whitelist and rejects ANY unknown
top-level key:

    Unknown dialog config key: _comment / Failed to create the dialog config

The device team hit exactly this on the v5 session and had to strip the key
by hand before the probes would load. JSON has no comment syntax; the notes
belong beside the file, not inside it.

---

0.6B LUT PROBE -- a diagnostic, not a product. The ctx-bin is built on the
SHIPPING model's ladekv recipe, so its graphs are shape-identical to the
device-proven 44.707 tok/s build: prefill mask [1,128,640] past [1,8,128,512],
decode mask [1,1,640] past [1,8,128,639]. The ONLY difference is that the
first input is inputs_embeds instead of input_ids -- i.e. the external-LUT
feed. verify32 is omitted: it exists for LADE, which is parked as a 30%
regression and unused in basic mode, so it would cost an export while
changing nothing this probe measures.

embedding.size is the HIDDEN dim (Dialog.cpp:190 maps it to context.n-embd,
and LUT.cpp strides by n_embd) -- 1024 for Qwen3-0.6B. context.size is the
CONTEXT LENGTH, 512 here. For this model both numbers are small and easily
confused; for Qwen3-VL-4B they are 2560 and 2048, plainly different. Do not
'fix' one to match the other.
