# Notes — `genie_dialog_qwen3_0.6b_lade_demo.json`

Moved out of the config's `_comment` key on 2026-08-19. libGenie 1.19
validates the config against a strict whitelist and rejects ANY unknown
top-level key:

    Unknown dialog config key: _comment / Failed to create the dialog config

The device team hit exactly this on the v5 session and had to strip the key
by hand before the probes would load. JSON has no comment syntax; the notes
belong beside the file, not inside it.

---

DEMO/interactive config: lade fast path + sampling. MUST NOT contain 'max-num-tokens' — 'type: lade' + 'max-num-tokens' SIGSEGVs on device (exit 139; device report 2026-08-13 §6.1). Generation is bounded by context.size (1024) and by EOS, which sampling reaches normally; the greedy PARITY dialogs in this bundle (temp 0) are the ones that can run to 'Context Size was exceeded' (HTP doc 8.2). Prompts must still be the full Qwen3 chat template WITH an empty <think>

</think> block in the assistant prefix (bos-token is -1 because the template supplies <|im_start|> itself; do not set both).
