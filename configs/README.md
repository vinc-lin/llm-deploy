# Config notes

(`_comment` keys removed from the JSONs — qnn-context-binary-generator
logs `Unknown Key` errors for them.)

## genie_dialog_qwen3_0.6b.json

Genie T2T dialog config for Qwen3-0.6B W8A16. Schema verified against SDK 2.48.40 examples (llama3-3b-htp-long-context.json) + summary §2.1 values. cpu-mask/eos values remain device/model-specific.

## htp_backend_config.json

DRAFT backend-extension config from summary §2.1 build-time values. VERIFY key names against SDK htp_backend_extensions doc before first build.

## htp_backend_ext_config.json

DRAFT runtime backend-extensions for Genie on-device (perf voting). Summary §2.3: perf-profile via Genie backend.extensions JSON works (4 tiers, 1.95x swing); use llm_decode_burst.

## htp_config.json

DRAFT ctx-bin build config from summary §2.1. VERIFY schema against QAIRT 2.48 SDK docs/examples (Task 2 Step 4) before use. vtcm_mb MUST stay 16 (24 is rejected on unsigned PD, err 0x138d).
