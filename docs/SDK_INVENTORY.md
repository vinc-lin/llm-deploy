# QAIRT 2.48.40.260702 Community SDK — Local Inventory (2026-08-10)

Installed at `$LLMDEPLOY_DATA/sdk/qairt/2.48.40.260702` (11,896 zip entries, 2.39 GB zip).
Same version as the remote host's SDK. All tools verified running on WSL2 after
local extraction of `libc++1-14`, `libc++abi1-14`, `libunwind-14` debs into
`$LLMDEPLOY_DATA/syslibs/` (no root needed) — see `scripts/env.sh`.

## Verified working (x86_64-linux-clang)

| Tool | Status |
|---|---|
| `qairt-converter` | ✓ runs under `qairt-py312` (needs SDK `lib/python` on PYTHONPATH + libpython3.12 on LD_LIBRARY_PATH) |
| `qnn-context-binary-generator` | ✓ prints `QNN SDK v2.48.40.260702151143` |
| `qnn-net-run` | ✓ |
| `genie-t2t-run` (x86!) | ✓ prints usage — local e2e Genie smoke tests possible |
| `qnn-context-binary-utility` | present |
| `qairt-quantizer` | present (untested; possible AIMET alternative for A/B) |

## Key libraries

- x86: `libQnnHtp.so`, `libQnnCpu.so`, `libQnnSystem.so`, `libQnnModelDlc.so`, `libGenie.so`, `libHtpPrepare.so`
- aarch64-android: all 7 device libs from summary §1.3 (`libQnnHtpV81Skel.so` comes from `lib/hexagon-v81/unsigned/`, which ships BOTH `libQnnHtpV81Skel.so` and `libQairtHtpV81Skel.so`)
- aarch64-android `genie-t2t-run` for the device bundle: present

## Notable

- **Genie/qualla engine SOURCE CODE** ships in `examples/Genie/Genie/src/` — used to
  extract the exact graph I/O contract (see `docs/NOTES-genie-io.md`).
- Example configs: `examples/Genie/configs/` (llama3, qwen3-eagle, phi3, …) validated
  our dialog JSON schema.
- **No x86 HTP simulator** in the Community drop (only LPAI sims) → HTP-kernel
  numerics cannot be simulated locally; CPU backend + AIMET quantsim are the proxies.
- Converter divergence vs remote notes: `--target_soc_model` EXISTS here (summary
  §2.2 said it doesn't on the remote converter). We don't use it either way.
- Docs tree: `docs/QAIRT-Docs/`.
