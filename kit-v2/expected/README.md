# Expected outputs (greedy, HF fp32 reference)

Diff the first ~30 tokens of each arm's `stdout_r1.txt` against these.
An INT8 device run will not match forever — **early divergence is the
signal, not eventual divergence.** Flag an arm if it diverges in the
first few tokens, or if it degenerates into repetition.

Speed on a broken graph is meaningless, and W8A16 bugs historically
produce *fluent but wrong* output rather than obvious garbage. The
2026-08-15 `gqafix_hybrid` arm looped on `"and parallel, and parallel…"`
forever, so this is not hypothetical.

| prompt | prompt tokens |
|---|---:|
| `simple` | 36 |
| `structured` | 71 |
| `technical` | 56 |

The `technical` prompt is 56 tokens, matching the 2026-08-13 protocol
exactly, so its rates are comparable to the **44.707 tok/s** post-fix
baseline and the **6.836** pre-fix control.

> ⚠️ These references were generated for the **W16-head, CL=1152** graph.
> Two arms in this kit legitimately differ:
>
> - **`p4_qh_ladekv`** has an INT8 `lm_head`. Quality is unchanged at 0.6B
>   greedy (`REFERENCE.md` §6.4), but token-exactness against these
>   references is **not** guaranteed. Judge it on coherence, not on a
>   token diff, and do not report a wording difference as a regression.
> - **`p4_cl512_ladekv`** has `context.size` 512. Identical for these
>   prompts and short generations; it can only diverge past 512 tokens.
>
> Every other arm (`ctrl`, `hvx8`, `udma`, `socmodel72`, `dlbc`, `wpack`)
> uses **byte-identical DLCs** to the baseline — only the ctx-bin build
> config differs — so those must match token-for-token. A divergence there
> is a real finding, not a quantization artifact.

⚠️ **Older kits said these rates are "directly comparable to the
11.72 / 9.18 tok/s baselines". Both figures are withdrawn.** 11.72 was
never an AR-1 decode rate — it is a two-phase blend measured on a
bertcache bundle. See `../runsheet.md` §0.2.
