# Expected outputs (greedy, HF fp32 reference)

Diff the first ~30 tokens of each arm's `stdout_r1.txt` against these.
An INT8 device run will not match forever -- early divergence is the
signal, not eventual divergence. Flag an arm if it diverges in the
first few tokens, or if it degenerates into repetition.

| prompt | prompt tokens |
|---|---:|
| `simple` | 36 |
| `structured` | 71 |
| `technical` | 56 |

The `technical` prompt is 56 tokens, matching the 2026-08-13 report's
protocol exactly, so its rates are directly comparable to the
11.72 / 9.18 tok/s baselines.
