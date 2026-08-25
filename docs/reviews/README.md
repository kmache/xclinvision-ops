# Review log

This project was developed against a series of code reviews. Each numbered issue below
was fixed in a dedicated commit whose message records the root cause, the fix, and any
deviation from what the review asked for. Read a commit body for the reasoning:

```bash
git show <sha>                    # full rationale for one fix
git log --grep="issue #" --oneline
```

## Numbered review issues

| # | Subject | Commit(s) |
|---|---|---|
| 1 | Multilabel headline was class-index order; secondary positives dropped | `7e92b71d`, `4656c031` |
| 2 | LLM health-probe cache TTL not configurable | `80fd0c89` |
| 3 | Drift baseline hardcoded; corrupt log lines unguarded | `dc92241a` |
| 4 | Decision thresholds inconsistent; no clinical floor on critical findings | `73ba92bc` |
| 5 | Agent imported per-request, costing ~9 s cold on first analysis | `58fbd88b` |
| 6 | `LLMManager` treated empty provider results as success | `06e070d0` |
| 7 | Served probabilities labelled calibrated while uncalibrated | `98c7ccb0`, `e549ddb3` |
| 8 | Wrong `LOCAL_LLM_URL` variable name in `.env.example` | `d0c08344` |
| 9 | Checkpoint loaded with `strict=False`, hiding architecture mismatch | `44f5a45f` |
| 10 | `/api/v2/history` decoded every row's heatmaps | `6504a123` |

## Unnumbered follow-up work

| Subject | Commit |
|---|---|
| Pin the six declared runtime deps; move weasyprint out of extras | `20a4bef5` |
| Resolve torch/torchvision CUDA-build conflict; pin the resolved pair | `ceb4b8bf` |
| Stop empty env override of `OPENAI_API_KEY`; document required API token | `321260a9` |
| Share the `analysis_data` builder between analyze and compare | `72e142eb` |
| Separate heatmap retention budget; distinguish evicted from absent | `63f3d2e4` |
| Fit per-class calibration; re-derive thresholds | `e549ddb3` |

## Two findings worth reading

Some reviews were wrong, and the commits say so rather than quietly complying.

**Issue #1 carried a self-contradictory worked example.** The review specified tiered
ranking (urgent beats routine) *and* an example where a routine finding at p=0.97 became
the headline over an urgent one at p=0.55. Both cannot hold — Cardiomegaly is in
`URGENT_CONDITIONS`. `7e92b71d` implemented the rule and recorded the contradiction;
`4656c031` documented the rule at the ranking site with a consistent example.

**A reported off-by-one did not exist.** A review flagged `len(files) >= image_max_count`
in the blob evictor as evicting one file early. Measured, it retains exactly
`image_max_count`. The real defect on that line was over-eviction when *replacing* an
existing key, which `63f3d2e4` fixed instead — with the correction stated in the commit.
