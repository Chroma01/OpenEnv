# Harbor qualification: 15 September 2026

Recorded results for the Harbor adapters. How qualification works, what each status means and how support tiers are derived is in the [Harbor provider qualification guide](https://huggingface.co/docs/openenv/guides/harbor-provider-qualification).

The completed qualification attempted all 29 adapters on four provider profiles, with two fixed tasks per pair (116 pairs). Results are compatibility smoke tests, not benchmark pass@1 scores. “Stable” means passing this recorded coverage; it does not certify arbitrary models, harness upgrades, or production-scale reliability.

| Provider profile | Model | Passing adapters |
|---|---|---:|
| OpenAI evaluation | `gpt-5.4-mini-2026-03-17` | 21/29 |
| Native Anthropic evaluation | `claude-sonnet-4-5-20250929` | 20/29 |
| Hugging Face evaluation | `Qwen/Qwen3.5-9B:together` | 19/29 |
| vLLM training capture and optimizer diagnostic | `Qwen/Qwen3.5-4B` | 21/29 |

The HF route is pinned, but its hosted weights are not an immutable revision. The vLLM model revision is `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. That profile used vLLM 0.25.1, TP=1, DP=1, BF16, a 131072-token context, processed log probabilities, engine token IDs, Qwen3 XML tool parsing, Qwen3 reasoning parsing with thinking disabled, and no image/video inputs.

There are **14 stable, 9 experimental, and 6 unstable adapters**. A failed pair means the two-task qualification did not pass; it does not necessarily mean both tasks failed or that the adapter can never support that provider.

| Adapter | Tier | OpenAI | Anthropic | HF | vLLM |
|---|---|---|---|---|---|
| acp | experimental | failed | eval_pass | eval_pass | optimizer_pass |
| antigravity-cli | experimental | failed | failed | eval_pass | optimizer_pass |
| antigravity-sdk | unstable | failed | failed | failed | failed |
| claude-code | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| cline-cli | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| codex | experimental | eval_pass | eval_pass | failed | optimizer_pass |
| computer-1 | unstable | failed | failed | failed | failed |
| copilot-cli | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| cursor-cli | unstable | failed | failed | failed | failed |
| devin | unstable | failed | failed | failed | failed |
| eve | unstable | failed | failed | failed | failed |
| gemini-cli | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| goose | experimental | eval_pass | eval_pass | failed | optimizer_pass |
| grok-build | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| kimi-cli | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| mimo | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| mini-swe-agent | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| nemo-agent | experimental | eval_pass | eval_pass | failed | optimizer_pass |
| openclaw | experimental | eval_pass | eval_pass | failed | failed |
| opencode | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| openhands | experimental | eval_pass | eval_pass | eval_pass | failed |
| openhands-sdk | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| pi | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| qwen-coder | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| rovodev-cli | unstable | failed | failed | failed | failed |
| swe-agent | experimental | eval_pass | failed | eval_pass | optimizer_pass |
| terminus-2 | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |
| trae-agent | experimental | eval_pass | failed | eval_pass | optimizer_pass |
| vibe | stable | eval_pass | eval_pass | eval_pass | optimizer_pass |

## Scope and known limitations

The optimizer diagnostics consumed 99 current capture rows across 21 adapters using the real `AsyncGRPOTrainer`, with finite losses and finite nonzero gradients. They used a diagnostic advantage of +1 and did not synchronize weights. This establishes capture consumption by the trainer, not reward-normalized learning, long-run stability, or correct weighting when a rollout produces multiple rows. Claude Code and other prompt-rewriting harnesses still need row-budget and weighting checks for a particular training configuration.

ACP qualification applies only to the `opencode-1.18.30` profile, and NeMo qualification only to `shell-1.9.0`. ACP has partial native usage evidence; NeMo lacks independent native token counts. Engine capture remains authoritative, and these results do not qualify arbitrary ACP agents or NeMo workflows.

Codex, Goose, and NeMo retain HF failures. Antigravity CLI retains OpenAI and Anthropic failures. SWE-agent timed out on Anthropic; Trae-agent captured no Anthropic calls. OpenClaw and OpenHands retain training trajectory reconciliation failures. Antigravity SDK also failed strict reconciliation despite executing tools. Missing vendor credentials or application prerequisites prevented qualification of Cursor, Devin, Rovo Dev, and Eve. Computer-1 needs a separate desktop/vision qualification. Keep these failures visible; do not relax token checks to promote an adapter.

The final combined regression run passed 567 tests with two skips; native Anthropic SDK streaming replay was also checked separately. Live qualification and optimizer replay used separate services and source snapshots. Updating this documentation or a qualification report does not restart training, change an existing training snapshot, or deploy the adapter changes. A running process continues to use its configured source and services.
