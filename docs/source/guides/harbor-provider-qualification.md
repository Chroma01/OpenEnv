# Harbor provider qualification

An installed adapter is not evidence that a harness works with a given model provider. Qualify the harness version, model route, sandbox, capture implementation and task set together.

## Evaluation and training capture

- **`purpose="eval"`**: hosted OpenAI, native Anthropic and Hugging Face routes produce graded evaluation traces without engine token IDs. An eval trace never exports a training contract, even if its endpoint returns token IDs.
- **`purpose="train"`**: only with a verified token-capable endpoint. Training export keeps engine prompt IDs, sampled completion IDs, processed log probabilities and loss masks. `openenv.harbor.contract.to_trace_entries` rejects evaluation traces and fatal capture findings. Never rebuild token IDs by tokenizing rendered text or fill missing log probabilities with zeros.

A prompt rewrite can turn one rollout into several training rows without invalidating the sampled tokens. Report rows per rollout, repeated context, retained supervision and downstream weighting separately. Correct capture does not mean an efficient training configuration.

Native Anthropic requests keep their signed blocks and supported native metadata. Translation to another harness protocol rejects output that can't be preserved. The native streaming bridge buffers the upstream response and replays SDK-compatible events, so there is no upstream first-token latency.

## Evidence and support tiers

A qualification report has one cell per harness/provider pair, with providers `openai`, `anthropic`, `hf` and `vllm`. Keep the capture artifacts and attempt configuration next to it: model routes, revision pins, harness versions, task identities, sampling and source hashes.

| Status | Meaning |
|--------|---------|
| `eval_pass` | Completed, graded rollout with captured calls, no fatal capture findings and no training export. A task score of zero is a valid evaluation. An infrastructure failure or missing grade is not. |
| `capture_and_reader_pass` | Exact capture passed validation and the real training reader kept the expected supervision. |
| `optimizer_pass` | Current capture artifacts were consumed by a real optimizer diagnostic. Record model revision, input fingerprints, consumed rows, finite losses and finite nonzero gradients, and say whether it was diagnostic replay and whether weight sync was tested. |
| `failed`, `blocked`, `in_progress`, `not_run` | Kept as is. Never replaced by a pass from a different configuration. |

`harness_maturity_rows` derives the tier from validated cells:

- **stable**: `eval_pass` on OpenAI, Anthropic and HF, plus `optimizer_pass` on vLLM.
- **unstable**: all four profiles `failed` or `blocked`.
- **experimental**: anything in between (partial or pending).

Tiers describe the recorded coverage only, not universal compatibility or production-scale reliability.

## Using a report in the UI

Set `OPENENV_HARBOR_QUALIFICATION_REPORT` to the report JSON path to show the evidence in the Harbor Gradio UI. The UI offers stable harnesses by default, experimental ones behind an opt-in, and hides unstable ones. Without a report, every adapter is unqualified and needs the opt-in. Recorded results don't certify a newly entered endpoint or pin the harness installation.

Some results apply to a specific profile, which the UI passes to the rollout and shows in the agent label: ACP is qualified with `opencode-1.18.30`, NeMo with `shell-1.9.0` (when the example workflow package is in the checkout). Selecting a profile uses a local seam copy and leaves the global adapter registry unchanged. In code, pass `harness_profile=` to `run_rollout` or `build_trial_config`. Unknown profiles fail.

## Recorded results

The latest recorded qualification (15 September 2026: 29 adapters, four provider profiles, two tasks per pair) lists each adapter's tier and per-provider status, the exact models and serving configuration, and the known limitations. See [Harbor qualification: 15 September 2026](https://github.com/huggingface/OpenEnv/blob/main/envs/harbor_env/qualification-2026-09-15.md).

## Regression and live validation

Run the deterministic Harbor tests from the repository root:

```bash
PYTHONPATH=src:envs python -m pytest tests/envs/test_harbor*.py -q
```

They cover provider conversion, capture graphs, export masks, reconciliation, routing, lifecycle and evidence gates. They don't replace live harness runs.

For live qualification:

- Use isolated services and immutable source snapshots. Fix the task set and versions before launch, bound sandbox concurrency and save each result before moving on.
- Resume by scheduling only the missing cases into a new attempt directory, keeping prior failures and provenance.
- A capture counts as optimizer evidence only if its source hash and row fingerprint match. A newer retry doesn't inherit an older optimizer pass.
- If an adapter needs a separate application, workflow, vision input or vendor account, report the missing prerequisite. Don't substitute another agent, drop observations, relax token checks or count a reachable endpoint as success.
