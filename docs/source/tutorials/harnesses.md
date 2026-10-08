# Harnesses in OpenEnv

A harness is the loop around a model: it sends the conversation to the model, runs the tools the model calls, and decides when the episode ends. There are three ways to put a harness and an OpenEnv environment together. They differ in who owns the loop and whether the model's tokens are recorded, and that decides what each one is for.

| | White-box | Black-box | Agent inside the environment |
|---|---|---|---|
| Who owns the loop | TRL (`environment_factory`) | A real agent (OpenCode, Claude Code, Codex, …) | A real agent |
| Who generates each turn | The trainer | The agent, calling your model through a capture proxy | The agent, calling its own model |
| Token ids and logprobs | Yes, the trainer generated them | Yes, captured by the proxy (against vLLM or SGLang) | No |
| Episode | The model acts through the environment's tools | A task: the agent runs to completion and a verifier scores it | A conversation: each `step()` is one message, and the environment's tools are injected over MCP |
| Use it to | Train a model on your environment's tools | Train or evaluate the model behind a real agent | Evaluate or serve an agent you don't train |
| Start here | [BrowserGym](browsergym-harness) | [Harbor](harbor-harness) | [Claude Code on τ²-bench](claude-code-harness) |

## Which One to Use

- **You want to train a model on your environment and the loop can be simple.** Use the white-box path. TRL's `GRPOTrainer` runs the tool loop, so token ids and logprobs come for free, and the environment only supplies tools and a reward. [Wordle](wordle-grpo) is the same pattern on a text game.
- **You want to train or evaluate the model behind an existing agent.** Use Harbor. The agent keeps its own planner, tools and context management, and a proxy records every model call it makes. Episodes are task-shaped: an instruction, a run to completion, and a verifier.
- **You want an existing agent to work with your environment's tools, over several turns.** Use `HarnessEnvironment` from [RFC 005](https://github.com/huggingface/OpenEnv/blob/main/rfcs/005-agentic-harnesses.md). The agent runs inside the environment, the environment injects its tools and scores the conversation with a rubric, for example against a simulated user. Nothing is recorded, so it evaluates and serves, but does not train.

[The ultimate guide to multi-harness RL](https://huggingface.co/spaces/FineEnvs/multi-harness-rl) goes deeper into the white-box and black-box paths and why training across several agents matters.

> [!NOTE]
> `openenv.core.harness` also has a session runtime (`ResourceSession`, `MCPHarnessAdapter`, `build_harness_rollout_func`), where OpenEnv runs a small tool loop and a model generates each turn. [`openenv collect`](sft-warmup) uses it to collect rollouts for SFT. Its `HarnessAdapter` is a different class from RFC 005's `AgenticHarnessAdapter`.
