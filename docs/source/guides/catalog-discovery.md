# Discover environments before running them

`openenv catalog` and `openenv discover` find an environment for a task from committed metadata only. They never install, import, pull or run candidate environments. To load an environment you already know, use [`AutoEnv`](auto-discovery) instead.

## Build a catalog

From a clone of the OpenEnv repository:

```bash
openenv catalog build \
  --repository . \
  --repository-uri https://github.com/huggingface/OpenEnv.git \
  --revision HEAD \
  --publisher example.org \
  --output catalog.json
```

- The catalog lists every tracked `openenv.yaml` in a direct child of `envs/` (change it with `--root`) at the resolved commit. Untracked files are ignored.
- `--publisher` names the publication authority. `example.org` is a placeholder. Setting it does not verify identity.
- Each card reads the environment's `openenv.yaml`, `pyproject.toml`, README front matter and optional `discovery.json`, and records the source URI, path and full revision.
- If any environment has invalid metadata, the command exits nonzero and the report has `complete: false`. An incomplete catalog can't be loaded for search.

## Search and inspect

```bash
openenv discover "client smoke test" --catalog catalog.json
openenv discover "client smoke test" --catalog catalog.json --json
openenv discover "" --catalog catalog.json --filter license=BSD-3-Clause
openenv catalog inspect "<identifier from the result>" --catalog catalog.json
```

Ranking is case-insensitive token overlap with descriptions, names, tags, capabilities and representative queries. Scores measure that lexical match only. `--filter` takes exact `field=value` matches on `license`, `provider`, `artifact_availability`, `name`, `tags` or `type`, and `--limit` caps the results (default 20).

`catalog inspect` returns the complete entry for an exact identifier from the snapshot. It never fetches URLs or visits deployments.

## Add a `discovery.json`

An optional `discovery.json` next to an environment's `openenv.yaml` adds metadata the other files lack. It accepts `description`, `tags`, `representative_queries` (empty, or two to five), `artifact_availability`, `license`, `license_source` and `agent_tools`. Echo's declaration:

```json
{
  "tags": ["openenv", "smoke-test"],
  "representative_queries": [
    "find an environment that echoes a message",
    "find a minimal environment for checking OpenEnv tool calls",
    "find an environment for a client smoke test"
  ],
  "agent_tools": {
    "protocol": "mcp",
    "names": ["echo_message", "echo_with_length"],
    "source": "envs/echo_env/server/echo_environment.py"
  }
}
```

`agent_tools.source` points to the file that defines the tools. Don't list simulation controls (such as `reset` or `step`) as agent tools. The build rejects known control names.

`discovery.json` takes precedence over `openenv.yaml`, the package description and README front matter. Conflicting license declarations become `unknown` with a warning.

Coding, BrowserGym, Calendar, Chess and Reasoning Gym also ship a `discovery.json`. For example, try `openenv discover "Python snippets and standard error" --catalog catalog.json`.

## Specification

The Environment Card format, identifiers, schemas (`openenv/discovery/schemas/0.1-draft/`), consumer validation rules and publication workflow are specified in [RFC 011](https://github.com/huggingface/OpenEnv/blob/main/rfcs/011-ard-catalog-discovery.md).
