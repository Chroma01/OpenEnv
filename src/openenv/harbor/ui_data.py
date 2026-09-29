"""What the Harbor UI shows about a dataset: its tasks, what is in each one, and what a task runs with.

Discovery (`tasks.py`) lists task directories and deliberately reads nothing inside them, because it
is on the latency path of every Task API call. A person choosing a task needs more than a directory
name: a title to find it by, the files to judge it by, and the settings it will run with. So this
module does read inside task directories, but only for the UI, on demand, and with a per-dataset
cache so a page load never walks a dataset twice.

Nothing here boots a sandbox or calls a model.
"""

from __future__ import annotations

import os
import re
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .tasks import HarborTaskProvider, resolve_task_dirs

# Metadata keys that hold the answer. Harbor puts no rules on `[metadata]`, and some datasets keep
# the gold answer there (`gold_answer = "4"`). The raw `task.toml` stays viewable, the dataset is
# public anyway, but a summary someone skims before running a model should not hand them the answer.
_ANSWER_KEY = re.compile(r"(gold|answer|solution|expected|reference_output)", re.I)

# The file tree is for reading a task, not for mirroring a repository: a task with a vendored
# codebase can hold tens of thousands of files, and the browser only needs the shape.
MAX_TREE_FILES = 2000
MAX_FILE_BYTES = 256 * 1024

_ROWS: dict[str, list[dict[str, Any]]] = {}
_ROWS_LOCK = threading.Lock()


def _toml(task_dir: Path) -> dict[str, Any]:
    try:
        return tomllib.loads((task_dir / "task.toml").read_text(errors="replace"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _first_line(task_dir: Path, limit: int = 160) -> str:
    """The first meaningful line of the instruction, used as a title when a task declares none."""
    try:
        with (task_dir / "instruction.md").open(errors="replace") as fh:
            for line in fh:
                text = line.strip().lstrip("#").strip()
                if text:
                    return text[:limit]
    except OSError:
        pass
    return ""


_MD = re.compile(
    r"`+|\*\*|__|^\s*[-*+>]\s+|^\s*\d+[.)]\s+|!?\[([^\]]*)\]\([^)]*\)", re.M
)


def _paragraphs(task_dir: Path, limit: int = 3, chars: int = 360) -> list[str]:
    """The instruction's first few prose paragraphs as plain text, headings and code left out.

    Only the start of the file is read: a brief is two lines on a card.
    """
    try:
        with (task_dir / "instruction.md").open(errors="replace") as fh:
            head = fh.read(6000)
    except OSError:
        return []
    out, block, fenced = [], [], False
    for line in head.splitlines() + [""]:
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        if fenced or line.lstrip().startswith("#"):
            continue
        if line.strip():
            block.append(line.strip())
            continue
        if block:
            text = _MD.sub(lambda m: m.group(1) or "", " ".join(block)).strip()
            if text:
                out.append(text[:chars])
            block = []
            if len(out) >= limit:
                break
    return out


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if isinstance(v, (str, int, float))]
    return []


def task_row(index: int, task_dir: Path) -> dict[str, Any]:
    """One task as the task list shows it.

    Harbor's schema fixes `[task]` (name, description, keywords) but leaves `[metadata]` free-form,
    so a title, a category and a difficulty are each looked for under the names datasets actually
    use, falling back to the instruction's first line for the title.

    Args:
        index (`int`):
            The task's position in its dataset, which is its identity everywhere downstream.
        task_dir (`Path`):
            The task directory.

    Returns:
        `dict` with `index`, `name`, `title`, `category`, `difficulty` and `keywords`.
    """
    doc = _toml(task_dir)
    task, meta = doc.get("task") or {}, doc.get("metadata") or {}
    title = (
        meta.get("title")
        or task.get("description")
        or _first_line(task_dir)
        or task_dir.name
    )
    keywords: list[str] = []
    for k in _strings(task.get("keywords")) + _strings(meta.get("keywords")):
        if k not in keywords and k != meta.get("category"):
            keywords.append(k)
    return {
        "index": index,
        "name": task_dir.name,
        "title": str(title)[:240],
        "category": str(meta.get("category") or meta.get("domain") or ""),
        "difficulty": str(meta.get("difficulty") or meta.get("difficulty_tier") or ""),
        "keywords": keywords[:8],
        "paragraphs": _paragraphs(task_dir),
    }


def _briefs(rows: list[dict[str, Any]]) -> None:
    """Give each row a `brief`: its first paragraph that says something about this task.

    Many datasets open every instruction with the same preamble ("You are an agent, your working
    directory is /app"), which on a card reads as the same sentence 2,000 times. A paragraph shared by
    more than a fifth of a dataset's tasks is that preamble, so it is skipped, as is one that only
    repeats the title.
    """
    counts: dict[str, int] = {}
    for r in rows:
        for p in set(r.get("paragraphs") or []):
            counts[p] = counts.get(p, 0) + 1
    common = max(3, len(rows) // 5)
    for r in rows:
        title = r["title"].strip().rstrip(".").lower()
        paras = r.pop("paragraphs", None) or []
        r["brief"] = next(
            (
                p[:220]
                for p in paras
                if counts.get(p, 0) < common
                and p.strip().rstrip(".").lower() != title
                and not title.startswith(p.strip().rstrip(".").lower()[:60])
            ),
            "",
        )


def task_rows(spec: str) -> list[dict[str, Any]]:
    """Every task in a dataset as the task list shows it, cached per dataset.

    Reading one small `task.toml` per task is milliseconds on local disk but far slower on a
    mounted bucket, so the reads run in parallel and the result is kept for the process lifetime.

    Args:
        spec (`str`):
            Dataset spec: HF dataset repo, local directory, or Harbor registry `name@version`.

    Returns:
        `list[dict]`, one row per task, in index order.
    """
    with _ROWS_LOCK:
        if spec in _ROWS:
            return _ROWS[spec]
    dirs = resolve_task_dirs(spec)
    workers = int(os.environ.get("OPENENV_UI_INDEX_WORKERS", "16"))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(lambda pair: task_row(*pair), enumerate(dirs)))
    _briefs(rows)
    with _ROWS_LOCK:
        _ROWS[spec] = rows
    return rows


def common_tags(spec: str) -> set[str]:
    """Tags nine in ten of a dataset's tasks carry (`terminal` on a terminal dataset): they tell no two
    tasks apart. Empty until the dataset's rows are cached, and for datasets of three tasks or fewer."""
    with _ROWS_LOCK:
        rows = _ROWS.get(spec) or []
    if len(rows) <= 3:
        return set()
    counts: dict[str, int] = {}
    for r in rows:
        for k in r.get("keywords") or []:
            counts[k] = counts.get(k, 0) + 1
    return {k for k, n in counts.items() if n >= 0.9 * len(rows)}


def _config_sections(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """`task.toml` as the settings a person checks before running: where, with what, for how long."""
    env = doc.get("environment") or {}
    agent = doc.get("agent") or {}
    verifier = doc.get("verifier") or {}

    def secs(value: Any) -> str:
        try:
            s = float(value)
        except (TypeError, ValueError):
            return ""
        return f"{s / 60:.0f} min" if s >= 120 else f"{s:.0f} s"

    network = env.get("network_mode") or (
        "internet"
        if env.get("allow_internet") is True
        else "none"
        if env.get("allow_internet") is False
        else ""
    )
    image = env.get("docker_image") or "built from environment/Dockerfile"
    sections = [
        (
            "Environment",
            [
                ("Image", image),
                ("Working directory", env.get("workdir", "")),
                ("CPUs", env.get("cpus", "")),
                ("Memory", f"{env['memory_mb']:,} MB" if env.get("memory_mb") else ""),
                (
                    "Storage",
                    f"{env['storage_mb']:,} MB" if env.get("storage_mb") else "",
                ),
                ("GPUs", env.get("gpus", "")),
                ("Network", network),
                ("Build timeout", secs(env.get("build_timeout_sec"))),
            ],
        ),
        (
            "Agent",
            [
                ("Timeout", secs(agent.get("timeout_sec"))),
                ("User", agent.get("user", "")),
                ("Network", (doc.get("agent") or {}).get("network_mode", "")),
            ],
        ),
        (
            "Verifier",
            [
                ("Timeout", secs(verifier.get("timeout_sec"))),
                ("User", verifier.get("user", "")),
                ("Network", verifier.get("network_mode", "")),
            ],
        ),
    ]
    out = []
    for label, rows in sections:
        kept = [_row(k, v) for k, v in rows if v not in ("", None)]
        if kept:
            out.append({"label": label, "rows": kept})
    health = env.get("healthcheck") or {}
    if health.get("command"):
        out.append(
            {
                "label": "Setup",
                "rows": [
                    {
                        "key": "Before the agent",
                        "value": f"runs a setup command ({len(str(health['command'])):,} characters)"
                        + (
                            f", up to {secs(health['timeout_sec'])}"
                            if health.get("timeout_sec")
                            else ""
                        ),
                    }
                ],
            }
        )
    servers = env.get("mcp_servers") or []
    if servers:
        out.append(
            {
                "label": "MCP servers",
                "rows": [
                    {
                        "key": str(s.get("name", "server")),
                        "value": str(s.get("transport") or s.get("url") or ""),
                    }
                    for s in servers
                    if isinstance(s, dict)
                ],
            }
        )
    # Names only. Values are templates such as `${HF_TOKEN}` here, but a dataset could hard-code one,
    # and the UI is the wrong place to find that out.
    for label, table in (
        ("Environment variables", env.get("env")),
        ("Verifier variables", verifier.get("env")),
    ):
        if isinstance(table, dict) and table:
            out.append(
                {
                    "label": label,
                    "rows": [{"key": k, "value": ""} for k in sorted(table)],
                }
            )
    return out


def _row(key: str, value: Any) -> dict[str, str]:
    """A settings row. A digest-pinned image is shortened for reading; `full` keeps it for hover."""
    text = str(value)
    if "@sha256:" in text:
        name, _, digest = text.partition("@sha256:")
        return {"key": key, "value": f"{name}@sha256:{digest[:12]}…", "full": text}
    return {"key": key, "value": text}


def _metadata(doc: dict[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    """Free-form `[metadata]` fields to show, and the answer-like ones withheld from the summary."""
    shown, hidden = [], []
    for key, value in (doc.get("metadata") or {}).items():
        if _ANSWER_KEY.search(key):
            hidden.append(key)
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        if isinstance(value, dict):
            continue
        text = str(value)
        if text:
            shown.append({"key": key, "value": text[:300]})
    return shown, hidden


def file_tree(task_dir: Path) -> tuple[list[dict[str, Any]], bool]:
    """Files under a task directory as `(path, size)`, capped so a vendored repo cannot flood the page.

    Returns:
        `tuple` of the file list and whether it was cut at `MAX_TREE_FILES`.
    """
    files: list[dict[str, Any]] = []
    root = task_dir.resolve()
    # Files at the task's top level first (instruction.md, task.toml), then each folder in turn,
    # walked lazily: a vendored tree of a million files is read only as far as the cap.
    for folder, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(n for n in names if not n.startswith(".")):
            if len(files) >= MAX_TREE_FILES:
                return files, True
            path = Path(folder) / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            files.append({"path": path.relative_to(root).as_posix(), "size": size})
    return files, False


def task_detail(spec: str, index: int) -> dict[str, Any]:
    """Everything the task view shows for one task. Reads files, runs nothing.

    Args:
        spec (`str`):
            Dataset spec.
        index (`int`):
            Task index within the dataset.

    Returns:
        `dict` with the row fields plus `description`, `instruction`, `config`, `metadata`,
        `withheld` (answer-like metadata keys left out of the summary), `files` and
        `files_truncated`.
    """
    task_dir = HarborTaskProvider([spec]).task_dir(spec, int(index))
    doc = _toml(task_dir)
    try:
        instruction = (task_dir / "instruction.md").read_text(errors="replace")
    except OSError:
        instruction = ""
    shown, hidden = _metadata(doc)
    files, truncated = file_tree(task_dir)
    row = task_row(int(index), task_dir)
    row.pop("paragraphs", None)
    return {
        **row,
        "dataset": spec,
        "description": str((doc.get("task") or {}).get("description") or ""),
        "instruction": instruction,
        "config": _config_sections(doc),
        "metadata": shown,
        "withheld": hidden,
        "files": files,
        "files_truncated": truncated,
    }


def reads_environment(spec: str, index: int) -> bool:
    """Whether a task would read this server's environment variables or files.

    Harbor resolves `${VAR}` in `task.toml` (`[verifier.env]`, `[environment.env]`) from the process
    environment, and Docker Compose expands `$VAR` in a compose file the same way. Either one hands
    the task this server's keys. A compose file can also reach past its own folder: `env_file`,
    `include` and `extends` read other files, and a bind mount of an absolute or parent path puts
    the host's files in the container, which a local container backend runs on this machine. Every
    YAML file under `environment/` is checked, since one compose file can include another.
    """
    task_dir = HarborTaskProvider([spec]).task_dir(spec, int(index))
    checks = [(task_dir / "task.toml", (_HARBOR_VAR,))]
    checks += [
        (p, _COMPOSE_READS)
        for p in (task_dir / "environment").rglob("*")
        if p.suffix in (".yml", ".yaml") and p.is_file()
    ]
    for path, patterns in checks:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if any(pattern.search(text) for pattern in patterns):
            return True
    return False


# Harbor expands only a whole value of `${VAR}` or `${VAR:-default}` (harbor.utils.env), so any `${`
# is flagged; Docker Compose also expands a bare `$VAR`.
_HARBOR_VAR = re.compile(r"\$\{")
_COMPOSE_VAR = re.compile(r"\$(\{|[A-Za-z_])")
_COMPOSE_READS = (
    _COMPOSE_VAR,
    re.compile(r"^\s*(env_file|include|extends)\s*:", re.M),
    # a bind mount from the host: `- /abs:/x`, `- ~/x:/x`, `- ../x:/x`, or `source: /abs`
    re.compile(r"^\s*-\s*[\"']?(/|~|\.\.)[^:\n]*:", re.M),
    re.compile(r"^\s*source\s*:\s*[\"']?(/|~|\.\.)", re.M),
)


def read_task_file(spec: str, index: int, path: str) -> dict[str, Any]:
    """One file from a task directory, for the file viewer.

    The path comes from the browser, so it is resolved and checked to stay inside the task
    directory: `..` segments and symlinks pointing elsewhere are refused.

    Args:
        spec (`str`):
            Dataset spec.
        index (`int`):
            Task index.
        path (`str`):
            Path relative to the task directory.

    Returns:
        `dict` with `path`, `size`, and either `text` (possibly `truncated`) or `binary: True`, or
        `error` when the path is refused or missing.
    """
    root = HarborTaskProvider([spec]).task_dir(spec, int(index)).resolve()
    target = (root / str(path)).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        return {"path": path, "error": "not a file in this task"}
    size = target.stat().st_size
    with target.open("rb") as fh:
        raw = fh.read(MAX_FILE_BYTES)
    if b"\x00" in raw[:8192]:
        return {"path": path, "size": size, "binary": True}
    return {
        "path": path,
        "size": size,
        "text": raw.decode("utf-8", errors="replace"),
        "truncated": size > MAX_FILE_BYTES,
    }


def search_hub(query: str = "", limit: int = 40) -> list[dict[str, Any]]:
    """Harbor datasets on the Hugging Face Hub, most downloaded first.

    Harbor datasets carry the `harbor` tag (and usually `rl-environment`), which is what makes them
    findable without a registry of our own.

    Args:
        query (`str`, *optional*, defaults to `""`):
            Free-text filter on the dataset id.
        limit (`int`, *optional*, defaults to `40`):
            Maximum number of results.

    Returns:
        `list[dict]` with `id`, `downloads`, `likes` and whether it is tagged `rl-environment`.
    """
    from huggingface_hub import HfApi

    # Anonymous: the server's own token would list its private datasets to any visitor.
    results = HfApi(token=False).list_datasets(
        filter="harbor",
        search=(query or "").strip() or None,
        sort="downloads",
        limit=max(1, min(int(limit), 100)),
    )
    return [
        {
            "id": d.id,
            "downloads": d.downloads or 0,
            "likes": d.likes or 0,
            "rl_environment": "rl-environment" in (d.tags or []),
        }
        for d in results
    ]


HF_ROUTER = "https://router.huggingface.co/v1"
_MODELS: tuple[float, list[dict[str, Any]]] = (0.0, [])
_MODELS_LOCK = threading.Lock()


def hf_models(ttl: float = 600.0) -> list[dict[str, Any]]:
    """Chat models on Hugging Face Inference Providers, for the model picker.

    The router's model list is public and says, per provider, whether it is live, whether it calls
    tools, its context length and its price. An agent is useless without tool calls, so each model
    carries `tools` (any live provider supports them) and the picker offers those first. Cached for
    `ttl` seconds: it is the same list for every visitor.

    Returns:
        `list[dict]` with `id`, `tools`, `context` (the largest across providers) and `providers`,
        each `{name, tools, context, price_in, price_out}` in USD per million tokens.
    """
    import json
    import time
    import urllib.request

    global _MODELS
    with _MODELS_LOCK:
        fetched, cached = _MODELS
        if cached and time.time() - fetched < ttl:
            return cached
    with urllib.request.urlopen(f"{HF_ROUTER}/models", timeout=20) as r:
        data = json.loads(r.read()).get("data") or []
    out = []
    for m in data:
        providers = []
        for p in m.get("providers") or []:
            if p.get("status", "live") != "live":
                continue
            price = p.get("pricing") or {}
            providers.append(
                {
                    "name": p.get("provider", ""),
                    "tools": bool(p.get("supports_tools")),
                    "context": p.get("context_length") or 0,
                    "price_in": price.get("input"),
                    "price_out": price.get("output"),
                }
            )
        if not providers or "text" not in (
            (m.get("architecture") or {}).get("output_modalities") or ["text"]
        ):
            continue
        out.append(
            {
                "id": m.get("id", ""),
                "tools": any(p["tools"] for p in providers),
                "context": max((p["context"] for p in providers), default=0),
                "providers": providers,
            }
        )
    # Tool-calling models first, then by how widely served, which is a fair proxy for how well used.
    out.sort(key=lambda m: (not m["tools"], -len(m["providers"])))
    with _MODELS_LOCK:
        _MODELS = (time.time(), out)
    return out


# ── adding a Hub dataset from the page ──────────────────────────────────────────────────────────
# A download of a task suite is thousands of small files and can take minutes, so an add runs in the
# background and the page polls it: first a check that the dataset is laid out the way Harbor reads
# it (before anything is downloaded), then the download with its file count, then reading the tasks.

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
MAX_ADDS_AT_ONCE = 2


def _max_add_bytes() -> int:
    try:
        return int(float(os.environ.get("OPENENV_HARBOR_UI_MAX_ADD_GB") or 5) * 1e9)
    except ValueError:
        return int(5e9)


def hub_summary(spec: str) -> dict[str, Any]:
    """What adding a Hub dataset would bring: `tasks` held as `tasks/<name>/`, the layout Harbor
    loads (`None` when it has none), and the repository's size in `bytes`.

    Read from the Hub's listing and metadata, anonymously, so nothing is downloaded to find out. The
    size is the whole repository's, an upper bound on what the `tasks/` download takes.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFolder

    api = HfApi(token=False)
    tasks = None
    try:
        folders = [
            e.path
            for e in api.list_repo_tree(
                spec, repo_type="dataset", path_in_repo="tasks", recursive=False
            )
            if isinstance(e, RepoFolder)
        ]
        if folders:
            first = {
                e.path.rsplit("/", 1)[-1]
                for e in api.list_repo_tree(
                    spec, repo_type="dataset", path_in_repo=folders[0], recursive=False
                )
            }
            if "task.toml" in first:
                tasks = len(folders)
            else:
                # Grouped (`tasks/<field>/<task>/`), which the loader searches the same way: count
                # the task files. Bounded, so a vast repository cannot stall the page.
                tasks = 0
                for i, e in enumerate(
                    api.list_repo_tree(
                        spec, repo_type="dataset", path_in_repo="tasks", recursive=True
                    )
                ):
                    tasks += e.path.endswith("/task.toml")
                    if i > 50_000:
                        break
                tasks = tasks or None
    except Exception:  # noqa: BLE001 - no `tasks/` folder answers 404
        pass
    size = None
    try:
        size = getattr(api.dataset_info(spec), "used_storage", None)
    except Exception:  # noqa: BLE001
        pass
    return {"tasks": tasks, "bytes": size}


def _tasks_bytes(spec: str, cap: int, max_files: int = 500_000) -> int | None:
    """The size of a Hub dataset's `tasks/` folder from its file listing, or `None` if unknown.

    Stops counting once the total passes `cap`, since that already settles the question; a listing
    longer than `max_files` is not read to the end, and counts as unknown.
    """
    from huggingface_hub import HfApi

    total = 0
    try:
        for n, entry in enumerate(
            HfApi(token=False).list_repo_tree(
                spec, repo_type="dataset", path_in_repo="tasks", recursive=True
            )
        ):
            total += int(getattr(entry, "size", 0) or 0)
            if total > cap:
                return total
            if n >= max_files:
                return None
    except Exception:  # noqa: BLE001 - no listing, no size
        return None
    return total


def _progress(job: dict[str, Any]) -> Any:
    """A silent tqdm that records the Hub download's file count in `job`.

    The Hub client uses the same class for each large file's byte counter (`unit="B"`), so only the
    bar that counts files is recorded.
    """
    import io

    from tqdm import tqdm

    class Progress(tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["file"] = io.StringIO()
            kwargs.pop("disable", None)
            self._files = kwargs.get("unit", "it") != "B"
            super().__init__(*args, **kwargs)
            if self._files and self.total:
                job["total"] = int(self.total)

        def update(self, n: float | None = 1) -> bool | None:
            out = super().update(n)
            if self._files:
                job["done"] = int(self.n)
            return out

    return Progress


def added_spec(spec: str, settings: Any) -> str:
    """What the loader is given for a Hub dataset added from the page.

    With the Space's bucket mounted, the dataset's folder on the mount, the same place `push` puts
    the datasets it serves; otherwise the Hub id, which downloads to this machine's cache.
    """
    if settings.bucket and settings.bucket_mount:
        return str(settings.bucket_mount / spec.replace("/", "__"))
    return spec


def hub_id(spec: str) -> str:
    """The Hub id a dataset spec stands for: `org/name` for a `/data/org__name` mount folder."""
    name = Path(spec).name if spec.startswith("/") else ""
    return name.replace("__", "/", 1) if "__" in name else spec


def added_in_bucket(settings: Any, served: list[str]) -> list[str]:
    """Datasets on the mounted bucket that the server was not started with: the ones added from the
    page. The bucket itself is the list, so it survives restarts and needs no file of its own."""
    mount = settings.bucket_mount
    if not (settings.bucket and mount):
        return []
    out = []
    for p in sorted(mount.iterdir()):
        if (
            p.is_dir()
            and "__" in p.name
            and not p.name.startswith(".")
            and (p / "tasks").is_dir()
        ):
            if str(p) not in served and hub_id(str(p)) not in served:
                out.append(str(p))
    return out


def _forget(spec: str) -> None:
    from . import tasks

    with tasks._LOCK:
        tasks._CACHE.pop(spec, None)
    with _ROWS_LOCK:
        _ROWS.pop(spec, None)


def remove_added(spec: str, settings: Any) -> None:
    """Delete a dataset added from the page: its folder in the bucket, or its local download.

    Only ever a folder this module put there: one under the bucket mount named `org__name`, or one in
    the dataset cache. The caller checks that `spec` was added from the page, not served.
    """
    from . import tasks

    if spec.startswith("/") and settings.bucket and settings.bucket_mount:
        folder = Path(spec).resolve()
        if folder.parent != settings.bucket_mount.resolve() or "__" not in folder.name:
            raise ValueError("not a dataset folder on the bucket")
        from huggingface_hub import HfApi

        api = HfApi()
        paths = [
            f.path
            for f in api.list_bucket_tree(
                settings.bucket, prefix=f"{folder.name}/", recursive=True
            )
            if getattr(f, "type", "file") != "directory"
            and not type(f).__name__.endswith("Folder")
        ]
        for i in range(0, len(paths), 1000):
            api.batch_bucket_files(settings.bucket, delete=paths[i : i + 1000])
    else:
        root = tasks._DATASET_ROOT.resolve()
        folder = (tasks._DATASET_ROOT / spec.replace("/", "__")).resolve()
        if folder.parent != root:
            raise ValueError("not a downloaded dataset")
        if folder.is_dir():
            import shutil

            shutil.rmtree(folder)
    _forget(spec)


def _copy_to_bucket(
    spec: str, settings: Any, job: dict[str, Any], expected: int
) -> str:
    """Copy a Hub dataset into the Space's bucket, server side, and wait for the mount to show it.

    The same copy `push` makes for the datasets it serves: by content hash, so nothing is downloaded
    or uploaded and a suite of thousands of files takes seconds. The mount then shows the new folder,
    which is waited for rather than assumed.
    """
    import time

    from huggingface_hub import HfApi

    prefix = spec.replace("/", "__")
    job["state"] = "copying"
    HfApi().copy_files(
        f"hf://datasets/{spec}/", f"hf://buckets/{settings.bucket}/{prefix}/"
    )
    job["state"] = "mounting"
    folder = settings.bucket_mount / prefix
    deadline = time.time() + 180
    from .tasks import _task_dirs_from_directory

    while time.time() < deadline:
        try:
            if (folder / "tasks").is_dir() and len(
                _task_dirs_from_directory(folder)
            ) >= expected:
                return str(folder)
        except OSError:
            pass
        time.sleep(3)
    raise ValueError(
        "Copied to the bucket, but the mount has not shown it yet. It appears when the Space restarts."
    )


def start_add(spec: str, on_added: Any, settings: Any = None) -> dict[str, Any]:
    """Start adding `spec` in the background, or return the add already under way.

    `on_added(loader_spec)` is called once its tasks are read. The caller has checked that `spec` is a
    public Hub dataset id; this checks its layout and size, puts it where `added_spec` says (the
    bucket, or this machine), and indexes it.
    """
    import time

    with _JOBS_LOCK:
        job = _JOBS.get(spec)
        if job and job["state"] not in ("error", "done"):
            return dict(job)
        busy = sum(1 for j in _JOBS.values() if j["state"] not in ("error", "done"))
        if busy >= MAX_ADDS_AT_ONCE:
            return {
                "spec": spec,
                "state": "error",
                "error": "Another dataset is being added. Try again in a moment.",
            }
        job = {
            "spec": spec,
            "state": "checking",
            "done": 0,
            "total": 0,
            "tasks": None,
            "error": None,
            "started": time.time(),
        }
        _JOBS[spec] = job

    def run() -> None:
        try:
            summary = hub_summary(spec)
            expected = summary["tasks"]
            if not expected:
                raise ValueError(
                    "This dataset has no tasks/<name>/ folder, which is how Harbor datasets are laid out."
                )
            if summary["bytes"] is None:
                # the Hub has no size for some repositories; measure what the download would take
                summary["bytes"] = _tasks_bytes(spec, _max_add_bytes())
            if summary["bytes"] is None:
                raise ValueError(
                    "Couldn't tell how big this dataset is, so it isn't added from the page."
                )
            if summary["bytes"] > _max_add_bytes():
                raise ValueError(
                    f"This dataset is {summary['bytes'] / 1e9:.1f} GB, over the "
                    f"{_max_add_bytes() / 1e9:.0f} GB this server adds from the page "
                    "(OPENENV_HARBOR_UI_MAX_ADD_GB)."
                )
            job["bytes"] = summary["bytes"]
            job["expected"] = expected
            if settings is not None and settings.bucket and settings.bucket_mount:
                target = _copy_to_bucket(spec, settings, job, expected)
            else:
                job["state"] = "downloading"
                resolve_task_dirs(spec, tqdm_class=_progress(job))
                target = spec
            job["state"] = "indexing"
            job["tasks"] = len(task_rows(target))
            job["target"] = target
            on_added(target)
            job["state"] = "done"
        except Exception as exc:  # noqa: BLE001 - reported to the page, never raised
            job.update(state="error", error=f"{str(exc)[:240]}")

    threading.Thread(target=run, daemon=True, name=f"harbor-add-{spec}").start()
    return dict(job)


def add_status(spec: str) -> dict[str, Any] | None:
    with _JOBS_LOCK:
        job = _JOBS.get(spec)
        return dict(job) if job else None
