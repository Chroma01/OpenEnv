"""UI downloads must preserve the exact supervision consumed by the trainer."""

import json

import pytest
from harbor_env.harness import to_trace_entries
from openenv.harbor.contract import export_training_contract
from openenv.harbor.models import HarborRolloutResult, HarborTurn
from openenv.harbor.ui import _contract


def rollout():
    return HarborRolloutResult(
        task_name="partial-mask",
        reward=1.0,
        capture_level="tokens",
        turns=[
            HarborTurn(
                turn=0,
                node_id="agent",
                prompt_token_ids=[1, 2],
                completion_token_ids=[3, 4],
                per_token_logps=[-0.2, -0.3],
                loss_mask=[0, 0, 1, 0],
            )
        ],
    )


def test_ui_download_and_trainer_share_partial_masks():
    result = rollout()
    # The download is this document serialised as JSON, so compare what survives the round trip.
    document = json.loads(json.dumps(_contract(result.model_dump())))
    assert document == export_training_contract(result)
    assert document["trace_entries"] == to_trace_entries(result)
    assert document["turns"][0]["loss_mask"] == [0, 0, 1, 0]
    assert document["n_trainable_tokens"] == 1


def test_auxiliary_and_rejected_turns_cannot_gain_supervision_in_download():
    result = rollout()
    for i, changes in enumerate(
        (dict(role="auxiliary"), dict(discarded=True), dict(trainable=False)), 1
    ):
        result.turns.append(
            result.turns[0].model_copy(update={"turn": i, "node_id": str(i), **changes})
        )
    exported = export_training_contract(result)
    assert len(exported["trace_entries"]) == 1
    assert all(t["loss_mask"] == [0, 0, 0, 0] for t in exported["turns"][1:])


def test_eval_or_fatal_results_cannot_be_exported_as_training():
    result = rollout()
    result.rollout_type = "eval"
    assert _contract(result.model_dump()) is None
    with pytest.raises(ValueError, match="eval-only"):
        to_trace_entries(result)
    result.rollout_type = "train"
    result.findings = ["[FATAL] invalid token provenance"]
    with pytest.raises(ValueError, match="fatal"):
        _contract(result.model_dump())


def test_invalid_mask_cannot_be_hidden_by_the_download_path():
    result = rollout()
    result.turns[0].loss_mask = [0, 1]
    with pytest.raises(ValueError):
        _contract(result.model_dump())


def test_ui_validation_keeps_the_qualified_acp_profile(tmp_path, monkeypatch):
    import importlib
    from types import SimpleNamespace

    cells = []
    for provider in ["openai", "anthropic", "hf", "vllm"]:
        cell = {
            "harness": "acp",
            "provider": provider,
            "status": "eval_pass",
            "evidence": ["capture.json"],
            "configuration": {"acp_profile": "opencode-1.18.30"},
        }
        if provider == "vllm":
            cell.update(
                status="optimizer_pass",
                optimizer_validated=True,
                optimizer_evidence={
                    "matches_current_captures": True,
                    "result": "result.json",
                    "inputs": "inputs.json",
                    "scope": "diagnostic",
                    "model": "model",
                    "revision": "pinned",
                    "rows": 2,
                },
            )
        cells.append(cell)
    report = tmp_path / "matrix.json"
    report.write_text(json.dumps({"cells": cells}))
    monkeypatch.setenv("OPENENV_HARBOR_QUALIFICATION_REPORT", str(report))
    monkeypatch.setattr(
        importlib.import_module("openenv.core.harness.capture.validate_llm"),
        "validate_llm",
        lambda *args, **kwargs: SimpleNamespace(
            reachable=True,
            trainable=False,
            ok=True,
            capture_level="text",
            param_fixes=[],
            findings=[],
        ),
    )
    monkeypatch.setattr(
        importlib.import_module("openenv.harbor.capabilities"),
        "capabilities",
        lambda **kwargs: SimpleNamespace(
            available_sandboxes=["e2b"],
            sandboxes=[],
            harnesses=[
                SimpleNamespace(name="acp", dialect="openai"),
                SimpleNamespace(name="opencode", dialect="openai"),
            ],
        ),
    )
    monkeypatch.setattr("openenv.harbor.serving.HarborService.current", lambda: None)
    from openenv.harbor.ui import validate_endpoint

    engine = validate_endpoint(
        "https://provider.example/v1", "model", "", "openai", "eval"
    )
    assert engine["ok"]
    assert engine["allowed_harnesses"] == ["acp"]
    assert engine["harness_profiles"] == {"acp": "opencode-1.18.30"}
    assert "profile: opencode-1.18.30" in engine["choices"][0][0]
    assert engine["sandboxes"] == ["e2b"]
    experimental = validate_endpoint(
        "https://provider.example/v1", "model", "", "openai", "eval", True
    )
    assert experimental["allowed_harnesses"] == ["acp", "opencode"]


def test_live_view_follows_its_own_session_with_concurrent_rollouts(
    tmp_path, monkeypatch
):
    """The run view polls the session its rollout reported, never "the newest one" in the registry.

    Two visitors running at once must each watch their own rollout; discovering sessions by listing
    the registry would show one the other's trace.
    """
    import asyncio
    import threading
    import time
    from types import SimpleNamespace

    from openenv.core.harness.capture.sessions import SessionRegistry
    from openenv.harbor import ui_runs
    from openenv.harbor.ui import harbor_gradio_builder

    registry = SessionRegistry()
    release = threading.Event()
    service = SimpleNamespace(
        capture=SimpleNamespace(
            registry=registry,
            app=SimpleNamespace(
                state=SimpleNamespace(upstreams=SimpleNamespace(default=(None, "text")))
            ),
        ),
        capture_level="text",
        model="test-model",
        public_url="https://capture.example",
        llm_url="",
    )
    monkeypatch.setattr("openenv.harbor.serving.HarborService.current", lambda: service)
    monkeypatch.setattr(
        "openenv.harbor.tasks.HarborTaskProvider.task_dir", lambda *args: tmp_path
    )
    # Registry-wide discovery is not a safe way to identify the caller's run.
    monkeypatch.setattr(
        registry,
        "list_ids",
        lambda: pytest.fail("must not discover other users' sessions"),
    )

    async def run(**kwargs):
        registry.create("unrelated-private-run")
        session = registry.create("this-ui-run")
        kwargs["on_session_created"](session.session_id)
        for _ in range(1000):
            if release.is_set():
                break
            await asyncio.sleep(0.01)
        assert release.is_set(), "the live view never found its own session"
        return HarborRolloutResult(
            rollout_type="eval", capture_level="text", session_id=session.session_id
        )

    def transcript(session):
        assert session.session_id == "this-ui-run"
        release.set()
        return "this-ui-run trace"

    monkeypatch.setattr("openenv.harbor.rollout.run_rollout", run)
    monkeypatch.setattr(
        "openenv.harbor.ui._capabilities",
        lambda *a, **k: SimpleNamespace(
            available_sandboxes=["e2b"],
            sandboxes=[SimpleNamespace(name="e2b", available=True, detail="")],
            harnesses=[],
            datasets=[],
        ),
    )
    monkeypatch.setattr("openenv.harbor.ui_pages._transcript_html", transcript)
    monkeypatch.setenv("OPENENV_HARBOR_UI_SECRET", "test")
    manager = ui_runs.RunManager(store=None)
    monkeypatch.setattr(ui_runs, "_MANAGER", manager)
    app = harbor_gradio_builder(datasets=["test"])
    handlers = {getattr(b.fn, "__name__", ""): b.fn for b in app.fns.values()}

    started = handlers["on_run"](
        {"ok": True, "allowed_harnesses": ["opencode"], "purpose": "eval"},
        {"spec": "test", "index": 0},
        "visitor-a",
        SimpleNamespace(_data={"agent": "opencode", "sandbox": "e2b"}),
    )
    selection, frames, sig = started[2], [started[4]], started[5]
    deadline = time.time() + 20
    while time.time() < deadline:
        tick = handlers["on_tick"](selection, sig, "visitor-a")
        frames.append(tick[1])
        if manager.live(selection["id"]) is None:
            break
        time.sleep(0.02)
    assert any("this-ui-run trace" in str(frame) for frame in frames)
    assert all("unrelated-private-run" not in str(frame) for frame in frames)


def test_ui_refuses_datasets_it_does_not_serve(monkeypatch, tmp_path):
    """A dataset spec from the browser can be a local path; the UI must only open served datasets."""
    from openenv.harbor.ui import harbor_gradio_builder

    secret = tmp_path / "home" / "notes"
    secret.mkdir(parents=True)
    (secret / "passwords.txt").write_text("hunter2")
    monkeypatch.setenv("OPENENV_HARBOR_UI_SECRET", "test")
    app = harbor_gradio_builder(datasets=["org/served"])
    functions = {}
    for block in app.blocks.values():
        for fn in getattr(block, "server_fns", None) or []:
            functions[fn.__name__] = fn
    assert (
        "That dataset is not served here."
        in functions["hb_tasks"](str(tmp_path / "home"))["error"]
    )
    assert functions["hb_file"]([str(tmp_path / "home"), 0, "passwords.txt"]).get(
        "error"
    )
    assert "hunter2" not in json.dumps(
        functions["hb_file"]([str(tmp_path / "home"), 0, "passwords.txt"])
    )
    monkeypatch.setenv("OPENENV_HARBOR_UI_ADD_DATASETS", "1")
    assert functions["hb_add"](str(tmp_path / "home")).get("error")


# --- deployment settings -----------------------------------------------------


def _clear_ui_env(monkeypatch):
    for name in (
        "SPACE_ID",
        "OPENENV_HARBOR_RUN_HISTORY",
        "OPENENV_HARBOR_RUNS_DIR",
        "OPENENV_HARBOR_RUN_VISIBILITY",
        "OPENENV_HARBOR_UI_PRIVATE_URLS",
        "OPENENV_HARBOR_UI_LOCAL_TOKEN",
        "OPENENV_HARBOR_UI_ADD_DATASETS",
        "OPENENV_HARBOR_UI_SERVER_ENDPOINT",
        "OPENENV_HARBOR_UI_VISITOR_ENDPOINTS",
        "OPENENV_HARBOR_UI_HOST",
        "OPENENV_HARBOR_UI_MAX_RUNS_PER_VISITOR",
    ):
        monkeypatch.delenv(name, raising=False)


def test_settings_default_open_locally_and_closed_on_a_space(monkeypatch):
    from openenv.harbor import ui_settings

    _clear_ui_env(monkeypatch)
    monkeypatch.setenv("OPENENV_HARBOR_UI_HOST", "127.0.0.1")
    local = ui_settings.load()
    assert local.run_visibility == "all" and local.private_urls and local.local_token
    assert local.run_history and local.add_datasets and not local.exposed

    # Bound to every interface, the laptop is a server for the whole network.
    monkeypatch.setenv("OPENENV_HARBOR_UI_HOST", "0.0.0.0")
    lan = ui_settings.load()
    assert (
        lan.exposed
        and not lan.private_urls
        and not lan.local_token
        and not lan.add_datasets
    )

    monkeypatch.setenv("SPACE_ID", "org/space")
    space = ui_settings.load()
    assert space.run_visibility == "own" and space.max_runs_per_visitor == 2
    assert not space.private_urls and not space.run_history and not space.add_datasets
    # Anyone with a Space's URL is a visitor: the operator's endpoint is shared only on request.
    assert not space.server_endpoint and space.visitor_endpoints and space.rollouts
    assert lan.server_endpoint and local.server_endpoint
    # The token on a Space is the operator's: no variable offers it to visitors.
    monkeypatch.setenv("OPENENV_HARBOR_UI_LOCAL_TOKEN", "1")
    assert not ui_settings.load().local_token
    monkeypatch.setenv("OPENENV_HARBOR_RUN_VISIBILITY", "all")
    assert ui_settings.load().run_visibility == "all"


def test_a_visitor_url_may_not_reach_private_addresses():
    from openenv.harbor.ui_settings import url_problem

    assert url_problem("ftp://example.com/v1", private_ok=True)
    assert url_problem("http://127.0.0.1:8000/v1", private_ok=False)
    assert url_problem("http://10.0.0.5/v1", private_ok=False)
    assert url_problem("http://169.254.169.254/latest", private_ok=False)
    # IPv6 forms that carry a private IPv4 address inside a range `is_global` passes
    for embedded in (
        "[::127.0.0.1]",
        "[64:ff9b::a9fe:a9fe]",
        "[2002:7f00:1::]",
        "[::ffff:10.0.0.1]",
    ):
        assert url_problem(f"http://{embedded}/v1", private_ok=False), embedded
    assert url_problem("http://localhost:8000/v1", private_ok=True) is None


def test_a_redirect_to_a_private_address_is_refused():
    import urllib.error
    import urllib.request

    from openenv.harbor.ui_settings import _PublicRedirects

    request = urllib.request.Request("https://example.com/v1/models")
    with pytest.raises(urllib.error.HTTPError):
        _PublicRedirects().redirect_request(
            request, None, 302, "Found", {}, "http://169.254.169.254/latest/meta-data"
        )


def test_each_visitor_sees_only_their_own_runs(tmp_path):
    from openenv.harbor import ui_runs
    from openenv.harbor.ui_settings import owner_of

    store = ui_runs.RunStore(tmp_path)
    for rid, visitor in (("r-a", "visitor-a"), ("r-b", "visitor-b")):
        store.save(
            {
                "id": rid,
                "status": "done",
                "created": 1,
                "owner": owner_of(visitor),
                "result": None,
            }
        )
    manager = ui_runs.RunManager(store)
    assert [r["id"] for r in manager.list(owner_of("visitor-a"))] == ["r-a"]
    assert manager.get("r-b", owner_of("visitor-a")) is None
    assert manager.get("r-b", owner_of("visitor-b"))["id"] == "r-b"
    assert {r["id"] for r in manager.list(None)} == {"r-a", "r-b"}, "`all` visibility"
    assert manager.list("") == [], "no visitor id sees nothing"
    assert "visitor-a" not in (tmp_path / "r-a.json").read_text(), (
        "only a digest is stored"
    )


def test_hugging_face_connect_routes_to_a_provider_and_needs_a_token(monkeypatch):
    from openenv.harbor import ui, ui_settings

    _clear_ui_env(monkeypatch)
    seen = {}

    def fake_validate(url, model, api_key, **kwargs):
        seen.update(url=url, model=model, api_key=api_key, **kwargs)
        return {"ok": True, "model": model, "api_key": api_key}

    monkeypatch.setattr(ui, "validate_endpoint", fake_validate)
    settings = ui_settings.load()
    refused = ui.connect_endpoint(
        {"source": "hf", "model": "org/m", "local_token": False}, settings=settings
    )
    assert not refused["ok"] and "token" in refused["reason"]
    engine = ui.connect_endpoint(
        {"source": "hf", "model": "org/m", "route": "cerebras", "api_key": "hf_secret"},
        settings=settings,
    )
    assert (
        seen["model"] == "org/m:cerebras"
        and seen["provider"] == "hf"
        and seen["purpose"] == "eval"
    )
    assert "api_key" not in engine["custom"], "the form kept for the card drops the key"


def test_the_card_never_carries_a_key(monkeypatch):
    from types import SimpleNamespace

    from openenv.harbor import ui

    monkeypatch.setattr("openenv.harbor.serving.HarborService.current", lambda: None)
    caps = SimpleNamespace(harnesses=[], sandboxes=[])
    engine = {
        "ok": True,
        "model": "m",
        "api_key": "sk-very-secret",
        "source": "url",
        "custom": {"url": "https://x"},
    }
    assert "sk-very-secret" not in json.dumps(ui._card(engine, None, caps), default=str)


def test_download_grants_cannot_be_guessed_or_reused_for_another_run():
    from openenv.harbor import ui_pages

    token = ui_pages.grant("run-1")
    assert ui_pages.granted(token) == "run-1"
    assert ui_pages.grant("run-1") == token, "a redrawn page reuses its grant"
    assert ui_pages.granted("run-1") is None and ui_pages.granted("") is None


def test_only_public_hub_datasets_can_be_added(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from openenv.harbor import ui

    monkeypatch.chdir(tmp_path)
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    for spec in ("src/..", "src/pkg", ".git/x", "org/.hidden", "not-an-id"):
        assert ui._hub_problem(spec), spec
    info = {
        "org/open": SimpleNamespace(private=False, gated=False),
        "org/secret": SimpleNamespace(private=True, gated=False),
        "org/gated": SimpleNamespace(private=False, gated="auto"),
    }

    class Api:
        def __init__(self, token=None):
            assert token is False, (
                "adds are looked up anonymously, never with the server's token"
            )

        def dataset_info(self, spec):
            if spec not in info:
                raise FileNotFoundError(spec)
            return info[spec]

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    assert ui._hub_problem("org/open") is None
    assert "private or gated" in ui._hub_problem("org/secret")
    assert "private or gated" in ui._hub_problem("org/gated")
    assert "not a public dataset" in ui._hub_problem("org/missing")


def test_tasks_that_read_the_server_environment_are_found(monkeypatch, tmp_path):
    from openenv.harbor import ui_data

    task = tmp_path / "t"
    (task / "environment").mkdir(parents=True)
    (task / "task.toml").write_text('[verifier.env]\nX = "literal"\n')
    monkeypatch.setattr(
        "openenv.harbor.tasks.HarborTaskProvider.task_dir", lambda *a: task
    )
    assert not ui_data.reads_environment("org/x", 0)
    (task / "environment" / "docker-compose.yaml").write_text(
        "environment:\n  - T=$HF_TOKEN\n"
    )
    assert ui_data.reads_environment("org/x", 0)
    (task / "environment" / "docker-compose.yaml").unlink()
    (task / "task.toml").write_text('[verifier.env]\nX = "${HF_TOKEN}"\n')
    assert ui_data.reads_environment("org/x", 0)


def test_a_compose_file_that_reads_the_host_is_found(monkeypatch, tmp_path):
    """Compose can pull the host's files into a container, which a local backend runs on this machine."""
    from openenv.harbor import ui_data

    task = tmp_path / "t"
    (task / "environment" / "sub").mkdir(parents=True)
    (task / "task.toml").write_text("[verifier]\ntimeout_sec = 60\n")
    monkeypatch.setattr(
        "openenv.harbor.tasks.HarborTaskProvider.task_dir", lambda *a: task
    )
    included = (
        task / "environment" / "sub" / "base.yaml"
    )  # one compose file can include another
    for text in (
        "services:\n  a:\n    env_file: /home/me/.env\n",
        "include:\n  - ../../other.yaml\n",
        "services:\n  a:\n    extends:\n      file: /etc/x.yaml\n",
        "services:\n  a:\n    volumes:\n      - /:/host\n",
        "services:\n  a:\n    volumes:\n      - ~/.cache:/c\n",
        "services:\n  a:\n    volumes:\n      - type: bind\n        source: ../..\n        target: /x\n",
    ):
        included.write_text(text)
        assert ui_data.reads_environment("org/x", 0), text
    included.write_text("services:\n  a:\n    volumes:\n      - ./data:/data\n")
    assert not ui_data.reads_environment("org/x", 0), "the task's own files are fine"


def test_the_file_tree_lists_the_top_level_first_and_stops_at_the_cap(
    monkeypatch, tmp_path
):
    from openenv.harbor import ui_data

    for rel in (
        "task.toml",
        "instruction.md",
        ".secret",
        "tests/test.sh",
        "environment/Dockerfile",
    ):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("not this task's")
    (tmp_path / "tests" / "leak.txt").symlink_to(outside)
    (tmp_path / "tests" / "same.sh").symlink_to(tmp_path / "tests" / "test.sh")
    files, cut = ui_data.file_tree(tmp_path)
    assert [f["path"] for f in files] == [
        "instruction.md",
        "task.toml",
        "environment/Dockerfile",
        "tests/same.sh",
        "tests/test.sh",
    ], "a symlink within the task is listed, one out of it is not"
    assert not cut
    monkeypatch.setattr(ui_data, "MAX_TREE_FILES", 2)
    files, cut = ui_data.file_tree(tmp_path)
    assert len(files) == 2 and cut


def test_a_dataset_of_unknown_size_is_measured_or_refused(monkeypatch):
    import time
    from types import SimpleNamespace

    from openenv.harbor import ui_data

    class Api:
        def __init__(self, token=None):
            pass

        def list_repo_tree(
            self, spec, repo_type=None, path_in_repo=None, recursive=None
        ):
            return iter([SimpleNamespace(size=3_000_000_000)] * 3)

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    assert ui_data._tasks_bytes("org/big", cap=5_000_000_000) == 6_000_000_000, (
        "stops once past the cap"
    )
    monkeypatch.setattr(
        ui_data, "hub_summary", lambda spec: {"tasks": 3, "bytes": None}
    )
    monkeypatch.setattr(ui_data, "_tasks_bytes", lambda spec, cap: None)
    ui_data._JOBS.pop("org/unsized", None)
    ui_data.start_add("org/unsized", on_added=lambda target: None)
    for _ in range(100):
        job = ui_data.add_status("org/unsized")
        if job["state"] == "error":
            break
        time.sleep(0.02)
    assert job["state"] == "error" and "Couldn't tell how big" in job["error"]


def test_harbor_expands_only_braced_variables_in_task_toml(monkeypatch):
    """Why `reads_environment` looks for `${` alone in task.toml: Harbor leaves a bare `$VAR` as is.

    Compose is checked for bare `$VAR` too, since Docker Compose expands both forms.
    """
    env = pytest.importorskip("harbor.utils.env")
    monkeypatch.setenv("HF_TOKEN", "secret")
    resolved = env.resolve_env_vars(
        {"bare": "$HF_TOKEN", "braced": "${HF_TOKEN}", "inline": "a $HF_TOKEN b"}
    )
    assert resolved == {
        "bare": "$HF_TOKEN",
        "braced": "secret",
        "inline": "a $HF_TOKEN b",
    }


def test_one_visitor_cannot_hold_every_slot(monkeypatch, tmp_path):
    import asyncio
    import threading
    from types import SimpleNamespace

    from openenv.harbor import ui_runs

    release = threading.Event()

    async def slow(*args, **kwargs):
        while not release.is_set():
            await asyncio.sleep(0.01)
        raise RuntimeError("stopped")

    monkeypatch.setattr(
        "openenv.harbor.tasks.HarborTaskProvider.task_dir", lambda *a: tmp_path
    )
    monkeypatch.setattr(ui_runs, "_rollout", slow)
    manager = ui_runs.RunManager(store=None, max_live=4)
    kwargs = {
        "engine": {"server_default": True},
        "spec": "s",
        "index": 0,
        "harness": "h",
        "sandbox": "e2b",
        "service": SimpleNamespace(),
    }
    try:
        manager.start(**kwargs, owner="a", per_owner=1)
        with pytest.raises(RuntimeError, match="already have 1"):
            manager.start(**kwargs, owner="a", per_owner=1)
        manager.start(**kwargs, owner="b", per_owner=1)
        # signed in, the cap follows the account: a fresh browser id is not a fresh allowance
        manager.start(**kwargs, owner="c", per_owner=1, quota="hf:alice")
        with pytest.raises(RuntimeError, match="already have 1"):
            manager.start(**kwargs, owner="d", per_owner=1, quota="hf:alice")
    finally:
        release.set()


def test_markdown_from_a_model_loads_no_images():
    from openenv.harbor.ui_trace import _markdown

    out = _markdown("look ![x](https://tracker.example/p.png)")
    assert "<img" not in out


def test_add_progress_counts_files_not_bytes():
    from openenv.harbor.ui_data import _progress

    job = {}
    Progress = _progress(job)
    files = Progress(total=10, desc="Fetching 10 files")
    files.update(3)
    big = Progress(total=5_000_000, unit="B", unit_scale=True)
    big.update(4_000_000)
    assert job == {"total": 10, "done": 3}


def test_run_history_skips_files_that_are_not_runs(tmp_path):
    """Anything else in the folder (a list, a broken file, the folder's own dotfiles) is skipped."""
    from openenv.harbor import ui_runs

    (tmp_path / "list.json").write_text('["org/dataset"]')
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / ".added-datasets.json").write_text('["org/dataset"]')
    (tmp_path / "r1.json").write_text('{"id": "r1", "status": "done", "created": 1}')
    assert [r["id"] for r in ui_runs.RunStore(tmp_path).list()] == ["r1"]


# --- the Space: bucket, datasets added from the page, sign-in ------------------------------------


def _bucket_settings(tmp_path):
    from types import SimpleNamespace

    mount = tmp_path / "data"
    mount.mkdir()
    return SimpleNamespace(bucket="org/space", bucket_mount=mount, hf_login=True)


def test_page_adds_go_into_the_bucket_and_the_bucket_is_the_list(monkeypatch, tmp_path):
    from openenv.harbor import ui_data

    settings = _bucket_settings(tmp_path)
    copies = []

    class Api:
        def __init__(self, token=None):
            pass

        def copy_files(self, src, dst):
            copies.append((src, dst))
            task = settings.bucket_mount / "org__suite" / "tasks" / "t1"
            task.mkdir(parents=True)
            (task / "task.toml").write_text(
                ""
            )  # what the mount shows once the copy lands

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    job = {}
    target = ui_data._copy_to_bucket("org/suite", settings, job, expected=1)
    assert copies == [
        ("hf://datasets/org/suite/", "hf://buckets/org/space/org__suite/")
    ]
    assert target == str(settings.bucket_mount / "org__suite")
    assert ui_data.added_spec("org/suite", settings) == target
    assert ui_data.hub_id(target) == "org/suite"
    assert ui_data.added_in_bucket(settings, served=[]) == [target]
    assert ui_data.added_in_bucket(settings, served=[target]) == [], (
        "a served one is not added"
    )


def test_removing_deletes_only_the_dataset_folder(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from openenv.harbor import tasks, ui_data

    # on the bucket: every file under the dataset's folder, and nothing else
    settings = _bucket_settings(tmp_path)
    folder = settings.bucket_mount / "org__suite"
    folder.mkdir()
    deleted = []

    class Api:
        def __init__(self, token=None):
            pass

        def list_bucket_tree(self, bucket, prefix=None, recursive=None):
            assert bucket == "org/space" and prefix == "org__suite/"
            return [SimpleNamespace(path="org__suite/tasks/t1/task.toml", type="file")]

        def batch_bucket_files(self, bucket, delete=None):
            deleted.extend(delete)

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    ui_data.remove_added(str(folder), settings)
    assert deleted == ["org__suite/tasks/t1/task.toml"]
    with pytest.raises(ValueError):
        ui_data.remove_added(
            str(tmp_path), settings
        )  # not a dataset folder on the mount

    # locally: the download in the dataset cache, and nothing else
    local = SimpleNamespace(bucket=None, bucket_mount=None)
    monkeypatch.setattr(tasks, "_DATASET_ROOT", tmp_path / "cache")
    (tmp_path / "cache" / "org__suite" / "tasks").mkdir(parents=True)
    ui_data.remove_added("org/suite", local)
    assert not (tmp_path / "cache" / "org__suite").exists()
    # `/` becomes `__`, so a spec can only name a folder inside the cache, and `..` names none.
    with pytest.raises(ValueError):
        ui_data.remove_added("..", local)


def test_signing_in_is_an_account_for_inference_providers(monkeypatch):
    from types import SimpleNamespace

    from openenv.harbor import ui

    seen = {}
    monkeypatch.setattr(
        ui,
        "validate_endpoint",
        lambda url, model, key, **kw: seen.update(key=key) or {"ok": True},
    )
    settings = SimpleNamespace(
        visitor_endpoints=True, hf_login=True, local_token=False, private_urls=False
    )
    form = {"source": "hf", "model": "org/m", "use_account": True}
    ui.connect_endpoint(form, settings=settings, account_token="hf_oauth_token")
    assert seen["key"] == "hf_oauth_token"
    expired = ui.connect_endpoint(form, settings=settings, account_token=None)
    assert not expired["ok"] and "expired" in expired["reason"]
    remembered = ui.connect_endpoint(
        form, settings=settings, account_token="hf_oauth_token"
    )
    assert "hf_oauth_token" not in json.dumps(remembered.get("custom"))


def test_a_cross_site_request_is_refused_and_others_pass(monkeypatch):
    import asyncio

    from openenv.harbor.serving import SameOrigin

    calls = []

    async def app(scope, receive, send):
        calls.append(scope["path"])

    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request"}

    def request(method, path="/web/gradio_api/queue/join", **headers):
        return {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [
                (k.replace("_", "-").encode(), v.encode()) for k, v in headers.items()
            ],
        }

    guard = SameOrigin(app)
    asyncio.run(
        guard(
            request("POST", host="me.hf.space", origin="https://evil.example"),
            receive,
            send,
        )
    )
    asyncio.run(
        guard(
            request("POST", host="me.hf.space", sec_fetch_site="cross-site"),
            receive,
            send,
        )
    )
    assert calls == [] and sent[0]["status"] == 403
    asyncio.run(
        guard(
            request("POST", host="me.hf.space", origin="https://me.hf.space"),
            receive,
            send,
        )
    )
    asyncio.run(
        guard(request("POST", host="me.hf.space"), receive, send)
    )  # a server, no browser headers
    asyncio.run(
        guard(
            request("GET", host="me.hf.space", origin="https://evil.example"),
            receive,
            send,
        )
    )
    assert len(calls) == 3
    # the Task API and MCP are not the UI: browser tools call them from other origins on purpose
    asyncio.run(
        guard(
            request(
                "POST", path="/mcp", host="localhost:8000", sec_fetch_site="cross-site"
            ),
            receive,
            send,
        )
    )
    # a client vouching for its own origin through X-Forwarded-Host is not believed
    asyncio.run(
        guard(
            request(
                "POST",
                host="10.0.0.5:8000",
                x_forwarded_host="evil.example",
                origin="https://evil.example",
            ),
            receive,
            send,
        )
    )
    assert len(calls) == 4
    # behind a proxy that rewrites Host, the public host is listed by the operator instead
    monkeypatch.setenv("OPENENV_HARBOR_UI_HOSTS", "harbor.example.org")
    asyncio.run(
        guard(
            request("POST", host="10.0.0.5:8000", origin="https://harbor.example.org"),
            receive,
            send,
        )
    )
    assert len(calls) == 5
    asyncio.run(
        guard(
            request(
                "POST",
                host="10.0.0.5:8000",
                origin="https://harbor.example.org",
                sec_fetch_site="cross-site",
            ),
            receive,
            send,
        )
    )
    refused = [m["status"] for m in sent if m["type"] == "http.response.start"]
    assert len(calls) == 5 and refused == [403, 403, 403, 403]


def test_the_cross_site_guard_is_on_without_sign_in(monkeypatch):
    """Sign-in is not the only reason to refuse other sites: a page could start rollouts either way."""
    from fastapi import FastAPI
    from openenv.core.env_server import http_server
    from openenv.harbor import serving

    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.delenv("OAUTH_CLIENT_ID", raising=False)
    monkeypatch.setattr(http_server, "create_app", lambda *a, **k: FastAPI())
    monkeypatch.setattr(serving.HarborService, "current", classmethod(lambda cls: None))
    app = serving.build_app(datasets=[])
    assert not serving._attach_hf_login(FastAPI())
    assert serving.SameOrigin in [m.cls for m in app.user_middleware]


def test_push_keeps_buckets_private_unless_told(monkeypatch, capsys):
    from types import SimpleNamespace

    from openenv.cli.commands import harbor

    made, changed, state = [], [], {}

    class Api:
        def bucket_info(self, bucket):
            if bucket not in state:
                raise FileNotFoundError(bucket)
            return SimpleNamespace(private=state[bucket])

        def create_bucket(self, bucket, private=None, exist_ok=False):
            made.append((bucket, private))
            state[bucket] = private

        def update_bucket_settings(self, bucket, private):
            changed.append((bucket, private))
            state[bucket] = private

        def list_bucket_tree(self, bucket):
            return []

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    harbor._fill_bucket("org/new", [])
    harbor._fill_bucket("org/open", [], public=True)
    assert made == [("org/new", True), ("org/open", False)]
    harbor._fill_bucket(
        "org/open", []
    )  # existing and public: left as it is, with a warning
    assert changed == [] and "is PUBLIC" in capsys.readouterr().out
    harbor._fill_bucket("org/open", [], public=False)
    assert changed == [("org/open", True)]


def test_push_refuses_a_space_whose_visitors_would_have_no_model():
    """On a Space the endpoint is shared only when asked, so turning visitors' own models off needs it."""
    import re

    from openenv.cli.commands.harbor import app
    from typer.testing import CliRunner

    args = ["push", "--repo-id", "org/space", "--llm-url", "http://engine:8000/v1"]
    refused = CliRunner().invoke(app, [*args, "--no-visitor-endpoints"])
    # Rich colours and boxes the error where it sees a colour terminal (GitHub Actions): read the text
    plain = re.sub(r"\x1b\[[0-9;]*m", "", refused.output)
    text = " ".join(re.sub(r"[│╭╮╰╯─]", " ", plain).split())
    assert refused.exit_code == 2 and "needs --share-endpoint" in text


def test_push_turns_on_sign_in_in_the_space_readme(tmp_path):
    from openenv.cli.commands import harbor

    readme = tmp_path / "README.md"
    readme.write_text("---\ntitle: Harbor\nsdk: docker\n---\n\n# Harbor\n")
    harbor._enable_hf_login(readme)
    text = readme.read_text()
    front = text.split("---")[1]
    assert (
        "hf_oauth: true" in front
        and "- inference-api" in front
        and "title: Harbor" in front
    )
    assert text.endswith("# Harbor\n")
    harbor._enable_hf_login(readme)
    assert readme.read_text() == text, "idempotent"


def test_push_turns_on_sign_in_for_the_real_env_readme(tmp_path):
    """The README documents `hf_oauth: true` in its text; only its front matter may count."""
    from pathlib import Path

    from openenv.cli.commands import harbor

    readme = tmp_path / "README.md"
    source = Path(__file__).parents[2] / "envs" / "harbor_env" / "README.md"
    readme.write_text(source.read_text())
    harbor._enable_hf_login(readme)
    assert "hf_oauth: true" in readme.read_text().split("\n---", 1)[0]


def test_sign_in_on_a_docker_space_is_the_real_one(monkeypatch):
    """Docker Spaces do not set SYSTEM=spaces; Gradio would then fake the login as the operator."""
    pytest.importorskip("itsdangerous")  # Gradio's OAuth libraries, in the Space image
    from fastapi import FastAPI
    from openenv.harbor import serving

    monkeypatch.setenv("SPACE_ID", "org/space")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "client")
    monkeypatch.setenv(
        "SYSTEM", ""
    )  # recorded, so the value the code sets is undone afterwards
    monkeypatch.delenv("SYSTEM")
    added = []
    monkeypatch.setattr(
        "gradio.oauth._add_oauth_routes", lambda app: added.append("real")
    )
    monkeypatch.setattr(
        "gradio.oauth._add_mocked_oauth_routes", lambda app: added.append("mocked")
    )
    monkeypatch.setattr(FastAPI, "add_middleware", lambda self, *a, **k: None)
    assert serving._attach_hf_login(FastAPI())
    assert added == ["real"]
