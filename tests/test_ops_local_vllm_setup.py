"""The local vLLM setup must be runnable on a machine that has never run it.

Four bugs shipped in `ops/local-vllm/` in a row, and every one of them was
invisible on the development box, because that box already had the weights
cached, a filled-in `.env`, and no reason to re-run `pull.sh`:

1. `env.example` shipped `VLLM_IMAGE=<FILL>`. In bash `<` opens a redirection,
   so `source .env` died with a syntax error on line 7 -- and `pull.sh` sources
   `.env` on line 12, then fills `VLLM_IMAGE` on line 99. The two values it
   exists to write were the two that stopped it starting.
2. The image candidate list began with the one image the guide's own table marks
   as "No -- the KeyError", so a fresh setup spent 20+ GB proving that.
3. `pull.sh` called `hf`, which nothing declared. `hf: command not found`.
4. `requirements.txt` asked for `huggingface_hub[cli]`; no such extra exists on
   1.x, so pip warned and ignored it.

These tests do the whole setup path except the parts that cost a GPU or a
download: no `docker pull`, no weights, no server. They need no network unless
noted.
"""

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
OPS = REPO / "ops" / "local-vllm"
ENV_EXAMPLE = OPS / "env.example"
SCRIPTS = [OPS / "pull.sh", OPS / "up.sh"]

# Tools the scripts invoke that a fresh machine may genuinely lack. Anything
# added here must also be named by the scripts' own preflight.
REQUIRED_TOOLS = ["docker", "curl"]


def _run(args, cwd, env=None, timeout=60):
    e = dict(os.environ)
    e.pop("HF_TOKEN", None)
    if env:
        e.update(env)
    return subprocess.run(args, cwd=cwd, env=e, capture_output=True,
                          text=True, timeout=timeout)


@pytest.fixture
def setup_dir(tmp_path):
    """A fresh checkout's worth of ops/local-vllm, with `.env` already copied."""
    for f in ("pull.sh", "up.sh", "env.example"):
        shutil.copy2(OPS / f, tmp_path / f)
        (tmp_path / f).chmod(0o755)
    shutil.copy2(OPS / "env.example", tmp_path / ".env")
    return tmp_path


@pytest.fixture
def tool_path(tmp_path):
    """PATH with the external tools stubbed present.

    The tool gate runs before the `.env` gates, by design. To exercise the
    `.env` gates at all, the tools have to look installed -- and stubbing them
    keeps these tests working on a machine (or a CI runner) that has no docker
    and no huggingface_hub.
    """
    d = tmp_path / "toolbin"
    d.mkdir()
    for tool in ("docker", "curl", "hf", "huggingface-cli"):
        p = d / tool
        p.write_text("#!/bin/sh\nexit 0\n")
        p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return f"{d}:{os.environ.get('PATH', '')}"


# --------------------------------------------------------------------------
# 1. env.example must be loadable. This is bug 1.
# --------------------------------------------------------------------------

def test_env_example_is_valid_shell():
    r = _run(["bash", "-c", f"set -e; source {ENV_EXAMPLE.name}"], cwd=OPS)
    assert r.returncode == 0, (
        f"env.example cannot be sourced, so pull.sh dies before it can do "
        f"anything:\n{r.stderr}"
    )


def test_env_example_has_no_angle_bracket_placeholders():
    """`NAME=<FILL>` parses as a redirection, not as a placeholder."""
    offenders = [
        f"{n}: {line.rstrip()}"
        for n, line in enumerate(ENV_EXAMPLE.read_text().splitlines(), 1)
        if not line.lstrip().startswith("#") and re.search(r"=\s*<", line)
    ]
    assert not offenders, (
        "A placeholder like <FILL> is a bash syntax error. Ship the value empty "
        "instead:\n  " + "\n  ".join(offenders)
    )


def test_every_env_example_line_is_name_equals_value():
    bad = [
        f"{n}: {line.rstrip()}"
        for n, line in enumerate(ENV_EXAMPLE.read_text().splitlines(), 1)
        if line.strip()
        and not line.lstrip().startswith("#")
        and not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", line)
    ]
    assert not bad, "Not NAME=value:\n  " + "\n  ".join(bad)


def test_the_values_pull_sh_writes_ship_empty():
    """`pull.sh` fills these. Shipping them non-empty makes up.sh accept junk."""
    text = ENV_EXAMPLE.read_text()
    for var in ("VLLM_IMAGE", "MODEL_REVISION", "MEDIA_ROOT"):
        m = re.search(rf"^{var}=(.*)$", text, re.M)
        assert m, f"{var} is missing from env.example"
        assert m.group(1).strip() == "", (
            f"{var} must ship empty; got {m.group(1)!r}"
        )


# --------------------------------------------------------------------------
# 2. The scripts must parse, and fail with a message rather than a stack trace.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_parses(script):
    r = _run(["bash", "-n", str(script)], cwd=OPS)
    assert r.returncode == 0, r.stderr


def test_unedited_env_stops_on_media_root_not_on_a_syntax_error(setup_dir, tool_path):
    """The fresh-machine path: copy env.example, run pull.sh, edit nothing."""
    r = _run(["./pull.sh"], cwd=setup_dir, env={"PATH": tool_path})
    assert r.returncode == 1, f"expected a clean refusal, got {r.returncode}"
    assert "MEDIA_ROOT" in r.stderr
    assert "syntax error" not in r.stderr.lower()


def test_a_reintroduced_placeholder_is_explained(setup_dir, tool_path):
    """Bug 1's symptom must now come with the fix attached."""
    env = setup_dir / ".env"
    env.write_text(env.read_text().replace("MEDIA_ROOT=", "MEDIA_ROOT=<FILL>"))
    r = _run(["./pull.sh"], cwd=setup_dir, env={"PATH": tool_path})
    assert r.returncode == 1
    assert "NAME=value" in r.stderr, r.stderr
    assert "<FILL>" in r.stderr, "the message should show what is wrong"


def test_an_unquoted_space_is_explained(setup_dir, tool_path):
    """Syntactically valid, so `bash -n` misses it; it fails only when run."""
    env = setup_dir / ".env"
    env.write_text(env.read_text().replace(
        "MEDIA_ROOT=", "MEDIA_ROOT=/home/me/my photos"))
    r = _run(["./pull.sh"], cwd=setup_dir, env={"PATH": tool_path})
    assert r.returncode == 1
    assert "NAME=value" in r.stderr, r.stderr


def test_media_root_must_exist(setup_dir, tool_path):
    env = setup_dir / ".env"
    env.write_text(env.read_text().replace(
        "MEDIA_ROOT=", "MEDIA_ROOT=/nonexistent/path"))
    r = _run(["./pull.sh"], cwd=setup_dir, env={"PATH": tool_path})
    assert r.returncode == 1
    assert "not a directory" in r.stderr


def test_up_sh_refuses_before_pull_sh_has_run(setup_dir, tmp_path, tool_path):
    env = setup_dir / ".env"
    env.write_text(env.read_text().replace("MEDIA_ROOT=", f"MEDIA_ROOT={tmp_path}"))
    r = _run(["./up.sh"], cwd=setup_dir, env={"PATH": tool_path})
    assert r.returncode == 1
    assert "VLLM_IMAGE" in r.stderr and "pull.sh" in r.stderr


# --------------------------------------------------------------------------
# 3. Declared tools. This is bug 3.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
@pytest.mark.parametrize("tool", REQUIRED_TOOLS)
def test_script_checks_for_the_tools_it_calls(script, tool):
    text = script.read_text()
    if not re.search(rf"(^|[\s;|&(]){tool}\s", text, re.M):
        pytest.skip(f"{script.name} does not call {tool}")
    assert f"command -v {tool}" in text, (
        f"{script.name} calls {tool} but never checks for it, so a machine "
        f"without it gets a bare 'command not found'."
    )


def test_pull_sh_checks_for_the_huggingface_cli():
    """It is spelled `hf` now and `huggingface-cli` before; accept either."""
    text = (OPS / "pull.sh").read_text()
    assert "command -v hf" in text
    assert "command -v huggingface-cli" in text, (
        "older huggingface_hub installs only the deprecated name"
    )
    assert '"$HF_CLI" download' in text, (
        "the download must use whichever CLI was found, not a hardcoded name"
    )


def test_missing_hf_is_named_and_stops_before_any_download(setup_dir, tmp_path):
    """Bug 3, end to end: hide the CLI and check what the user is told."""
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    for tool in ("bash", "cat", "sed", "grep", "curl", "docker", "printf",
                 "ls", "head", "env", "dirname", "sleep", "tr", "sort"):
        src = shutil.which(tool)
        if src:
            (fakebin / tool).symlink_to(src)
    env = setup_dir / ".env"
    env.write_text(env.read_text().replace("MEDIA_ROOT=", f"MEDIA_ROOT={tmp_path}"))

    r = _run(["bash", "./pull.sh"], cwd=setup_dir, env={"PATH": str(fakebin)})
    assert r.returncode == 1
    assert "hf" in r.stderr and "huggingface_hub" in r.stderr, r.stderr
    assert "requirements.txt" in r.stderr, "say how to get it"


# --------------------------------------------------------------------------
# 4. requirements.txt. This is bug 4.
# --------------------------------------------------------------------------

REQ_FILES = [REPO / "requirements.txt", OPS / "requirements.txt"]


@pytest.mark.parametrize("req", REQ_FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_requirements_parse_as_pep508(req):
    Requirement = pytest.importorskip("packaging.requirements").Requirement
    for n, line in enumerate(req.read_text().splitlines(), 1):
        line = line.split("#")[0].strip()
        if line:
            Requirement(line)  # raises on malformed


@pytest.mark.parametrize("req", REQ_FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_requested_extras_actually_exist(req):
    """`huggingface_hub[cli]` warned and was ignored; nothing caught it."""
    from importlib.metadata import PackageNotFoundError, distribution

    Requirement = pytest.importorskip("packaging.requirements").Requirement

    checked = 0
    for line in req.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        r = Requirement(line)
        if not r.extras:
            continue
        try:
            dist = distribution(r.name)
        except PackageNotFoundError:
            continue  # not installed here; CI installs it and will check
        declared = set(dist.metadata.get_all("Provides-Extra") or [])
        missing = sorted(r.extras - declared)
        assert not missing, (
            f"{r.name} {dist.version} declares no extra {missing}. "
            f"It has: {sorted(declared) or 'none'}. pip only warns, so this "
            f"silently does nothing."
        )
        checked += 1
    assert checked >= 0


def test_ops_requirements_declare_the_hf_cli():
    text = (OPS / "requirements.txt").read_text()
    assert re.search(r"^huggingface_hub", text, re.M), (
        "pull.sh needs the hf CLI; it must be declared here"
    )


# --------------------------------------------------------------------------
# 5. The image candidate order. This is bug 2.
# --------------------------------------------------------------------------

def test_known_bad_image_is_not_tried_first():
    """The guide's table is the source of truth for which images work."""
    guide = (OPS / "README_vllm.md").read_text()
    bad = {
        m.group(1)
        for m in re.finditer(r"^\|\s*`([^`]+)`\s*\|[^|]*\|\s*\*\*No\*\*", guide, re.M)
    }
    if not bad:
        pytest.skip("no image is marked 'No' in the guide's table")

    block = re.search(r"^CANDIDATES=\((.*?)^\)", (OPS / "pull.sh").read_text(),
                      re.M | re.S)
    assert block, "CANDIDATES list not found in pull.sh"
    order = re.findall(r'"([^"]+)"', block.group(1))
    assert order, "no candidates parsed"
    assert order[0] not in bad, (
        f"pull.sh tries {order[0]} first, but README_vllm.md marks it as not "
        f"loading this checkpoint. That is a 20+ GB download to prove a known "
        f"failure."
    )


# --------------------------------------------------------------------------
# 6. Intra-document links, which two renderers slug differently.
# --------------------------------------------------------------------------

DOCS = [REPO / "README.md", REPO / "user_manual.md", OPS / "README_vllm.md"]


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_intra_document_links_resolve(doc):
    text = doc.read_text()

    def slugs(heading):
        a = re.sub(r"[^\w\s-]", "", heading.strip().lower())
        # GitHub turns each space into a hyphen and does not collapse runs;
        # other renderers collapse. A link must work under both.
        return {a.replace(" ", "-"), re.sub(r"\s+", "-", a)}

    ids, ambiguous = set(), set()
    for m in re.finditer(r"^(#{1,6})\s+(.*)$", text, re.M):
        variants = slugs(m.group(2))
        ids |= variants
        if len(variants) > 1:
            ambiguous |= variants

    problems = []
    for m in re.finditer(r"\]\(#([^)]+)\)", text):
        target = m.group(1)
        if target not in ids:
            problems.append(f"broken: #{target}")
        elif target in ambiguous:
            problems.append(
                f"renderer-dependent: #{target} (its heading has a spaced dash; "
                f"use a colon instead)"
            )
    assert not problems, f"{doc.name}:\n  " + "\n  ".join(problems)
