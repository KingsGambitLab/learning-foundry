"""Deterministic post-author normalization of runtime-protocol files.

The LLM authoring layer produces a `Dockerfile` and `.coursegen/runtime/*.sh`.
This module rewrites those bytes (after authoring, before they reach disk)
to apply a language-agnostic performance guardrail: BuildKit cache mounts on
the install RUN step, keyed off the creator-declared `package_manager`. The
mounts survive layer-cache invalidations so the next rebuild reuses pip /
npm / cargo / go / maven / etc. wheels.

Earlier versions also stripped `COPY . <dst>` lines, assuming a runtime
bind-mount at `/workspace` made the bake-in dead weight. A narrow heuristic
tried to detect when the strip would break the build (`chmod`, `./script`,
`sh|bash|python|cat <relpath>`), but it missed common ecosystem-standard
cases: `RUN pip install -r requirements.txt`, `RUN npm install`, `RUN mvn
install`, `RUN cargo build`, `RUN go build`, and any multiline `RUN` with
backslash continuations. Missing those produced builds that fail because
`requirements.txt` / `package.json` / `pom.xml` / `go.mod` weren't present
in the image. The strip is removed; cache-mount injection is the remaining
optimization, which already delivers the bulk of the rebuild speedup and
composes correctly with `COPY .`.

For Python + CUDA-pulling deps (`torch`, `sentence-transformers`,
`transformers`, …) we also prepend a CPU-only `torch` install in `install.sh`
so the sandbox doesn't pull ~2 GB of NVIDIA wheels it can't use.
"""
from __future__ import annotations


_CACHE_MOUNT_TARGETS: dict[str, list[str]] = {
    "pip": ["/root/.cache/pip"],
    "pip3": ["/root/.cache/pip"],
    "poetry": ["/root/.cache/pip", "/root/.cache/pypoetry"],
    "pdm": ["/root/.cache/pip", "/root/.cache/pdm"],
    "uv": ["/root/.cache/uv"],
    "npm": ["/root/.npm"],
    "pnpm": ["/root/.local/share/pnpm/store"],
    "yarn": ["/usr/local/share/.cache/yarn"],
    "bundler": ["/usr/local/bundle/cache"],
    "gem": ["/usr/local/bundle/cache"],
    "bundle": ["/usr/local/bundle/cache"],
    "gomod": ["/root/.cache/go-build", "/go/pkg/mod"],
    "go": ["/root/.cache/go-build", "/go/pkg/mod"],
    "cargo": ["/root/.cargo/registry", "/root/.cargo/git"],
    "maven": ["/root/.m2"],
    "mvn": ["/root/.m2"],
    "gradle": ["/root/.gradle/caches", "/root/.gradle/wrapper"],
    "composer": ["/root/.composer/cache"],
}


_TORCH_DEP_NAMES = frozenset(
    {
        "torch",
        "torchvision",
        "torchaudio",
        "sentence-transformers",
        "transformers",
    }
)

_CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
_CPU_TORCH_INSTALL_LINE = f"pip install --index-url {_CPU_TORCH_INDEX} torch"

_PIP_LIKE_MANAGERS = frozenset({"pip", "pip3", "poetry", "pdm", "uv"})


def runtime_cache_mount_args(package_manager: str | None) -> list[str]:
    """Return BuildKit `--mount=type=cache,target=...` flags for a package
    manager. Unknown / unset managers return an empty list so callers can
    splice the output into a RUN line without changing behavior."""
    if not package_manager:
        return []
    targets = _CACHE_MOUNT_TARGETS.get(package_manager.strip().lower())
    if not targets:
        return []
    return [f"--mount=type=cache,target={t}" for t in targets]


def _is_install_run(line: str) -> bool:
    stripped = line.strip()
    if not stripped.upper().startswith("RUN "):
        return False
    return "install.sh" in stripped


def _apply_cache_mounts(run_line: str, mounts: list[str]) -> str:
    """Splice BuildKit cache mount flags into a RUN line, idempotently."""
    stripped = run_line.lstrip()
    indent = run_line[: len(run_line) - len(stripped)]
    rest = stripped[4:]  # strip leading "RUN "
    needed = [m for m in mounts if m not in stripped]
    if not needed:
        return run_line
    return f"{indent}RUN {' '.join(needed)} {rest}"


def normalize_dockerfile(dockerfile: str, *, package_manager: str | None) -> str:
    """Rewrite the LLM-authored Dockerfile for cacheability.

    Adds BuildKit cache mounts to the RUN line that invokes
    `.coursegen/runtime/install.sh`. Every `COPY` line is preserved exactly
    as authored — see the module docstring for why the prior `COPY . <dst>`
    strip was removed. The cache-mount transform is idempotent.
    """
    mounts = runtime_cache_mount_args(package_manager)
    out_lines: list[str] = []
    for raw_line in dockerfile.splitlines():
        if mounts and _is_install_run(raw_line):
            out_lines.append(_apply_cache_mounts(raw_line, mounts))
            continue
        out_lines.append(raw_line)
    trailing = "\n" if dockerfile.endswith("\n") else ""
    return "\n".join(out_lines) + trailing


def _requirements_mentions_torch(requirements_content: str | None) -> bool:
    if not requirements_content:
        return False
    for raw in requirements_content.splitlines():
        name = raw.split("#", 1)[0].strip()
        if not name:
            continue
        for sep in ("==", ">=", "<=", "~=", ">", "<", "!="):
            if sep in name:
                name = name.split(sep, 1)[0].strip()
                break
        name = name.split("[", 1)[0].strip().lower()
        if name in _TORCH_DEP_NAMES:
            return True
    return False


def normalize_install_sh(
    install_sh: str,
    *,
    requirements_content: str | None,
    package_manager: str | None,
) -> str:
    """Prepend a CPU-only torch install when the requirements file would
    otherwise drag in NVIDIA CUDA wheels. No-op for non-pip stacks, or when
    a CPU torch install is already present."""
    pm = (package_manager or "").strip().lower()
    if pm not in _PIP_LIKE_MANAGERS:
        return install_sh
    if not _requirements_mentions_torch(requirements_content):
        return install_sh
    if _CPU_TORCH_INDEX in install_sh:
        return install_sh
    lines = install_sh.splitlines(keepends=True)
    insert_at = len(lines)
    for idx, line in enumerate(lines):
        if "pip install" in line and "requirements" in line:
            insert_at = idx
            break
    newline = "\n"
    if lines:
        sample = next((ln for ln in lines if ln.endswith(("\r\n", "\n"))), None)
        if sample is not None and sample.endswith("\r\n"):
            newline = "\r\n"
    lines.insert(insert_at, _CPU_TORCH_INSTALL_LINE + newline)
    return "".join(lines)


_DOCKERFILE_KEY = "Dockerfile"
_INSTALL_SH_KEY = ".coursegen/runtime/install.sh"


def normalize_runtime_protocol_dict(
    runtime_files: dict[str, str],
    *,
    requirements_content: str | None,
    package_manager: str | None,
) -> dict[str, str]:
    """Apply Dockerfile + install.sh normalization to an authored
    `{relative_path: content}` runtime-protocol bundle. Returns a new dict;
    the input is not mutated. Missing keys are a no-op."""
    out = dict(runtime_files)
    if _DOCKERFILE_KEY in out:
        out[_DOCKERFILE_KEY] = normalize_dockerfile(
            out[_DOCKERFILE_KEY], package_manager=package_manager
        )
    if _INSTALL_SH_KEY in out:
        out[_INSTALL_SH_KEY] = normalize_install_sh(
            out[_INSTALL_SH_KEY],
            requirements_content=requirements_content,
            package_manager=package_manager,
        )
    return out
