from app.services.runtime_normalization import (
    normalize_dockerfile,
    normalize_install_sh,
    normalize_runtime_protocol_dict,
    runtime_cache_mount_args,
)


def test_dockerfile_strips_trailing_copy_workspace_to_app():
    docker_in = (
        "FROM python:3.11-slim\n"
        "WORKDIR /app\n"
        "COPY requirements.txt /app/\n"
        "RUN pip install -r requirements.txt\n"
        "COPY . /app\n"
        "EXPOSE 8000\n"
    )
    out = normalize_dockerfile(docker_in, package_manager="pip")
    assert "COPY . /app" not in out
    assert "FROM python:3.11-slim" in out
    assert "EXPOSE 8000" in out


def test_dockerfile_strips_trailing_copy_dot_dot():
    docker_in = "FROM node:18\nWORKDIR /app\nRUN npm ci\nCOPY . .\n"
    out = normalize_dockerfile(docker_in, package_manager="npm")
    assert "COPY . ." not in out


def test_dockerfile_keeps_targeted_copy_unchanged():
    docker_in = (
        "FROM python:3.11-slim\n"
        "WORKDIR /app\n"
        "COPY requirements.txt /app/requirements.txt\n"
        "COPY app /app/app\n"
        "RUN pip install -r requirements.txt\n"
    )
    out = normalize_dockerfile(docker_in, package_manager="pip")
    # COPY of a specific subpath, not the whole context, must survive
    assert "COPY app /app/app" in out
    assert "COPY requirements.txt" in out


def test_dockerfile_inserts_pip_cache_mount_on_install_run():
    docker_in = (
        "FROM python:3.11-slim\n"
        "WORKDIR /app\n"
        "COPY requirements.txt /app/requirements.txt\n"
        "COPY .coursegen/runtime/install.sh /app/.coursegen/runtime/install.sh\n"
        "RUN chmod +x /app/.coursegen/runtime/install.sh && /app/.coursegen/runtime/install.sh\n"
        "EXPOSE 8000\n"
    )
    out = normalize_dockerfile(docker_in, package_manager="pip")
    assert "--mount=type=cache,target=/root/.cache/pip" in out
    # the mount must land on a RUN line (Dockerfile-level), not as a free-standing line
    assert "RUN --mount=type=cache" in out


def test_dockerfile_inserts_npm_cache_mount_on_install_run():
    docker_in = (
        "FROM node:18\n"
        "WORKDIR /app\n"
        "COPY package.json /app/package.json\n"
        "COPY .coursegen/runtime/install.sh /app/.coursegen/runtime/install.sh\n"
        "RUN chmod +x /app/.coursegen/runtime/install.sh && /app/.coursegen/runtime/install.sh\n"
    )
    out = normalize_dockerfile(docker_in, package_manager="npm")
    assert "--mount=type=cache,target=/root/.npm" in out


def test_dockerfile_cache_mount_is_idempotent():
    docker_in = (
        "FROM python:3.11-slim\n"
        "RUN --mount=type=cache,target=/root/.cache/pip /app/.coursegen/runtime/install.sh\n"
    )
    out = normalize_dockerfile(docker_in, package_manager="pip")
    assert out.count("--mount=type=cache,target=/root/.cache/pip") == 1


def test_install_sh_prepends_cpu_torch_when_torch_in_requirements_pip():
    install_in = (
        "#!/usr/bin/env sh\n"
        "set -eu\n"
        "python -m pip install --upgrade pip\n"
        "pip install -r requirements.txt\n"
    )
    requirements = "sentence-transformers==3.1.1\nnumpy==1.26.4\n"
    out = normalize_install_sh(
        install_in,
        requirements_content=requirements,
        package_manager="pip",
    )
    assert "download.pytorch.org/whl/cpu" in out
    # The original requirements install must still run after the CPU torch install
    cpu_idx = out.index("download.pytorch.org/whl/cpu")
    req_idx = out.index("pip install -r requirements.txt")
    assert cpu_idx < req_idx
    # And the script must still start with the shebang line preserved
    assert out.startswith("#!/usr/bin/env sh")


def test_install_sh_unchanged_for_python_without_torch():
    install_in = "#!/usr/bin/env sh\npip install -r requirements.txt\n"
    out = normalize_install_sh(
        install_in,
        requirements_content="fastapi\npydantic\n",
        package_manager="pip",
    )
    assert "download.pytorch.org" not in out


def test_install_sh_unchanged_for_non_pip():
    install_in = "#!/usr/bin/env sh\nnpm ci\n"
    out = normalize_install_sh(
        install_in,
        requirements_content=None,
        package_manager="npm",
    )
    assert out.strip() == install_in.strip()


def test_install_sh_cpu_torch_idempotent():
    install_in = (
        "#!/usr/bin/env sh\n"
        "set -eu\n"
        "pip install --index-url https://download.pytorch.org/whl/cpu torch\n"
        "pip install -r requirements.txt\n"
    )
    requirements = "torch\nfastapi\n"
    out = normalize_install_sh(
        install_in,
        requirements_content=requirements,
        package_manager="pip",
    )
    # Should not double up the cpu torch install
    assert out.count("download.pytorch.org/whl/cpu") == 1


def test_runtime_cache_mount_args_known_managers():
    assert runtime_cache_mount_args("pip") == ["--mount=type=cache,target=/root/.cache/pip"]
    assert runtime_cache_mount_args("npm") == ["--mount=type=cache,target=/root/.npm"]
    assert runtime_cache_mount_args("bundler") == ["--mount=type=cache,target=/usr/local/bundle/cache"]
    # Go gets two cache dirs (module cache + build cache)
    args = runtime_cache_mount_args("gomod")
    assert "--mount=type=cache,target=/root/.cache/go-build" in args
    assert "--mount=type=cache,target=/go/pkg/mod" in args


def test_runtime_cache_mount_args_unknown_manager_returns_empty():
    assert runtime_cache_mount_args(None) == []
    assert runtime_cache_mount_args("") == []
    assert runtime_cache_mount_args("totally-unknown-pm") == []


def test_normalize_runtime_protocol_dict_applies_both_transforms():
    runtime = {
        "Dockerfile": "FROM python:3.11-slim\nCOPY . /app\n",
        ".coursegen/runtime/install.sh": (
            "#!/usr/bin/env sh\npip install -r requirements.txt\n"
        ),
        ".coursegen/runtime/verify.sh": "#!/usr/bin/env sh\npython -c 'import fastapi'\n",
    }
    out = normalize_runtime_protocol_dict(
        runtime,
        requirements_content="sentence-transformers==3.1.1\n",
        package_manager="pip",
    )
    assert "COPY . /app" not in out["Dockerfile"]
    assert "download.pytorch.org/whl/cpu" in out[".coursegen/runtime/install.sh"]
    # Untouched scripts pass through unchanged
    assert out[".coursegen/runtime/verify.sh"] == runtime[".coursegen/runtime/verify.sh"]


def test_normalize_runtime_protocol_dict_safe_when_keys_absent():
    out = normalize_runtime_protocol_dict(
        {}, requirements_content=None, package_manager="pip"
    )
    assert out == {}


def test_normalize_runtime_protocol_dict_does_not_mutate_caller_dict():
    runtime = {"Dockerfile": "FROM python:3.11-slim\nCOPY . /app\n"}
    snapshot = dict(runtime)
    normalize_runtime_protocol_dict(
        runtime, requirements_content=None, package_manager="pip"
    )
    assert runtime == snapshot
