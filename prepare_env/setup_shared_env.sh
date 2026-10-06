#!/usr/bin/env bash
# 共享服务器的独立环境安装器；放在仓库根目录，使用 bash 执行。
# 安装：bash setup_shared_env.sh
# CPU 验收：bash setup_shared_env.sh --check
# GPU 空闲且已约定使用时：bash setup_shared_env.sh --gpu-check
# 显式修正已核验的过期清单：bash setup_shared_env.sh --repair-manifest
# 默认不改原仓库；--repair-manifest 只备份并修正清单的一项哈希。
# 不调用原 setup.sh，不改系统驱动 / Conda base / uv.lock / 已有源码。
# Python、下载、缓存、CUDA 用户态兼容库均留在本项目目录。
set -Eeuo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '请使用 bash 执行安装器，不要 source 此文件。\n' >&2
  return 1
fi

usage() {
  cat <<'HELP'
用法：bash setup_shared_env.sh [--project 仓库目录] [--check | --gpu-check | --repair-manifest]
默认只安装并执行 CPU 验收，GPU 对安装进程不可见。
--check      不下载、不安装，只验收已有环境。
--gpu-check  不安装、不训练；GPU 存在计算进程或忙碌时拒绝测试。
--repair-manifest  显式授权备份并修正 data/environment.json 中已核验的一项过期哈希；不安装、不使用 GPU。
环境变量：
  SHOPPING_DOWNLOAD_PROXY  国外下载代理，默认 http://127.0.0.1:17897
  SHOPPING_DOWNLOAD_RATE   单个大文件下载限速，默认 8M（curl 格式）
安装结束后：source .venv-setup/activate.sh
激活文件默认隐藏 GPU；约定空闲后才可 export CUDA_VISIBLE_DEVICES=0。
HELP
}

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODE=install
while (($#)); do
  case "$1" in
    --project)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      ROOT="$(cd -- "$2" && pwd -P)"
      shift 2
      ;;
    --check|--gpu-check|--repair-manifest)
      [[ "$MODE" == install ]] || { usage >&2; exit 2; }
      MODE="${1#--}"
      shift
      ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

log() { printf '\n[shopping-env] %s\n' "$*"; }
die() { printf '\n[shopping-env] 错误：%s\n' "$*" >&2; exit 1; }
trap 'printf "\n[shopping-env] 第 %s 行失败；未执行任何驱动更改。请保留现有目录，修复报错后重试。\n" "$LINENO" >&2' ERR

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die '需要 Linux x86_64；此脚本应在服务器运行。'
[[ $EUID -ne 0 ]] || die '请以普通用户执行，不要使用 sudo。'
[[ -f "$ROOT/pyproject.toml" && -f "$ROOT/uv.lock" ]] || die '未找到项目根目录。'
SHOP_ENV="$ROOT/environments/ShopSimulator/shop_env"
MAIN_ENV="$ROOT/.venv"
SIM_ENV="$ROOT/environments/ShopSimulator/.venv-shopsim"
STATE="$ROOT/.venv-setup"
PRODUCT_SHA=57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f
PROXY_URL="${SHOPPING_DOWNLOAD_PROXY:-http://127.0.0.1:17897}"
DOWNLOAD_RATE="${SHOPPING_DOWNLOAD_RATE:-8M}"
TRANSFORMERS_URL=https://github.com/huggingface/transformers.git
TRANSFORMERS_REV=7ea2320c76117e6742364808a666ef6f2fb40a67
GIT_TOOL_PATH="$PATH"
[[ "$PROXY_URL" == http://* || "$PROXY_URL" == https://* ]] || die '代理地址必须为 http:// 或 https://。'

# 只改变安装器及其子进程；既有 SSH 会话和学长的进程不受影响。
export CUDA_VISIBLE_DEVICES=-1 NVIDIA_VISIBLE_DEVICES=none
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1
# install / find 显式使用 --managed-python；不要再同时设置 Python preference。
# 清除从父 shell 继承的互斥选项，设置只影响安装器及其子进程。
unset UV_PYTHON_PREFERENCE UV_MANAGED_PYTHON UV_NO_MANAGED_PYTHON
export UV_PYTHON_INSTALL_BIN=0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD VIRTUAL_ENV
unset CONDA_PREFIX CONDA_DEFAULT_ENV
export UV_CACHE_DIR="$STATE/uv-cache" UV_PYTHON_INSTALL_DIR="$STATE/python"
export HF_HOME="$STATE/huggingface" XDG_CACHE_HOME="$STATE/cache"
export CUDA_CACHE_PATH="$STATE/cuda-cache" TRITON_CACHE_DIR="$STATE/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$STATE/torchinductor-cache"
export SHOP_SEARCH_INDEX="$STATE/products.sqlite3"
export SHOPPING_GRPO_ROOT="$ROOT"
export SHOPPING_ENVIRONMENT_VERSION=shopsimulator-environment-v2.1
export SHOPPING_ENV_MANIFEST="$ROOT/data/environment.json"

for command_name in git gzip sha256sum awk patch dpkg-deb curl flock nice ionice nvidia-smi tee timeout chmod; do
  command -v "$command_name" >/dev/null 2>&1 || die "缺少命令：$command_name（脚本不会安装系统包）。"
done
UV="$(command -v uv)" || die '找不到 uv；请使用服务器已有的 uv。'
REAL_GIT="$(command -v git)"
SCRIPT="$(cd -- "$(dirname -- "$0")" && pwd -P)/$(basename -- "$0")"
if [[ "${SHOPPING_SETUP_LOW_PRIORITY:-0}" != 1 ]]; then
  restart_args=(--project "$ROOT")
  [[ "$MODE" == install ]] || restart_args+=("--$MODE")
  exec env SHOPPING_SETUP_LOW_PRIORITY=1 nice -n 19 ionice -c 3 \
    bash "$SCRIPT" "${restart_args[@]}"
fi

umask 077
[[ ! -L "$STATE" ]] || die '私有安装目录不能是符号链接。'
if [[ -e "$STATE" ]]; then
  [[ -f "$STATE/owner" && "$(cat "$STATE/owner")" == "$ROOT" ]] || die '私有安装目录已存在且不属于本安装器；请使用独立项目副本。'
elif [[ "$MODE" != install ]]; then
  die '尚未安装；先运行 bash setup_shared_env.sh。'
else
  mkdir -- "$STATE"
  printf '%s\n' "$ROOT" > "$STATE/owner"
fi
exec 9>"$STATE/install.lock"
flock -n 9 || die '本项目已有安装或验收进程；请等待它完成。'

guard_env() {
  local env_dir="$1" marker="$2"
  [[ ! -L "$env_dir" ]] || die "拒绝使用已有环境符号链接：$env_dir"
  if [[ -e "$env_dir" ]]; then
    [[ -f "$marker" && "$(cat "$marker")" == "$env_dir" ]] || die "拒绝修改已有环境：$env_dir；请在你的独立项目副本运行。"
  fi
}
guard_env "$MAIN_ENV" "$STATE/main-owner"
guard_env "$SIM_ENV" "$STATE/sim-owner"

domestic_uv() {
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    -u GIT_CONFIG_PARAMETERS -u GIT_SSL_NO_VERIFY \
    PATH="$GIT_TOOL_PATH" GIT_CONFIG_COUNT=0 GIT_TERMINAL_PROMPT=0 \
    SHOPPING_SETUP_REAL_GIT="$REAL_GIT" SHOPPING_SETUP_PROXY="$PROXY_URL" \
    SHOPPING_SETUP_GIT_CACHE="$UV_CACHE_DIR" SHOPPING_SETUP_TRANSFORMERS_URL="$TRANSFORMERS_URL" \
    SHOPPING_SETUP_TRANSFORMERS_REV="$TRANSFORMERS_REV" \
    "$UV" "$@"
}
proxy_uv() {
  env -u GIT_CONFIG_PARAMETERS -u GIT_SSL_NO_VERIFY \
    HTTP_PROXY="$PROXY_URL" HTTPS_PROXY="$PROXY_URL" ALL_PROXY="$PROXY_URL" \
    http_proxy="$PROXY_URL" https_proxy="$PROXY_URL" all_proxy="$PROXY_URL" \
    NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
    PATH="$GIT_TOOL_PATH" GIT_CONFIG_COUNT=0 GIT_TERMINAL_PROMPT=0 \
    SHOPPING_SETUP_REAL_GIT="$REAL_GIT" SHOPPING_SETUP_PROXY="$PROXY_URL" \
    SHOPPING_SETUP_GIT_CACHE="$UV_CACHE_DIR" SHOPPING_SETUP_TRANSFORMERS_URL="$TRANSFORMERS_URL" \
    SHOPPING_SETUP_TRANSFORMERS_REV="$TRANSFORMERS_REV" \
    "$UV" "$@"
}

prepare_private_git() {
  local tools="$STATE/git-tools"
  [[ ! -L "$tools" && ! -L "$tools/git" ]] || die '私有 Git 工具路径不能是符号链接。'
  mkdir -p -- "$tools"
  # uv 仍使用原 Git URL / 提交，direct_url.json 由 uv 正常生成。
  # 只给安装子进程使用此路径，不写入激活文件或用户的 Git 配置。
  cat > "$tools/git" <<'GIT_HELPER'
#!/usr/bin/env bash
set -euo pipefail
real_git="${SHOPPING_SETUP_REAL_GIT:?}"
proxy="${SHOPPING_SETUP_PROXY:?}"
upstream="${SHOPPING_SETUP_TRANSFORMERS_URL:?}"
revision="${SHOPPING_SETUP_TRANSFORMERS_REV:?}"
args=("$@")
url_index=-1
pinned=false
for ((i=0; i<${#args[@]}; i++)); do
  [[ "${args[i]}" != "$upstream" ]] || url_index=$i
  if [[ "${args[i]}" == "+$revision:refs/commit/$revision" || "${args[i]}" == "$revision" ]]; then
    pinned=true
  fi
done

if [[ "${args[0]-}" == fetch && "$url_index" -ge 0 && "$pinned" == true ]]; then
  cache="$(cd -- "${SHOPPING_SETUP_GIT_CACHE:?}" && pwd -P)"
  case "$(pwd -P)" in
    "$cache"/git-*/db/*) ;;
    *) printf '拒绝在项目私有 uv 缓存之外执行固定提交拉取。\n' >&2; exit 2 ;;
  esac
  # 国内镜像必须提供同一完整 SHA；没有该提交就回到官方源 + 代理。
  sources=(https://gitee.com/mirrors/huggingface-transformers.git "$upstream")
  status=1
  for source in "${sources[@]}"; do
    if [[ "$source" == "$upstream" ]]; then
      source_proxy="$proxy"; attempts=6; deadline=900
    else
      source_proxy=''; attempts=1; deadline=120
    fi
    fetch_args=("${args[@]}")
    fetch_args[url_index]="$source"
    for ((attempt=1; attempt<=attempts; attempt++)); do
      printf '固定 Transformers 提交：浅拉取，第 %s/%s 次，来源 %s\n' "$attempt" "$attempts" "$source" >&2
      if env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
          timeout --kill-after=10s "${deadline}s" "$real_git" \
          -c http.version=HTTP/1.1 -c "http.proxy=$source_proxy" \
          -c "http.$source.proxy=$source_proxy" -c "http.$source.sslVerify=true" \
          -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=60 -c pack.threads=1 \
          fetch --depth=1 --no-tags --no-recurse-submodules "${fetch_args[@]:1}"; then
        # 检查真实 Git 对象，不修改安装后的来源元数据。
        actual="$("$real_git" rev-parse --verify "refs/commit/$revision^{commit}")"
        [[ "$actual" == "$revision" ]] || { printf 'Git 提交校验失败。\n' >&2; exit 1; }
        "$real_git" fsck --connectivity-only --no-dangling "$revision"
        # uv 随后从此缓存 clone。浅仓库必须有可见分支和有效 HEAD，
        # 否则 Git 会把仅有 refs/commit 的来源当成空仓库。
        "$real_git" update-ref refs/heads/shopping-setup-pinned "$revision"
        "$real_git" symbolic-ref HEAD refs/heads/shopping-setup-pinned
        exit 0
      else
        status=$?
      fi
      if ((attempt < attempts)); then
        delay=$((2 ** attempt)); ((delay <= 20)) || delay=20
        printf 'Git 拉取未完成，保留私有缓存，%s 秒后重试。\n' "$delay" >&2
        sleep "$delay"
      fi
    done
  done
  exit "$status"
fi

# 使用 Git 2.25 已支持的 -c；不依赖较新版本的 GIT_CONFIG_COUNT。
exec "$real_git" -c http.version=HTTP/1.1 -c "http.proxy=$proxy" \
  -c http.sslVerify=true -c pack.threads=1 "$@"
GIT_HELPER
  chmod 700 -- "$tools/git"
  GIT_TOOL_PATH="$tools:$PATH"
}
pip_install() {
  local status diagnostics="$STATE/uv-pip-install.stderr" error_fd error_pid
  # 实时显示 stderr，并保留一份用于区分命令行用法错误与下载失败。
  exec {error_fd}> >(tee "$diagnostics" >&2)
  error_pid=$!
  if domestic_uv --no-config pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple "$@" 2>&"$error_fd"; then
    status=0
  else
    status=$?
  fi
  exec {error_fd}>&-
  wait "$error_pid"
  [[ "$status" -ne 0 ]] || return 0
  # uv 的网络错误也可能返回 2，不能仅凭退出码跳过官方源重试。
  if [[ "$status" -eq 2 ]] && awk '
    /^Usage: uv/ || /^error:.*(unexpected argument|cannot be used with|invalid value.*--|required arguments)/ {bad=1}
    END {exit !bad}
  ' "$diagnostics"; then
    die 'uv 安装参数检查失败；请先修复上方报错后重试。'
  fi
  log '国内 PyPI 安装未完成，使用官方 PyPI + 指定代理重试同一组固定依赖。'
  proxy_uv --no-config pip install --index-url https://pypi.org/simple "$@"
}

check_main_dependencies() {
  "$MAIN_ENV/bin/python" - "$ROOT" "$UV" <<'PY'
import importlib.metadata as md, pathlib, subprocess, sys, tomllib
root, uv = pathlib.Path(sys.argv[1]), sys.argv[2]
project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
locked = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
if (project["tool"]["uv"]["override-dependencies"] != ["numpy==2.2.6"]
        or locked["manifest"]["overrides"] != [{"name": "numpy", "specifier": "==2.2.6"}]):
    raise SystemExit("仓库依赖覆盖声明与预期不符；拒绝放宽依赖验收。")
check = subprocess.run(
    [uv, "--no-config", "--color", "never", "pip", "check", "--python", sys.executable],
    capture_output=True, text=True, encoding="utf-8",
)
print(check.stdout, end="")
print(check.stderr, end="", file=sys.stderr)
if check.returncode == 0:
    raise SystemExit(0)
lines = [line.strip() for line in (check.stdout + check.stderr).splitlines() if line.strip()]
expected = [
    "Found 1 incompatibility",
    "The package `verl` requires `numpy<2.0.0`, but `2.2.6` is installed",
]
# 原仓库已经明确覆盖此上限；保留上游 METADATA 和完整检查输出。
# 只接受这一项，缺包、其他冲突、不同版本或 uv 命令失败均停止。
if (check.returncode == 1 and lines[-2:] == expected
        and md.version("verl") == "0.8.0" and md.version("numpy") == "2.2.6"):
    print("按仓库明确声明接受 verl==0.8.0 的 NumPy 上限覆盖；其余基础依赖检查通过。")
    print("随后必须通过 NumPy / PyTorch / veRL 的 CPU 兼容性验收。")
else:
    raise SystemExit("发现仓库覆盖声明之外的依赖问题；请处理上方原始诊断。")
PY
}

check_frozen_manifest() {
  "$MAIN_ENV/bin/python" - "$ROOT" "$STATE" "$1" <<'PY'
import hashlib, json, os, pathlib, re, subprocess, sys, tempfile
root, state, mode = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve(), sys.argv[3]
old_sha = "448f5ec31fe5a71d1e43376d42cd84787a0e48fe96a8032fc0383f65e5928ea2"
new_sha = "d6db15f282849847a364d8d36af0eed113c1c97184a1ded98eb1fd29dab577d1"
relative = "environments/ShopSimulator/shop_env/web_agent_site/envs/web_agent_text_env.py"
files = {
    "observation.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/observation.py",
    "pack_api.py": "environments/ShopSimulator/shop_env/shop_env/pack_api.py",
    "reward.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py",
    "slot_lease_pool.py": "environments/ShopSimulator/shop_env/shop_env/slot_lease_pool.py",
    "web_agent_text_env.py": relative,
}
def sha(data):
    return hashlib.sha256(data).hexdigest()
def git_blob(name):
    return subprocess.run(["git", "-C", str(root), "show", "HEAD:" + name], capture_output=True, check=True).stdout
def safe_path(path):
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise SystemExit("拒绝修改符号链接或项目目录之外的文件：" + str(path))
def atomic_write(path, data):
    safe_path(path)
    permissions = path.stat().st_mode & 0o777 if path.exists() else 0o600
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".shopping-manifest-", delete=False) as stream:
            temporary = pathlib.Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, permissions)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
def backup(path, data):
    safe_path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise SystemExit("已有备份内容不符，拒绝覆盖：" + str(path))
    else:
        atomic_write(path, data)

target = root / "data/environment.json"
safe_path(target)
raw = target.read_bytes()
manifest = json.loads(raw)
expected = manifest.get("runtime_files_sha256", {})
if set(expected) != set(files):
    raise SystemExit("冻结清单的运行时文件列表不完整；不会自动重新生成哈希。")
mismatches = {}
for name, name_in_repo in files.items():
    with (root / name_in_repo).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected[name]:
        mismatches[name] = {"expected": expected[name], "actual": actual}
if not mismatches:
    print("Environment v2.1 冻结源码哈希预检：通过。")
    raise SystemExit(0)
known = {"web_agent_text_env.py": {"expected": old_sha, "actual": new_sha}}
if mismatches != known or (root / relative).read_bytes() != git_blob(relative):
    raise SystemExit("源码哈希存在未经核验的变化；拒绝自动修正：" + json.dumps(mismatches, sort_keys=True))
if raw != git_blob("data/environment.json"):
    raise SystemExit("data/environment.json 已有本地修改；拒绝覆盖。")
# Git 修复 d99a0ac 仅删除未定义的 selected_option 参数；只接受上面固定的一对哈希。
pattern = rb'("web_agent_text_env.py"\s*:\s*")' + old_sha.encode() + rb'(")'
proposed, count = re.subn(pattern, lambda match: match[1] + new_sha.encode() + match[2], raw)
if count != 1:
    raise SystemExit("清单目标字段不能唯一定位；拒绝修改。")
updated = json.loads(proposed)
updated["runtime_files_sha256"]["web_agent_text_env.py"] = old_sha
if updated != manifest:
    raise SystemExit("修正方案涉及其他清单字段；拒绝修改。")
atomic_write(state / "environment-manifest.proposed.json", proposed)
print("已核验 Git 最新源码；修正方案只更新 web_agent_text_env.py 的冻结哈希：")
print("  " + old_sha + " -> " + new_sha)
if mode != "repair-manifest":
    raise SystemExit("原清单未修改。请先审阅 .venv-setup/environment-manifest.proposed.json；同意这一项修正后显式运行 bash setup_shared_env.sh --repair-manifest。")

snapshot = state / "source-sha256.json"
snapshot_raw = snapshot.read_bytes() if snapshot.exists() else None
snapshot_updated = None
if snapshot_raw is not None:
    original = json.loads(snapshot_raw)
    if original.get("data/environment.json") != sha(raw):
        raise SystemExit("安装器的源码快照与原清单不符；拒绝更新快照。")
    for name, wanted in original.items():
        path = root / name
        if wanted is None:
            if path.exists():
                raise SystemExit("原本缺失的跟踪文件发生改变：" + name)
        else:
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != wanted:
                    raise SystemExit("其他跟踪文件发生改变；拒绝修正清单：" + name)
    pattern = rb'("data/environment.json"\s*:\s*")' + sha(raw).encode() + rb'(")'
    snapshot_updated, count = re.subn(pattern, lambda match: match[1] + sha(proposed).encode() + match[2], snapshot_raw)
    if count != 1:
        raise SystemExit("源码快照目标字段不能唯一定位；拒绝修改。")
backup(state / "environment-manifest.before-repair.json", raw)
if snapshot_raw is not None:
    backup(state / "source-sha256.before-manifest-repair.json", snapshot_raw)
try:
    atomic_write(target, proposed)
    if snapshot_updated is not None:
        atomic_write(snapshot, snapshot_updated)
except Exception:
    atomic_write(target, raw)
    if snapshot_raw is not None:
        atomic_write(snapshot, snapshot_raw)
    raise
print("已备份并仅修正这一项冻结哈希；源码、奖励和工具契约未改变。")
print("随后运行 bash setup_shared_env.sh；已有环境、下载和索引可继续复用。")
PY
}

set_cuda_library_path() {
  local base_libraries driver_version driver_major
  base_libraries="$("$MAIN_ENV/bin/python" - <<'PY'
import importlib.metadata as md, pathlib, sysconfig
site = pathlib.Path(sysconfig.get_paths()["purelib"])
paths = [site / "torch" / "lib"]
# FlashInfer 的 cuTile 附加依赖带有 CUDA 13 编译工具；只加入 cu12 包的库目录。
# 按实际 distribution 文件归属筛选，避免无后缀的 CUDA 13 库被递归加入。
libraries = set()
for distribution in md.distributions():
    name = distribution.metadata["Name"].lower().replace("_", "-")
    if name.startswith("nvidia-") and name.endswith("-cu12"):
        for file in distribution.files or []:
            if file.parent.name in {"lib", "lib64"}:
                libraries.add(pathlib.Path(distribution.locate_file(file)).parent.resolve())
paths += sorted(libraries)
print(":".join(dict.fromkeys(str(path) for path in paths if path.is_dir())))
PY
)"
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits)"
  driver_major="${driver_version%%.*}"
  [[ "$driver_major" =~ ^[0-9]+$ ]] || die '无法可靠读取单卡驱动版本。'
  if [[ "$driver_major" -eq 570 ]]; then
    export LD_LIBRARY_PATH="$STATE/cuda-compat/usr/local/cuda-12.9/compat:$base_libraries"
  elif [[ "$driver_major" -ge 575 ]]; then
    # 系统后来升级驱动时，较旧的兼容库不应继续覆盖系统 libcuda。
    export LD_LIBRARY_PATH="$base_libraries"
  else
    die "此脚本仅验证 R570 或原生支持 CUDA 12.9 的驱动：$driver_version"
  fi
}

if [[ "$MODE" == repair-manifest ]]; then
  [[ -x "$MAIN_ENV/bin/python" ]] || die '此修正入口需要已有项目 Python 环境；请先完成独立 Python 安装。'
  check_frozen_manifest repair-manifest
  exit 0
fi

if [[ "$MODE" == install ]]; then
  [[ -f "$SHOP_ENV/data/fine_items_eval_train_all.json.gz" ]] || die '缺少仓库内嵌商品压缩包。'
  free_kb="$(df -Pk "$ROOT" | awk 'END {print $4}')"
  [[ "$free_kb" =~ ^[0-9]+$ && "$free_kb" -ge 41943040 ]] || die '安装前至少预留 40 GiB 磁盘空间（不含模型权重）。'
  log '安装独立 Python 3.12 / 3.10；不使用 Conda base 或系统 Python。'
  if ! domestic_uv --no-config python install --no-bin --managed-python \
      --mirror https://registry.npmmirror.com/-/binary/python-build-standalone 3.12 3.10; then
    log '国内 Python 二进制镜像不可用；通过代理下载 Astral 官方发行包。'
    proxy_uv --no-config python install --no-bin --managed-python 3.12 3.10
  fi
  MAIN_PYTHON="$("$UV" --no-config python find --managed-python 3.12)"
  SIM_PYTHON="$("$UV" --no-config python find --managed-python 3.10)"
  for env_kind in main sim; do
    if [[ "$env_kind" == main ]]; then
      env_dir="$MAIN_ENV"; env_python="$MAIN_PYTHON"; marker="$STATE/main-owner"
    else
      env_dir="$SIM_ENV"; env_python="$SIM_PYTHON"; marker="$STATE/sim-owner"
    fi
    if [[ ! -e "$env_dir" ]]; then
      printf '%s\n' "$env_dir" > "$marker"
      "$UV" --no-config venv --python "$env_python" "$env_dir"
    fi
    [[ -x "$env_dir/bin/python" ]] || die "环境不完整，拒绝覆盖：$env_dir"
  done
  "$MAIN_ENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,12)'
  "$SIM_ENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,10)'
  check_frozen_manifest check

  # 在安装前后比较所有 Git 跟踪文件，包括数据，确保原仓库内容不变。
  "$MAIN_ENV/bin/python" - "$ROOT" "$STATE/source-sha256.json" <<'PY'
import hashlib, json, pathlib, subprocess, sys
root, target = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
listing = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True)
if listing.returncode:
    raise SystemExit("请在完整 Git 工作区运行，才能核验原仓库文件未改变。")
checks = {}
for item in listing.stdout.split(b"\0"):
    if not item:
        continue
    name = item.decode("utf-8")
    path = root / name
    if path.is_file():
        with path.open("rb") as stream:
            checks[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    else:
        checks[name] = None
target.write_text(json.dumps(checks, ensure_ascii=False), encoding="utf-8")
PY

  log '准备匹配的 CUDA 12.9 wheels 和私有 cuda-compat-12-9；不会执行 dpkg 安装。'
  "$MAIN_ENV/bin/python" - "$STATE" "$PROXY_URL" "$DOWNLOAD_RATE" <<'PY'
import gzip, hashlib, pathlib, re, subprocess, sys, time
from html.parser import HTMLParser
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

state, proxy, rate = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
wheelhouse = state / "wheels"
wheelhouse.mkdir(exist_ok=True)

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def get(url, output, overseas=False, *, resume=False, retries=2):
    command = ["curl", "--disable", "--fail", "--location", "--silent", "--show-error", "--http1.1",
               "--proto", "=https", "--proto-redir", "=https", "--connect-timeout", "15",
               "--max-time", "1800", "--speed-limit", "1024", "--speed-time", "60",
               "--retry", str(retries), "--limit-rate", rate,
               "--proxy", proxy if overseas else "", "--noproxy", "" if overseas else "*",
               "--write-out", "%{http_code}", "--output", str(output)]
    if resume:
        command += ["--continue-at", "-", "--header", "Cache-Control: no-cache"]
    command.append(url)
    return subprocess.run(command, stdout=subprocess.PIPE, text=True)

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            link = dict(attrs).get("href")
            if link:
                self.links.append(link)

def match(index, filename, overseas):
    html = state / "wheel-index.html"
    if get(index, html, overseas).returncode != 0:
        return None
    parser = Links()
    parser.feed(html.read_text(encoding="utf-8", errors="replace"))
    for href in parser.links:
        url = urljoin(index, href)
        parts = urlsplit(url)
        if unquote(parts.path.rsplit("/", 1)[-1]) != filename:
            continue
        digest = re.search(r"(?:^|&)sha256=([0-9a-f]{64})(?:&|$)", parts.fragment)
        if digest and parts.scheme == "https":
            return urlunsplit(parts._replace(fragment="")), digest.group(1)
    return None

def obtain(filename, candidates):
    output = wheelhouse / filename
    temporary = output.with_name(output.name + ".partial")
    for url, digest, overseas in candidates:
        if output.is_file() and sha(output) == digest:
            print("复用已校验下载：", filename, flush=True)
            return output
        if temporary.is_file() and sha(temporary) == digest:
            temporary.replace(output)
            print("复用已完成并校验的断点文件：", filename, flush=True)
            return output
        allow_resume = True
        for attempt in range(1, 7):
            offset = temporary.stat().st_size if temporary.is_file() else 0
            action = f"断点续传（已有 {offset / 1048576:.1f} MiB）" if offset and allow_resume else "下载"
            print(f"{action}：{filename}（{'代理' if overseas else '国内镜像'}，第 {attempt}/6 次）", flush=True)
            # 每次重新启动 curl 并读取当前文件长度；覆盖 curl 18 / 56 等中断，
            # 不依赖 Ubuntu 20.04 的旧 curl 可能缺少的 --retry-all-errors。
            result = get(url, temporary, overseas, resume=allow_resume, retries=0)
            http_status = result.stdout.strip()
            if temporary.is_file() and sha(temporary) == digest:
                temporary.replace(output)
                print("SHA-256 校验通过：", filename, flush=True)
                return output
            if result.returncode == 0:
                temporary.unlink(missing_ok=True)
                print("完整文件 SHA-256 不匹配；仅清除这个未校验的断点文件并重新下载。", flush=True)
            elif offset and (result.returncode == 33 or http_status == "416"):
                temporary.unlink(missing_ok=True)
                allow_resume = False
                print("此来源无法继续当前断点；仅对此文件改用完整下载。", flush=True)
            elif result.returncode in {2, 3, 4, 23, 26, 27, 60, 77, 78} or (
                result.returncode == 22 and http_status in {"401", "404", "410"}
            ):
                print(f"此来源无法下载（curl {result.returncode}，HTTP {http_status}）；尝试下一个来源。", flush=True)
                break
            else:
                print(f"传输未完成（curl {result.returncode}，HTTP {http_status}）；保留断点文件。", flush=True)
            if attempt < 6:
                delay = min(2 ** attempt, 20)
                print(f"{delay} 秒后重试。", flush=True)
                time.sleep(delay)
        print("此来源未完成下载；尝试下一个来源。", flush=True)
    raise SystemExit("下载失败：" + filename + "；断点文件已保留，请检查代理端口和网络后重跑同一脚本。")

for name, version in (("torch", "2.11.0"), ("torchvision", "0.26.0"), ("torchaudio", "2.11.0")):
    filename = f"{name}-{version}+cu129-cp312-cp312-manylinux_2_28_x86_64.whl"
    domestic = f"https://mirror.sjtu.edu.cn/pytorch-wheels/cu129/{name}/"
    found = match(domestic, filename, False)
    if found:
        url, digest = found
        parts = urlsplit(url)
        if parts.netloc in {"download.pytorch.org", "download-r2.pytorch.org"}:
            url = "https://mirror.sjtu.edu.cn/pytorch-wheels/" + parts.path.removeprefix("/whl/")
        try:
            obtain(filename, [(url, digest, False)])
            continue
        except SystemExit:
            pass
    official = f"https://download.pytorch.org/whl/cu129/{name}/"
    found = match(official, filename, True)
    if not found:
        raise SystemExit("官方源缺少预期 CUDA 12.9 wheel：" + filename)
    obtain(filename, [(found[0], found[1], True)])

vllm_file = "vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
vllm_sha = "9e206f370c934a2d4b6b1f05d3d09708d344e05d80260189ef19f60755709431"
candidates = []
found = match("https://pypi.tuna.tsinghua.edu.cn/simple/vllm/", vllm_file, False)
if found and found[1] == vllm_sha:
    candidates.append((found[0], vllm_sha, False))
candidates.append(("https://github.com/vllm-project/vllm/releases/download/v0.25.1/" + vllm_file, vllm_sha, True))
obtain(vllm_file, candidates)

# NVIDIA 官方 Ubuntu 20.04 仓库；取固定版本，并核对其仓库 SHA-256。
base = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/"
catalog = state / "nvidia-Packages.gz"
if get(base + "Packages.gz", catalog, True).returncode != 0:
    raise SystemExit("无法读取 NVIDIA 官方包目录；请检查代理。")
paragraphs = re.split(r"\n\s*\n", gzip.decompress(catalog.read_bytes()).decode())
selected = None
for paragraph in paragraphs:
    fields = dict(re.findall(r"^([A-Za-z0-9-]+):[ \t]*(.*)$", paragraph, re.M))
    if fields.get("Package") == "cuda-compat-12-9" and fields.get("Version") == "575.57.08-0ubuntu1" and fields.get("Architecture") == "amd64":
        selected = fields
        break
if selected is None:
    raise SystemExit("NVIDIA 源缺少固定 cuda-compat-12-9 包；不会自动换驱动版本。")
filename = selected["Filename"].removeprefix("./")
if "/" in filename or not re.fullmatch(r"cuda-compat-12-9_[A-Za-z0-9.+-]+_amd64\.deb", filename):
    raise SystemExit("NVIDIA 包路径不符合预期。")
if not re.fullmatch(r"[0-9a-f]{64}", selected["SHA256"]):
    raise SystemExit("NVIDIA 包校验值不符合预期。")
deb = obtain(filename, [(base + filename, selected["SHA256"], True)])
(state / "compat-deb-path").write_text(str(deb), encoding="utf-8")
PY

  dpkg-deb --extract "$(cat "$STATE/compat-deb-path")" "$STATE/cuda-compat"
  [[ -f "$STATE/cuda-compat/usr/local/cuda-12.9/compat/libcuda.so.1" ]] || die '兼容库解包路径不符合预期。'

  log '准备固定 Transformers 提交的浅拉取；Git 设置只供本次安装使用。'
  prepare_private_git

  # 导出锁文件而不更新它；保留通用依赖版本，CUDA 相关依赖由 cu129 wheels 决定。
  "$UV" --no-config export --project "$ROOT" --frozen --no-dev --extra sft --extra grpo \
    --no-emit-project --no-hashes --output-file "$STATE/locked-requirements.txt" >/dev/null
  "$MAIN_ENV/bin/python" - "$STATE" "$ROOT" <<'PY'
import pathlib, re, sys, tomllib
state = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
locked = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
overrides = project["tool"]["uv"]["override-dependencies"]
if (overrides != ["numpy==2.2.6"]
        or locked["manifest"]["overrides"] != [{"name": "numpy", "specifier": "==2.2.6"}]):
    raise SystemExit("仓库依赖覆盖声明与冻结锁文件不符；不会自动更换 NumPy 版本。")
# uv pip install --no-config 不读取项目覆盖；显式传递原仓库的既有策略。
(state / "dependency-overrides.txt").write_text("\n".join(overrides) + "\n", encoding="utf-8")
kept = []
for line in (state / "locked-requirements.txt").read_text().splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        continue
    match = re.match(r"([A-Za-z0-9_.-]+)", stripped)
    if match is None:
        raise SystemExit("锁文件导出格式不符合预期：" + stripped)
    name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
    if name.startswith(("cuda-", "nvidia-")) or name == "shopping-grpo":
        continue
    kept.append(line)
kept += ["torch==2.11.0+cu129", "torchvision==0.26.0+cu129", "torchaudio==2.11.0+cu129",
         "vllm==0.25.1+cu129", "cuda-python>=12.8,<13", "cuda-bindings>=12.8,<13"]
(state / "cu129-constraints.txt").write_text("\n".join(kept) + "\n")
PY
  log '安装项目 SFT / GRPO 依赖；仅安装指定 cu129 的 PyTorch 与 vLLM。'
  # pip install 用 --only-binary 指定包；项目 editable 和 Transformers Git 源仍可构建。
  pip_install --python "$MAIN_ENV/bin/python" \
    --constraint "$STATE/cu129-constraints.txt" \
    --overrides "$STATE/dependency-overrides.txt" \
    --only-binary torch --only-binary vllm \
    "$STATE/wheels/torch-2.11.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl" \
    "$STATE/wheels/torchvision-0.26.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl" \
    "$STATE/wheels/torchaudio-2.11.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl" \
    "$STATE/wheels/vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    --editable "$ROOT[sft,grpo]"
  log '安装隔离的 ShopSimulator Python 3.10 依赖。'
  pip_install --python "$SIM_ENV/bin/python" -r "$SHOP_ENV/requirements.txt"

  set_cuda_library_path
  log '应用仓库自带 veRL 校验补丁；仅修改本项目虚拟环境中的包。'
  "$MAIN_ENV/bin/python" "$ROOT/scripts/apply_verl_dynamic_sampling_patch.py"

  products="$SHOP_ENV/data/items_eval_train.json"
  if [[ -e "$products" || -L "$products" ]]; then
    [[ -f "$products" && "$(sha256sum "$products" | awk '{print $1}')" == "$PRODUCT_SHA" ]] || die '已有商品数据不符合冻结哈希；拒绝覆盖。'
  else
    log '流式解压并校验商品数据到私有目录。'
    gzip -cd "$SHOP_ENV/data/fine_items_eval_train_all.json.gz" > "$STATE/items_eval_train.json.partial"
    [[ "$(sha256sum "$STATE/items_eval_train.json.partial" | awk '{print $1}')" == "$PRODUCT_SHA" ]] || die '商品数据 SHA-256 不匹配。'
    mv -- "$STATE/items_eval_train.json.partial" "$STATE/items_eval_train.json"
    ln -s -- "$STATE/items_eval_train.json" "$products"
  fi
  if [[ ! -f "$SHOP_SEARCH_INDEX" ]]; then
    log '构建本项目私有搜索索引；不启动 ShopSimulator 服务。'
    "$SIM_ENV/bin/python" "$SHOP_ENV/scripts/build_index.py" \
      --products "$products" --output "$SHOP_SEARCH_INDEX"
  fi

  # 文件可 source；所有设置局限于当前 shell，不写 ~/.bashrc 或 Git 配置。
  {
    printf '# 由 setup_shared_env.sh 生成；仅供本项目 shell 使用。\n'
    printf 'source %q\n' "$MAIN_ENV/bin/activate"
    printf '# deactivate 时恢复此 shell 原有的动态库路径，避免流入后续 Conda 任务。\n'
    printf '_shopping_had_ld_path=${LD_LIBRARY_PATH+x}\n'
    printf '_shopping_old_ld_path=${LD_LIBRARY_PATH-}\n'
    printf '_shopping_had_cuda_devices=${CUDA_VISIBLE_DEVICES+x}\n'
    printf '_shopping_old_cuda_devices=${CUDA_VISIBLE_DEVICES-}\n'
    printf 'eval "$(declare -f deactivate | sed '\''1s/deactivate/_shopping_original_deactivate/'\'')"\n'
    printf 'deactivate() {\n'
    printf '  if [[ "${_shopping_had_ld_path-}" == x ]]; then export LD_LIBRARY_PATH="${_shopping_old_ld_path-}"; else unset LD_LIBRARY_PATH; fi\n'
    printf '  if [[ "${_shopping_had_cuda_devices-}" == x ]]; then export CUDA_VISIBLE_DEVICES="${_shopping_old_cuda_devices-}"; else unset CUDA_VISIBLE_DEVICES; fi\n'
    printf '  unset _shopping_had_ld_path _shopping_old_ld_path _shopping_had_cuda_devices _shopping_old_cuda_devices\n'
    printf '  _shopping_original_deactivate "$@"\n'
    printf '}\n'
    printf 'export SHOPPING_GRPO_ROOT=%q\n' "$ROOT"
    printf 'export SHOPPING_ENV_MANIFEST=%q\n' "$ROOT/data/environment.json"
    printf 'export SHOPPING_ENVIRONMENT_VERSION=shopsimulator-environment-v2.1\n'
    printf '# 已安装的 CUDA 12.9 runtime libraries；仅当前进程使用。\n'
    printf 'export LD_LIBRARY_PATH=%q\n' "${LD_LIBRARY_PATH#"$STATE/cuda-compat/usr/local/cuda-12.9/compat:"}"
    printf 'case "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits)" in\n'
    printf '  570.*) export LD_LIBRARY_PATH=%q:"$LD_LIBRARY_PATH" ;;\n' "$STATE/cuda-compat/usr/local/cuda-12.9/compat"
    printf 'esac\n'
    printf 'export SHOP_SEARCH_INDEX=%q\n' "$SHOP_SEARCH_INDEX"
    printf 'export UV_CACHE_DIR=%q UV_PYTHON_INSTALL_DIR=%q\n' "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
    printf 'export HF_HOME=%q XDG_CACHE_HOME=%q\n' "$HF_HOME" "$XDG_CACHE_HOME"
    printf 'export CUDA_CACHE_PATH=%q TRITON_CACHE_DIR=%q TORCHINDUCTOR_CACHE_DIR=%q\n' "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
    printf 'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1\n'
    printf 'unset PYTHONHOME PYTHONPATH LD_PRELOAD\n'
    printf 'export CUDA_VISIBLE_DEVICES=-1\n'
    printf 'printf "项目环境已激活，GPU 默认隐藏；约定空闲后才可 export CUDA_VISIBLE_DEVICES=0。\\n"\n'
  } > "$STATE/activate.sh"
fi

[[ -x "$MAIN_ENV/bin/python" && -x "$SIM_ENV/bin/python" && -f "$STATE/activate.sh" ]] || die '安装尚未完成；请重试安装模式。'
[[ "$MODE" == install ]] || check_frozen_manifest check
set_cuda_library_path
log 'CPU 验收：固定版本、CUDA 构建、兼容库、项目契约、补丁及索引。'
check_main_dependencies
"$UV" --no-config pip check --python "$SIM_ENV/bin/python"
"$MAIN_ENV/bin/python" - "$ROOT" "$STATE" "$PRODUCT_SHA" <<'PY'
import ctypes, hashlib, importlib.metadata as md, json, pathlib, pickle, runpy, sqlite3, sys
root, state, product_sha = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
assert sys.version_info[:2] == (3, 12)
runtime = runpy.run_path(str(root / "scripts/check_grpo_runtime.py"))
for name, expected in runtime["EXPECTED_VERSIONS"].items():
    installed = md.version(name)
    if installed.split("+", 1)[0] != expected:
        raise SystemExit(f"版本不匹配：{name}={installed}，期望 {expected}")
    print(f"{name}=={installed}")
runtime["validate_transformers_revision"]()
runtime["validate_environment_contract"]()
import torch
import vllm._C
if torch.version.cuda != "12.9" or "+cu129" not in md.version("vllm"):
    raise SystemExit("检测到错误 CUDA 构建；要求 PyTorch 与 vLLM 均为 cu129。")
assert not torch.cuda.is_initialized(), "CPU 验收不得创建 CUDA 上下文"
# 仓库覆盖了 veRL 的 NumPy 上限；实际测试数组互转和 GRPO 数据批操作。
import numpy as np
from verl import DataProto
array = np.arange(6, dtype=np.float32).reshape(2, 3)
tensor = torch.from_numpy(array)
tensor[0, 0] = 9
assert array[0, 0] == 9, "NumPy / PyTorch 共享数组不兼容"
np.testing.assert_array_equal(tensor.numpy(), array)
proto = DataProto.from_dict(
    tensors={"probe": tensor},
    non_tensors={"uid": np.array(["first", "second"], dtype=object)},
)
selected = proto[np.array([1, 0], dtype=np.int64)]
joined = DataProto.concat([selected[:1], selected[1:]])
restored = pickle.loads(pickle.dumps(joined))
assert torch.equal(restored.batch["probe"], tensor[[1, 0]])
assert restored.non_tensor_batch["uid"].tolist() == ["second", "first"]
assert not torch.cuda.is_initialized(), "CPU 兼容性验收不得创建 CUDA 上下文"
print("NumPy 2.2.6 / PyTorch / veRL DataProto 互转、选择、合并和序列化：通过。")
cuda13_toolchain = {
    "nvidia-cuda-crt", "nvidia-cuda-nvcc", "nvidia-cuda-runtime",
    "nvidia-cuda-tileiras", "nvidia-nvjitlink", "nvidia-nvvm",
}
cuda13_runtime = {
    "nvidia-cublas", "nvidia-cuda-cccl", "nvidia-cuda-cupti", "nvidia-cuda-nvrtc",
    "nvidia-cudnn", "nvidia-cufft", "nvidia-cufile", "nvidia-curand", "nvidia-cusolver",
    "nvidia-cusparse", "nvidia-cusparselt", "nvidia-nccl", "nvidia-npp",
    "nvidia-nvfatbin", "nvidia-nvjpeg", "nvidia-nvshmem", "nvidia-nvtx",
}
compiler_libraries = set()
for distribution in md.distributions():
    name = distribution.metadata["Name"].lower().replace("_", "-")
    if name in cuda13_runtime or (name.startswith("nvidia-") and name.endswith("-cu13")):
        raise SystemExit("检测到 CUDA 13 核心运行库混入 CUDA 12.9 环境：" + name)
    if name in cuda13_toolchain:
        compiler_libraries.update(
            str(pathlib.Path(distribution.locate_file(file)).resolve())
            for file in distribution.files or [] if ".so" in file.name
        )
# 工具链可随依赖共存，但 CUDA 12.9 进程不得实际加载它的 CUDA 13 库。
loaded = {
    str(pathlib.Path(parts[5].removesuffix(" (deleted)")).resolve())
    for line in pathlib.Path("/proc/self/maps").read_text().splitlines()
    if len(parts := line.split(maxsplit=5)) == 6 and parts[5].startswith("/")
}
bad_libraries = sorted(loaded & compiler_libraries)
if bad_libraries:
    raise SystemExit("CUDA 12.9 进程实际加载了 CUDA 13 辅助库：" + ", ".join(bad_libraries))
if compiler_libraries:
    print("附带的 cuTile CUDA 13 编译工具未被当前 CUDA 12.9 进程加载。")
driver = ctypes.CDLL("libcuda.so.1")
driver.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
driver.cuDriverGetVersion.restype = ctypes.c_int
version = ctypes.c_int()
if driver.cuDriverGetVersion(ctypes.byref(version)) != 0 or version.value < 12090:
    raise SystemExit("CUDA 用户态库版本不足 12.9。")
# cuDriverGetVersion 不需要 cuInit，不分配显存或执行 GPU 计算。
print("CUDA 用户态 Driver API：", version.value, "（尚未执行 GPU 验收）")
connection = sqlite3.connect((state / "products.sqlite3").as_uri() + "?mode=ro", uri=True)
manifest = json.loads(connection.execute("SELECT payload FROM manifest").fetchone()[0])
connection.close()
if manifest["product_data_sha256"] != product_sha or manifest["search_version"] != "shopsimulator-multifield-bm25-v2":
    raise SystemExit("搜索索引契约不匹配。")
original = json.loads((state / "source-sha256.json").read_text(encoding="utf-8"))
for name, expected in original.items():
    path = root / name
    if expected is None:
        if path.exists():
            raise SystemExit("原本缺失的跟踪文件发生改变：" + name)
        continue
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise SystemExit("原仓库跟踪文件发生改变：" + name)
print("原仓库跟踪文件内容未改变。")
PY
"$MAIN_ENV/bin/python" "$ROOT/scripts/apply_verl_dynamic_sampling_patch.py" --check
"$SIM_ENV/bin/python" - <<'PY'
import importlib.metadata as md, sqlite3, sys
assert sys.version_info[:2] == (3, 10)
for name, expected in {"numpy":"1.26.4", "gym":"0.24.0", "flask":"2.1.2", "Werkzeug":"2.1.2"}.items():
    assert md.version(name) == expected, (name, md.version(name))
with sqlite3.connect(":memory:") as connection:
    connection.execute("CREATE VIRTUAL TABLE fts_probe USING fts5(text)")
print("ShopSimulator Python 3.10 / 独立 NumPy / SQLite FTS5：通过。")
PY

if [[ "$MODE" == gpu-check ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || die '找不到 nvidia-smi。'
  # 不能仅看 GPU-Util=0；正在等待数据的训练进程仍然占用 GPU。
  compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${compute_pids//[[:space:]]/}" ]] || die "GPU 仍有计算进程（PID: $compute_pids）；未执行 GPU 测试，请等学长的任务结束。"
  gpu_state="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)"
  [[ "$(printf '%s\n' "$gpu_state" | wc -l)" -eq 1 ]] || die '仅支持单卡验收。'
  IFS=',' read -r gpu_util gpu_memory <<< "$gpu_state"
  gpu_util="${gpu_util//[[:space:]]/}"; gpu_memory="${gpu_memory//[[:space:]]/}"
  [[ "$gpu_util" =~ ^[0-9]+$ && "$gpu_memory" =~ ^[0-9]+$ ]] || die 'GPU 状态无法可靠读取，拒绝测试。'
  [[ "$gpu_util" -eq 0 && "$gpu_memory" -lt 1024 ]] || die 'GPU 尚未空闲或存在明显显存占用；拒绝测试。'
  log 'GPU 空闲检查通过；执行小规模 PyTorch / vLLM / Triton 验收，不加载模型。'
  cat > "$STATE/gpu-smoke.py" <<'PY'
import ctypes
driver = ctypes.CDLL("libcuda.so.1")
driver.cuInit.argtypes = [ctypes.c_uint]
driver.cuInit.restype = ctypes.c_int
status = driver.cuInit(0)
if status:
    detail = {803:"系统内核驱动与用户态库不兼容", 804:"这张 GPU 不支持当前前向兼容配置"}.get(status, "CUDA 初始化失败")
    raise SystemExit(f"{detail}，CUDA 错误码 {status}。请保留 CPU 环境；脚本不会更改系统驱动。")
device = ctypes.c_int()
driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
driver.cuDeviceGet.restype = ctypes.c_int
assert driver.cuDeviceGet(ctypes.byref(device), 0) == 0
driver.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
driver.cuDeviceGetAttribute.restype = ctypes.c_int
for attribute in (102, 103):  # VMM、POSIX 文件描述符；veRL / vLLM 内存管理需要。
    value = ctypes.c_int()
    if driver.cuDeviceGetAttribute(ctypes.byref(value), attribute, device.value) or not value.value:
        raise SystemExit(f"驱动缺少所需虚拟内存能力（属性 {attribute}）。")
import torch
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (12, 0)
matrix = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
torch.testing.assert_close(matrix @ matrix, torch.full_like(matrix, 32))
import vllm._custom_ops as ops
x = torch.randn((2, 32), device="cuda", dtype=torch.bfloat16)
y = torch.empty((2, 16), device="cuda", dtype=torch.bfloat16)
ops.silu_and_mul(y, x)
torch.testing.assert_close(y, torch.nn.functional.silu(x[:, :16]) * x[:, 16:], atol=0.02, rtol=0.02)
import triton
import triton.language as tl
@triton.jit
def add_one(source, target, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(source + offsets, offsets < N, other=0)
    tl.store(target + offsets, values + 1, offsets < N)
source = torch.arange(128, device="cuda", dtype=torch.float32)
target = torch.empty_like(source)
add_one[(1,)](source, target, N=128, BLOCK=128)
torch.testing.assert_close(target, source + 1)
torch.cuda.synchronize()
print("GPU 验收通过：", torch.cuda.get_device_name(0), "；BF16、vLLM CUDA 算子、Triton JIT 均通过。")
PY
  CUDA_VISIBLE_DEVICES=0 "$MAIN_ENV/bin/python" "$STATE/gpu-smoke.py"
fi

log '环境安装 / CPU 验收完成；没有启动训练、模型服务或正式评测。'
printf '激活命令：source %q\n' "$STATE/activate.sh"
if [[ "$MODE" != gpu-check ]]; then
  printf 'GPU 能否运行尚待空闲时验收：bash %q --project %q --gpu-check\n' "$SCRIPT" "$ROOT"
fi
printf '请勿用原 scripts/setup.sh 或 uv sync 覆盖此 cu129 环境；更新后请重新验收。\n'
