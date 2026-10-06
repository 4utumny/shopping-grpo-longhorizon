#!/usr/bin/env bash
# 为 README 的原始 Baseline 命令准备已经安装好的独立环境。
# 仅下载模型并生成 .venv-setup/activate-baseline.sh；不安装包、不运行自检、
# 不启动服务或评测、不改源码/清单/驱动/Conda base/全局配置。
set -Eeuo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '请用 bash 执行准备脚本，不要 source 此文件。\n' >&2
  return 1
fi

usage() {
  cat <<'HELP'
用法：bash prepare_readme_baseline.sh [--project 仓库目录]
需要已经安装的 .venv、ShopSimulator 环境和 .venv-setup 商品索引。
优先使用 HF-Mirror；失败后通过指定代理下载 Hugging Face 官方模型。
不会加载模型、调用 CUDA、启动服务或运行 Baseline。
环境变量：SHOPPING_DOWNLOAD_PROXY（默认 http://127.0.0.1:17897）
准备完成后，在运行 README 命令的各个终端中先执行：
  source .venv-setup/activate-baseline.sh
GPU 默认隐藏；约定使用时只在模型服务终端执行：
  export CUDA_VISIBLE_DEVICES=0
HELP
}

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
while (($#)); do
  case "$1" in
    --project)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      ROOT="$(cd -- "$2" && pwd -P)"
      shift 2
      ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

log() { printf '\n[baseline-prepare] %s\n' "$*"; }
die() { printf '\n[baseline-prepare] 错误：%s\n' "$*" >&2; exit 1; }
trap 'printf "\n[baseline-prepare] 第 %s 行失败；已有环境和下载断点均保留，未更改驱动。\n" "$LINENO" >&2' ERR

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die '请在 Linux x86_64 服务器上执行。'
[[ $EUID -ne 0 ]] || die '共享服务器上请使用安装环境的普通用户，不要使用 sudo。'
[[ -f "$ROOT/pyproject.toml" ]] || die '未找到项目根目录。'
MAIN_ENV="$ROOT/.venv"
SIM_ENV="$ROOT/environments/ShopSimulator/.venv-shopsim"
STATE="$ROOT/.venv-setup"
PROXY_URL="${SHOPPING_DOWNLOAD_PROXY:-http://127.0.0.1:17897}"
[[ "$PROXY_URL" == http://* || "$PROXY_URL" == https://* ]] || die '代理必须是 http:// 或 https:// 地址。'
for command_name in nice ionice flock awk sed mktemp mv; do
  command -v "$command_name" >/dev/null 2>&1 || die "找不到 $command_name；不会修改系统安装依赖。"
done

# 隐藏 GPU 仅影响本脚本及其下载子进程，不影响其他 SSH 会话或现有训练。
export CUDA_VISIBLE_DEVICES=-1 NVIDIA_VISIBLE_DEVICES=none
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
if [[ "${SHOPPING_BASELINE_LOW_PRIORITY:-0}" != 1 ]]; then
  SCRIPT="$(cd -- "$(dirname -- "$0")" && pwd -P)/$(basename -- "$0")"
  exec env SHOPPING_BASELINE_LOW_PRIORITY=1 nice -n 19 ionice -c 3 \
    bash "$SCRIPT" --project "$ROOT"
fi

umask 077
[[ ! -L "$STATE" && -f "$STATE/owner" && "$(cat "$STATE/owner")" == "$ROOT" ]] || die '现有环境属于其他路径；请在环境原来的项目目录执行。'
for env_kind in main sim; do
  if [[ "$env_kind" == main ]]; then
    env_path="$MAIN_ENV"; marker="$STATE/main-owner"
  else
    env_path="$SIM_ENV"; marker="$STATE/sim-owner"
  fi
  [[ ! -L "$env_path" && -x "$env_path/bin/python" && -f "$marker" && "$(cat "$marker")" == "$env_path" ]] || die "未找到此项目已安装的环境：$env_path"
done
[[ -x "$MAIN_ENV/bin/vllm" && -f "$STATE/products.sqlite3" ]] || die '需要先完成依赖和商品索引的安装。'
[[ -f "$ROOT/environments/ShopSimulator/shop_env/data/items_eval_train.json" ]] || die '现有商品数据链接不可用。'
exec 9>"$STATE/install.lock"
flock -n 9 || die '本项目已有安装或准备进程，暂不修改私有激活文件。'
[[ ! -L "$STATE/huggingface" && ! -L "$STATE/huggingface/hub" && ! -L "$STATE/activate-baseline.sh" ]] || die '私有缓存或激活文件不能是外部符号链接。'
export HF_HOME="$STATE/huggingface" HF_HUB_CACHE="$STATE/huggingface/hub"
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_ETAG_TIMEOUT=45 HF_HUB_DOWNLOAD_TIMEOUT=120

# 仅按已安装的 cu12 distribution 的文件归属生成动态库目录，
# 不把 cuTile 附带的 CUDA 13 编译工具目录混入运行时。
BASE_LIBRARIES="$("$MAIN_ENV/bin/python" - <<'PY'
import importlib.metadata as md
import pathlib
import sysconfig

site = pathlib.Path(sysconfig.get_paths()["purelib"])
paths = [site / "torch" / "lib"]
libraries = set()
for distribution in md.distributions():
    name = distribution.metadata.get("Name", "").lower().replace("_", "-")
    if name.startswith("nvidia-") and name.endswith("-cu12"):
        for file in distribution.files or []:
            if file.parent.name in {"lib", "lib64"}:
                libraries.add(pathlib.Path(distribution.locate_file(file)).parent.resolve())
paths += sorted(libraries)
print(":".join(dict.fromkeys(str(path) for path in paths if path.is_dir())))
PY
)"
[[ -n "$BASE_LIBRARIES" ]] || die '无法生成已安装 CUDA 12.9 包的动态库路径。'

download_model() {
  local endpoint="$1" route="$2"
  local -a download_env=(
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy
    -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN
    "HF_ENDPOINT=$endpoint" "NO_PROXY=localhost,127.0.0.1" "no_proxy=localhost,127.0.0.1"
  )
  if [[ "$route" == proxy ]]; then
    download_env+=("HTTP_PROXY=$PROXY_URL" "HTTPS_PROXY=$PROXY_URL"
      "http_proxy=$PROXY_URL" "https_proxy=$PROXY_URL")
  fi
  env "${download_env[@]}" "$MAIN_ENV/bin/python" - "$endpoint" "$HF_HUB_CACHE" <<'PY'
import pathlib
import sys
from huggingface_hub import snapshot_download

endpoint, cache = sys.argv[1:]
# 使用 main 让官方缓存系统写入 refs/main，原 README 的模型 ID 可直接读取缓存。
# 只下载文件；不导入 torch / transformers / vllm，也不加载模型。
snapshot = pathlib.Path(snapshot_download(
    repo_id="Qwen/Qwen3.5-2B",
    revision="main",
    cache_dir=cache,
    endpoint=endpoint,
    token=False,
    max_workers=1,
))
if not (snapshot / "config.json").is_file() or not list(snapshot.glob("*.safetensors")):
    raise SystemExit("下载目录缺少模型配置或权重；未生成新的激活文件，请保留断点重试。")
print("Qwen/Qwen3.5-2B 模型文件已缓存：", snapshot)
PY
}

log '预下载 Qwen/Qwen3.5-2B 到项目私有缓存；单并发、低 CPU/I/O 优先级，不加载模型。'
downloaded=0
for source_kind in domestic proxy; do
  if [[ "$source_kind" == domestic ]]; then
    endpoint=https://hf-mirror.com; attempts=2
  else
    endpoint=https://huggingface.co; attempts=6
  fi
  for ((attempt=1; attempt<=attempts; attempt++)); do
    log "下载来源：$endpoint（第 $attempt/$attempts 次；已有缓存和断点自动复用）。"
    if download_model "$endpoint" "$source_kind"; then
      downloaded=1
      break
    fi
    if ((attempt < attempts)); then sleep "$((attempt * 2))"; fi
  done
  ((downloaded == 0)) || break
done
((downloaded == 1)) || die '模型下载未完成；保留缓存，修复网络后重新执行此脚本。'

# 独立的新激活文件；不覆盖原安装器的 activate.sh。
activation_temporary="$(mktemp "$STATE/.activate-baseline.XXXXXX")"
trap 'rm -f -- "${activation_temporary-}"' EXIT
{
  printf '# 为 README Baseline 准备的项目私有激活文件；请 source。\n'
  printf 'if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then printf "请 source 激活文件。\\n" >&2; exit 1; fi\n'
  printf '_shopping_baseline_native_libraries=%q\n' "$BASE_LIBRARIES"
  printf '_shopping_baseline_compat=%q\n' "$STATE/cuda-compat/usr/local/cuda-12.9/compat"
  cat <<'ACTIVATE'
# 只读取驱动版本，不调用 CUDA 或探测 GPU；用卡是否获准由使用者约定。
_shopping_baseline_driver=$(awk '/^NVRM version:/ {for (i=1; i<=NF; i++) if ($i ~ /^[0-9]+\.[0-9]+\.[0-9]+$/) {print $i; exit}}' /proc/driver/nvidia/version)
_shopping_baseline_driver_major=${_shopping_baseline_driver%%.*}
if [[ "$_shopping_baseline_driver_major" == 570 ]]; then
  if [[ ! -f "$_shopping_baseline_compat/libcuda.so.1" ]]; then
    printf '缺少已解包的 CUDA 12.9 私有兼容库。\n' >&2
    return 1
  fi
  _shopping_baseline_libraries="$_shopping_baseline_compat:$_shopping_baseline_native_libraries"
elif [[ "$_shopping_baseline_driver_major" =~ ^[0-9]+$ ]] && (( _shopping_baseline_driver_major >= 575 )); then
  _shopping_baseline_libraries="$_shopping_baseline_native_libraries"
else
  printf '驱动版本不适用于当前准备配置：%s\n' "$_shopping_baseline_driver" >&2
  return 1
fi
if [[ -n "${VIRTUAL_ENV-}" ]] && declare -F deactivate >/dev/null; then deactivate; fi
ACTIVATE
  printf 'source %q\n' "$MAIN_ENV/bin/activate"
  cat <<'ACTIVATE'
_shopping_baseline_keys=(LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES SHOPPING_GRPO_ROOT SHOPPING_ENV_MANIFEST SHOPPING_ENVIRONMENT_VERSION SHOP_SEARCH_INDEX SHOPSIM_PORT LLM_PORT SHOPSIM_BASE_URL LLM_BASE_URL LLM_API_KEY SERVED_MODEL_NAME HF_HOME HF_HUB_CACHE HF_ENDPOINT HF_HUB_OFFLINE TRANSFORMERS_OFFLINE NO_PROXY no_proxy XDG_CACHE_HOME CUDA_CACHE_PATH TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE PYTHONHOME PYTHONPATH LD_PRELOAD)
declare -A _shopping_baseline_values=() _shopping_baseline_present=()
for _shopping_baseline_key in "${_shopping_baseline_keys[@]}"; do
  if [[ -v "$_shopping_baseline_key" ]]; then
    _shopping_baseline_present["$_shopping_baseline_key"]=1
    _shopping_baseline_values["$_shopping_baseline_key"]="${!_shopping_baseline_key}"
  fi
done
eval "$(declare -f deactivate | sed '1s/deactivate/_shopping_baseline_original_deactivate/')"
deactivate() {
  local key
  for key in "${_shopping_baseline_keys[@]}"; do
    if [[ "${_shopping_baseline_present[$key]-}" == 1 ]]; then
      printf -v "$key" '%s' "${_shopping_baseline_values[$key]}"
      export "$key"
    else
      unset "$key"
    fi
  done
  unset _shopping_baseline_keys _shopping_baseline_present _shopping_baseline_values
  _shopping_baseline_original_deactivate "$@"
  [[ "${1-}" == nondestructive ]] || unset -f _shopping_baseline_original_deactivate
}
export LD_LIBRARY_PATH="$_shopping_baseline_libraries"
ACTIVATE
  printf 'export SHOPPING_GRPO_ROOT=%q\n' "$ROOT"
  printf 'export SHOPPING_ENV_MANIFEST=%q\n' "$ROOT/data/environment.json"
  printf 'export SHOPPING_ENVIRONMENT_VERSION=shopsimulator-environment-v2.1\n'
  printf 'export SHOP_SEARCH_INDEX=%q\n' "$STATE/products.sqlite3"
  printf 'export SHOPSIM_PORT=5700 LLM_PORT=8000 SHOPSIM_BASE_URL=http://127.0.0.1:5700 LLM_BASE_URL=http://127.0.0.1:8000/v1 LLM_API_KEY=EMPTY SERVED_MODEL_NAME=shopping-agent\n'
  printf 'export HF_HOME=%q HF_HUB_CACHE=%q\n' "$HF_HOME" "$HF_HUB_CACHE"
  printf 'export HF_ENDPOINT=https://huggingface.co HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n'
  printf 'export NO_PROXY="localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}" no_proxy="localhost,127.0.0.1${no_proxy:+,$no_proxy}"\n'
  printf 'export XDG_CACHE_HOME=%q\n' "$STATE/cache"
  printf 'export CUDA_CACHE_PATH=%q TRITON_CACHE_DIR=%q TORCHINDUCTOR_CACHE_DIR=%q\n' \
    "$STATE/cuda-cache" "$STATE/triton-cache" "$STATE/torchinductor-cache"
  printf 'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1\n'
  printf 'unset PYTHONHOME PYTHONPATH LD_PRELOAD\n'
  printf 'unset _shopping_baseline_key _shopping_baseline_driver _shopping_baseline_driver_major _shopping_baseline_compat _shopping_baseline_native_libraries _shopping_baseline_libraries\n'
  printf 'printf "Baseline 环境已激活，模型使用本地缓存，GPU 默认隐藏。\\n"\n'
} > "$activation_temporary"
mv -- "$activation_temporary" "$STATE/activate-baseline.sh"
trap - EXIT
log '准备完成。正式验收仍由 README 的服务启动和 Baseline 命令完成，尚未验证 GPU 或训练。'
printf '各终端先执行：source %q\n' "$STATE/activate-baseline.sh"
printf 'ShopSimulator 可先启动：bash scripts/start_environment.sh\n'
printf '约定 GPU 空闲后，在模型服务终端执行：export CUDA_VISIBLE_DEVICES=0\n'
printf '随后执行：bash scripts/serve_model.sh Qwen/Qwen3.5-2B\n'
printf '服务就绪后，在第三个终端执行：bash scripts/baseline.sh\n'
