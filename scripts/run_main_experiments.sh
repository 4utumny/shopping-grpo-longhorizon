#!/usr/bin/env bash
# 按 study/beginner-guide.zh-CN.md 3.2–3.8 执行；需先安装 3.1 的依赖。
# GPU 轮询不是独占分配。flock 仅约束使用同一锁文件的启动器；没有调度器时，
# 无法阻止其他人同时启动任务。本脚本发现外部占用后只停止自己的进程。
# 用法：bash scripts/run_main_experiments.sh [--check-only|--prepare-only]
# --check-only 只做 CPU/只读检查，不等待、不建索引、不加载模型、不训练或评测。
# --prepare-only 仅补齐索引并核验 CPU 准备项，不启动商店、模型、训练或评测。
set -Eeuo pipefail
umask 077

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-run}"
if [[ "$MODE" == --help || "$MODE" == -h ]]; then
  sed -n '2,7p' "$0"
  exit 0
fi
[[ $# -le 1 && ( "$MODE" == run || "$MODE" == --check-only || "$MODE" == --prepare-only ) ]] || {
  echo "用法：bash scripts/run_main_experiments.sh [--check-only|--prepare-only]" >&2
  exit 2
}
# 学习指南 7.1 要求保留已经准备好的 CUDA 动态库、商品索引和模型缓存路径。
# 只 source 现有激活文件，不调用安装器，不改写原有配置。
export CUDA_VISIBLE_DEVICES=-1
if [[ -f "$ROOT/.venv-setup/activate-baseline.sh" ]]; then
  source "$ROOT/.venv-setup/activate-baseline.sh"
elif [[ -f "$ROOT/.venv-setup/activate.sh" ]]; then
  source "$ROOT/.venv-setup/activate.sh"
fi
for tool in python3 nvidia-smi flock setsid fuser ss tee; do
  command -v "$tool" >/dev/null || { echo "缺少命令：$tool" >&2; exit 1; }
done
PYTHON="$ROOT/.venv/bin/python"
SHOP_ROOT="$ROOT/environments/ShopSimulator/shop_env"
SHOP_PYTHON="$ROOT/environments/ShopSimulator/.venv-shopsim/bin/python"
[[ -x "$PYTHON" && -x "$SHOP_PYTHON" && -x "$ROOT/.venv/bin/vllm" ]] || {
  echo "依赖环境不完整，请先完成指南 3.1。" >&2; exit 1;
}

GPU_ID="${GPU_ID:-0}"
GPU_UUID="$(nvidia-smi -i "$GPU_ID" --query-gpu=uuid --format=csv,noheader)"
[[ "$GPU_UUID" =~ ^GPU-[a-fA-F0-9-]+$ ]] || { echo "GPU 标识无效" >&2; exit 1; }
RUN_TOKEN="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
RUN_DIR="$ROOT/exp_log/$(date +%Y%m%d_%H%M%S)_${RUN_TOKEN:0:8}"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/pipeline.log") 2>&1

# 固定使用同一张卡；不修改驱动、计算模式、功率或其他人的环境。
export CUDA_VISIBLE_DEVICES=-1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT/src"
export SHOPSIM_PORT=5700 LLM_PORT=8000 SHOPSIM_ENV_SLOTS=8 SHOP_MAX_STEPS=35
export SHOPSIM_BASE_URL="http://127.0.0.1:$SHOPSIM_PORT"
export LLM_BASE_URL="http://127.0.0.1:$LLM_PORT/v1" SERVED_MODEL_NAME=shopping-agent
export BASE_MODEL=Qwen/Qwen3.5-2B
export SFT_ADAPTER_DIR="$ROOT/outputs/models/sft-lora"
export SFT_MERGED_DIR="$ROOT/outputs/models/sft-merged"
export SHOP_SEARCH_INDEX="${SHOP_SEARCH_INDEX:-$SHOP_ROOT/search_engine/products.sqlite3}"
export SHOP_ENV_CONFIG="$SHOP_ROOT/configs/environment.json"
export SHOPPING_PIPELINE_RUN_TOKEN="$RUN_TOKEN"
export SHOPPING_PIPELINE_GPU_UUID="$GPU_UUID"
export SHOPPING_PIPELINE_ROOT="$ROOT"
export SHOPPING_PIPELINE_RUN_DIR="$RUN_DIR"
unset EVAL_OUTPUT_DIR RAY_ADDRESS RAY_NAMESPACE

# 启动前连续 120 秒满足全部条件；每 5 秒采样，忙碌或未知均重新计时。
readonly IDLE_SECONDS=120 POLL_SECONDS=5
HELPER="$RUN_DIR/.pipeline_helper.py"
cat > "$HELPER" <<'PY'
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(os.environ['SHOPPING_PIPELINE_ROOT'])
TOKEN = os.environ['SHOPPING_PIPELINE_RUN_TOKEN']
UUID = os.environ['SHOPPING_PIPELINE_GPU_UUID']
JOB_KEY = b'SHOPPING_PIPELINE_JOB_TOKEN='
# 仅容许已核实的常驻桌面程序，且必须是纯 G 类型；C、C+G、M、未知类型均阻止启动。
DESKTOP = {'/usr/lib/xorg/Xorg', '/usr/bin/gnome-shell',
           '/usr/local/sunlogin/bin/sunloginclient', '/usr/lib/firefox/firefox'}

def job_token(pid):
    try:
        env = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
        return next((x[len(JOB_KEY):].decode() for x in env if x.startswith(JOB_KEY)), '')
    except (OSError, UnicodeError):
        return ''

def owned(pid):
    return job_token(pid).startswith(TOKEN + ':')

def executable(pid):
    try:
        return os.readlink(f'/proc/{pid}/exe')
    except OSError:
        return ''

def quantity(gpu, path, unit):
    text = gpu.findtext(path, '').strip()
    match = re.fullmatch(r'(\d+)\s*' + re.escape(unit), text)
    if not match:
        raise ValueError(f'{path} 无可验证数值：{text!r}')
    return int(match[1])

def gpu_check(mode):
    try:
        result = subprocess.run(['nvidia-smi', '-i', UUID, '-q', '-x'],
                                capture_output=True, text=True, check=True, timeout=10)
        gpus = ET.fromstring(result.stdout).findall('gpu')
        if len(gpus) != 1 or gpus[0].findtext('uuid') != UUID:
            raise ValueError('GPU 查询结果不唯一或 UUID 不匹配')
        gpu = gpus[0]
        total = quantity(gpu, 'fb_memory_usage/total', 'MiB')
        used = quantity(gpu, 'fb_memory_usage/used', 'MiB')
        util = quantity(gpu, 'utilization/gpu_util', '%')
        mem_util = quantity(gpu, 'utilization/memory_util', '%')
        if total < 92160:
            raise ValueError('默认实验需要至少 90 GiB 显存的单卡')
        if gpu.findtext('mig_mode/current_mig', 'N/A') not in {'N/A', 'Disabled'}:
            raise ValueError('此启动器不接管 MIG GPU')
        processes = gpu.find('processes')
        if processes is None or (processes.text or '').strip():
            raise ValueError('无法完整读取 GPU 进程列表')
        blockers = []
        for process in processes:
            pid = process.findtext('pid', '')
            kind = process.findtext('type', '')
            name = process.findtext('process_name', '').split(' --', 1)[0]
            if process.tag != 'process_info' or not pid.isdigit():
                raise ValueError('GPU 进程记录不完整')
            if mode == 'guard' and owned(int(pid)):
                continue
            if kind == 'G' and name in DESKTOP:
                continue
            blockers.append(f'pid={pid},type={kind or "未知"}')
        minor = gpu.findtext('minor_number', '')
        if not minor.isdigit():
            raise ValueError('无法确认 GPU 设备节点')
        device = f'/dev/nvidia{minor}'
        # 补查已经打开设备、但尚未出现在 CUDA 进程表中的程序；绝不使用 fuser -k。
        handles = subprocess.run(['fuser', device], capture_output=True, text=True, timeout=10)
        if handles.returncode not in {0, 1} or 'Permission denied' in handles.stderr:
            raise ValueError('无法验证 GPU 设备句柄')
        for raw_pid in handles.stdout.split():
            if not raw_pid.isdigit():
                raise ValueError('设备句柄 PID 格式异常')
            pid = int(raw_pid)
            if mode == 'guard' and owned(pid):
                continue
            if executable(pid) not in DESKTOP:
                blockers.append(f'device_pid={pid}')
        if mode == 'idle' and (used > 1024 or util != 0 or mem_util != 0):
            blockers.append('显存>1024MiB或GPU/显存利用率非零')
        state = 'BUSY' if blockers else ('IDLE' if mode == 'idle' else 'SAFE')
        print(f'{state} GPU={UUID} memory={used}/{total}MiB util={util}% '
              f'mem_util={mem_util}% ' + ';'.join(blockers), flush=True)
        return 1 if blockers else 0
    except Exception as exc:
        print(f'UNKNOWN {type(exc).__name__}: {exc}', flush=True)
        return 2

def stop_job(token):
    # 以本次随机 token 识别进程，包括另建 session 的 Ray/vLLM 子进程；不按用户名或名称杀进程。
    def targets():
        result = []
        for item in Path('/proc').iterdir():
            if not item.name.isdigit() or job_token(item.name) != token:
                continue
            try:
                fields = (item / 'stat').read_text().rsplit(')', 1)[1].split()
                if fields[0] != 'Z':
                    result.append((int(item.name), fields[19], (item / 'comm').read_text().strip()))
            except OSError:
                continue
        return result
    for sig, grace in [(signal.SIGINT, 10), (signal.SIGTERM, 10), (signal.SIGKILL, 5)]:
        for pid, started, comm in targets():
            if comm == 'tee' and sig != signal.SIGKILL:
                continue  # 先让 tee 排空输出。
            try:
                fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
                if fields[19] == started and job_token(pid) == token:
                    os.kill(pid, sig)
            except (OSError, ProcessLookupError):
                pass
        deadline = time.monotonic() + grace
        while targets() and time.monotonic() < deadline:
            time.sleep(0.2)
        if not targets():
            return
    raise RuntimeError(f'本任务残留进程未退出：{token}')

def preflight():
    import pyarrow.parquet as parquet
    paths = ['data/sft/train.jsonl', 'data/sft/validation.jsonl',
             'data/grpo/train.parquet', 'data/grpo/validation.parquet',
             'data/evaluation/tasks.jsonl', 'data/environment.json',
             'environments/ShopSimulator/shop_env/data/items_eval_train.json']
    for name in paths:
        if not (ROOT / name).is_file():
            raise ValueError(f'缺少输入文件：{name}；请先完成 3.1')
    for name in ['sft-lora', 'sft-merged', 'grpo', 'grpo-merged']:
        path = ROOT / 'outputs/models' / name
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError(f'拒绝覆盖已有模型产物：{path}')
    for name in ['baseline', 'sft', 'grpo']:
        path = ROOT / 'outputs/evaluation' / name
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError(f'拒绝覆盖已有评测：{path}')
    for port in [5700, 8000]:
        with socket.socket() as sock:
            sock.bind(('0.0.0.0', port))
    expected_list = [int(json.loads(x)['task_id']) for x in
                     (ROOT / 'data/evaluation/tasks.jsonl').read_text().splitlines() if x.strip()]
    expected = set(expected_list)
    if len(expected_list) != 200 or len(expected) != 200:
        raise ValueError('评测集不是 200 个唯一任务')
    for name in ['train', 'validation']:
        sft_ids = {int(json.loads(x)['task_id']) for x in
                   (ROOT / f'data/sft/{name}.jsonl').read_text().splitlines() if x.strip()}
        grpo_ids = {int(row['extra_info']['task_id']) for row in
                    parquet.read_table(ROOT / f'data/grpo/{name}.parquet').to_pylist()}
        if expected & (sft_ids | grpo_ids):
            raise ValueError(f'{name} 数据与 Final-200 有重叠')
    manifest = json.loads((ROOT / 'data/environment.json').read_text())
    from shopping_grpo.environment.manifest import validate_manifest
    validate_manifest(manifest)
    products = ROOT / 'environments/ShopSimulator/shop_env/data/items_eval_train.json'
    with products.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != manifest['product_data_sha256']:
        raise ValueError('商品数据哈希与冻结 manifest 不符')
    required_versions = {'torch': '2.11.0', 'vllm': '0.25.1', 'verl': '0.8.0',
                         'ray': '2.56.1', 'transformers': '5.15.0.dev0',
                         'tensordict': '0.10.0', 'numpy': '2.2.6', 'swanlab': '0.9.1'}
    versions = {name: metadata.version(name) for name in [*required_versions, 'peft']}
    for name, required in required_versions.items():
        if versions[name].split('+', 1)[0] != required:
            raise ValueError(f'{name} 版本不符：{versions[name]}，需要 {required}')
    print('依赖版本：' + json.dumps(versions, ensure_ascii=False))
    shop_python = ROOT / 'environments/ShopSimulator/.venv-shopsim/bin/python'
    requirements = ROOT / 'environments/ShopSimulator/shop_env/requirements.txt'
    dependency_probe = '''
from importlib.metadata import version
from pathlib import Path
import sys
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line.strip() or line.startswith('#'):
        continue
    name, expected = line.strip().split('==', 1)
    actual = version(name)
    if actual != expected:
        raise RuntimeError(f'{name}: {actual} != {expected}')
print('ShopSimulator 固定依赖检查通过。')
'''
    subprocess.run([str(shop_python), '-B', '-c', dependency_probe, str(requirements)], check=True)
    # 这些检查只导入 CPU 模块、读取本地缓存，绝不初始化 CUDA 或加载模型权重。
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer
    import torch
    import vllm._C_stable_libtorch
    snapshot = Path(snapshot_download('Qwen/Qwen3.5-2B', local_files_only=True))
    weight_index = snapshot / 'model.safetensors.index.json'
    if weight_index.is_file():
        weights = set(json.loads(weight_index.read_text())['weight_map'].values())
    else:
        weights = {'model.safetensors'}
    for filename in weights:
        file = snapshot / filename
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f'基础模型缓存不完整：{file}')
    AutoConfig.from_pretrained('Qwen/Qwen3.5-2B', local_files_only=True)
    AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B', local_files_only=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU 预检查意外初始化了 CUDA')
    print(f'基础模型权重、配置和 tokenizer 缓存完整：{snapshot}')
    print('Torch/vLLM 原生库可导入；CUDA 未初始化。')
    spec = importlib.util.spec_from_file_location(
        'pipeline_runtime_check', ROOT / 'scripts/check_grpo_runtime.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    os.environ['SHOPPING_ENV_MANIFEST'] = str(ROOT / 'data/environment.json')
    runtime.validate_environment_contract()
    runtime.validate_transformers_revision()
    verify_index(allow_missing=True)
    print('基础预检查通过；正式运行仍会在启动商店前完成索引核验。')

def verify_index(allow_missing=False):
    index = Path(os.environ['SHOP_SEARCH_INDEX']).resolve()
    if not index.is_file():
        if allow_missing:
            print(f'索引待构建：{index}；正式运行会先构建再核验。')
            return
        raise ValueError(f'索引未生成：{index}')
    frozen = json.loads((ROOT / 'data/environment.json').read_text())
    with sqlite3.connect(index.as_uri() + '?mode=ro', uri=True) as conn:
        if conn.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError(f'SQLite 索引损坏：{index}')
        manifest = json.loads(conn.execute('SELECT payload FROM manifest').fetchone()[0])
        count = conn.execute('SELECT count(*) FROM products').fetchone()[0]
        if manifest.get('product_data_sha256') != frozen['product_data_sha256']:
            raise ValueError('索引商品哈希与冻结 manifest 不符')
        if manifest.get('search_version') != frozen['search']['version']:
            raise ValueError('索引 search_version 不符')
        if manifest.get('field_weights') != frozen['search']['field_weights']:
            raise ValueError('索引搜索权重与原仓库不同')
        if count <= 0 or count != manifest.get('product_count'):
            raise ValueError('索引商品数量与 manifest 不符')
        conn.execute("SELECT asin FROM products WHERE products MATCH '手机' LIMIT 1").fetchall()
    print(f'索引核验通过：{index}；商品数={count}；SQLite/FTS5/哈希/搜索权重一致。')

def ready(kind, token):
    port = 5700 if kind == 'shop' else 8000
    listeners = subprocess.check_output(['ss', '-ltnp', f'sport = :{port}'], text=True)
    pids = re.findall(r'pid=(\d+)', listeners)
    if not pids or not all(job_token(pid) == token for pid in pids):
        return 1
    url = f'http://127.0.0.1:{port}/' + ('api/shop_agent' if kind == 'shop' else 'v1/models')
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            data = json.load(response)
        return 0 if kind == 'model' and any(x.get('id') == 'shopping-agent'
                                           for x in data.get('data', [])) else 1
    except urllib.error.HTTPError as exc:
        return 0 if kind == 'shop' and exc.code == 405 else 1
    except (OSError, ValueError):
        return 1

def verify_model(name):
    path = ROOT / 'outputs/models' / name
    weights = ['model.safetensors', 'model.safetensors.index.json',
               'pytorch_model.bin', 'pytorch_model.bin.index.json']
    if not (path / 'config.json').is_file() or not (path / 'tokenizer_config.json').is_file():
        raise ValueError(f'{path} 缺少 config/tokenizer 配置')
    if not any((path / item).is_file() for item in weights):
        raise ValueError(f'{path} 缺少权重')
    print(f'模型产物检查通过：{path}')

def verify_eval(name):
    from shopping_grpo.evaluation.summary import summarize_trajectories
    path = ROOT / 'outputs/evaluation' / name
    expected = [json.loads(x)['task_id'] for x in
                (ROOT / 'data/evaluation/tasks.jsonl').read_text().splitlines() if x.strip()]
    rows = [json.loads(x) for x in (path / 'trajectories.jsonl').read_text().splitlines() if x.strip()]
    summary = json.loads((path / 'summary.json').read_text())
    actual = summarize_trajectories(expected, rows)
    if len(rows) != 200 or {x['task_id'] for x in rows} != set(expected):
        raise ValueError(f'{name} 轨迹没有完整覆盖同一批 200 题')
    for key in ['expected_tasks', 'completed_tasks', 'missing_tasks',
                'strict_successes', 'strict_success_rate', 'reward_contract']:
        if summary.get(key) != actual[key]:
            raise ValueError(f'{name} 汇总字段 {key} 与轨迹不符')
    if not (path / 'report.html').is_file():
        raise ValueError(f'{name} 缺少 report.html')
    print(f'{name}: completed_tasks=200 strict_success_rate={actual["strict_success_rate"]:.3%}')

def choose_checkpoint():
    # 只读 GRPO 验证指标；绝不读取 Final-200 的结果挑 checkpoint。
    metric = 'val-core/shopsimulator/reward/mean@1'
    folder = ROOT / 'outputs/models/grpo'
    candidates = []
    with (folder / 'training_diagnostics.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            step = int(row['global_step'])
            value = row.get('metrics', {}).get(metric)
            actor = folder / f'global_step_{step}' / 'actor'
            if row.get('event') == 'optimizer_step' and value is not None and actor.is_dir():
                score = float(value)
                if math.isfinite(score):
                    candidates.append((score, -step, actor))
    if not candidates:
        raise ValueError(f'没有同时具备 checkpoint 和 {metric} 验证指标的记录；拒绝猜测')
    score, negative_step, actor = max(candidates)
    selection = {'metric': metric, 'score': score, 'step': -negative_step,
                 'actor': str(actor), 'tie_break': 'earliest_step'}
    destination = Path(os.environ['SHOPPING_PIPELINE_RUN_DIR'])
    (destination / 'checkpoint_selection.json').write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + '\n')
    (destination / '.selected_actor').write_text(str(actor) + '\n')
    print(json.dumps(selection, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    action, *args = sys.argv[1:]
    if action == 'gpu':
        sys.exit(gpu_check(args[0]))
    if action == 'ready':
        sys.exit(ready(*args))
    {'stop': stop_job, 'preflight': preflight, 'index': verify_index, 'model': verify_model,
     'eval': verify_eval, 'choose': choose_checkpoint}[action](*args)
PY

MAIN_PID=$$
WATCH_PID=0 SHOP_PID=0 MODEL_PID=0
SHOP_TOKEN="" MODEL_TOKEN="" ACTIVE_JOB_TOKEN=""
GPU_ACTIVE="$RUN_DIR/.gpu_active"
cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP
  set +e
  rm -f "$GPU_ACTIVE"
  if (( WATCH_PID > 0 )); then kill "$WATCH_PID" 2>/dev/null; wait "$WATCH_PID" 2>/dev/null; fi
  for token in "$ACTIVE_JOB_TOKEN" "$MODEL_TOKEN" "$SHOP_TOKEN"; do
    if [[ -n "$token" ]] && ! "$PYTHON" "$HELPER" stop "$token"; then
      (( status == 0 )) && status=1
    fi
  done
  printf '[%s] 流水线退出，exit_code=%s；日志：%s\n' "$(date -Is)" "$status" "$RUN_DIR"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

if [[ "$MODE" == --check-only ]]; then
  "$PYTHON" "$HELPER" preflight
  "$PYTHON" "$HELPER" gpu idle || true
  exit 0
fi

# 同 GPU 的本启动器互斥；整个流水线期间持有锁。绝不自动解除其他人的锁。
LOCK_PATH="${TMPDIR:-/tmp}/shopping-grpo-${GPU_UUID}.lock"
exec 9>"$LOCK_PATH"
flock -n 9 || { log "同 GPU 的启动器锁已被占用：$LOCK_PATH"; exit 1; }

wait_for_idle() {
  local start=-1 now
  log "等待 GPU 连续 ${IDLE_SECONDS}s 空闲；计算进程即使利用率为 0% 也视为占用。"
  while true; do
    assert_services_alive
    if "$PYTHON" "$HELPER" gpu idle | tee -a "$RUN_DIR/00_gpu_wait.log"; then
      now=$SECONDS
      (( start < 0 )) && start=$now
      if (( now - start >= IDLE_SECONDS )); then
        log "已完成连续空闲确认。"
        return
      fi
    else
      start=-1
    fi
    sleep "$POLL_SECONDS"
  done
}

assert_services_alive() {
  if (( WATCH_PID > 0 )) && ! kill -0 "$WATCH_PID" 2>/dev/null; then
    log "GPU 监测进程异常退出。"; return 1
  fi
  if (( SHOP_PID > 0 )) && ! kill -0 "$SHOP_PID" 2>/dev/null; then
    log "商店服务异常退出。"; return 1
  fi
  if (( MODEL_PID > 0 )) && ! kill -0 "$MODEL_PID" 2>/dev/null; then
    log "模型服务异常退出。"; return 1
  fi
}

start_job() {
  local label=$1 token=$2
  shift 2
  # 独立 session + 随机任务 token；pipefail 保留真正的命令失败，不被 tee 的成功掩盖。
  SHOPPING_PIPELINE_JOB_TOKEN="$token" setsid --wait bash -c \
    'set -euo pipefail; stage_log=$1; shift; "$@" 2>&1 | tee -a "$stage_log"' \
    bash "$RUN_DIR/$label.log" "$@" 9>&- &
  STARTED_PID=$!
}

run_step() {
  local label=$1 pid status=0
  shift
  ACTIVE_JOB_TOKEN="$RUN_TOKEN:$label"
  log "开始 $label：$*"
  start_job "$label" "$ACTIVE_JOB_TOKEN" "$@"
  pid=$STARTED_PID
  while kill -0 "$pid" 2>/dev/null; do assert_services_alive; sleep 2; done
  wait "$pid" || status=$?
  "$PYTHON" "$HELPER" stop "$ACTIVE_JOB_TOKEN"
  ACTIVE_JOB_TOKEN=""
  (( status == 0 )) || { log "$label 失败，exit_code=$status；停止后续阶段。"; return "$status"; }
  log "完成 $label"
}

wait_service() {
  local kind=$1 pid=$2 token=$3 deadline=$((SECONDS + 3600))
  while (( SECONDS < deadline )); do
    assert_services_alive
    kill -0 "$pid" 2>/dev/null || { log "$kind 服务启动失败。"; return 1; }
    if "$PYTHON" "$HELPER" ready "$kind" "$token"; then log "$kind 服务就绪。"; return; fi
    sleep 2
  done
  log "$kind 服务在 3600s 内未就绪。"; return 1
}

start_model() {
  local label=$1 model=$2
  wait_for_idle
  touch "$GPU_ACTIVE"
  MODEL_TOKEN="$RUN_TOKEN:$label"
  start_job "$label" "$MODEL_TOKEN" bash scripts/serve_model.sh "$model"
  MODEL_PID=$STARTED_PID
  wait_service model "$MODEL_PID" "$MODEL_TOKEN"
}

stop_model() {
  log "停止本脚本的模型服务并等待子进程退出。"
  rm -f "$GPU_ACTIVE"
  "$PYTHON" "$HELPER" stop "$MODEL_TOKEN"
  wait "$MODEL_PID" 2>/dev/null || true
  MODEL_PID=0 MODEL_TOKEN=""
}

prepare_inputs() {
  run_step 01_preflight "$PYTHON" "$HELPER" preflight
  run_step 01_verl_patch_check "$PYTHON" scripts/apply_verl_dynamic_sampling_patch.py --check
  if [[ ! -f "$SHOP_SEARCH_INDEX" ]]; then
    run_step 01_build_index "$SHOP_PYTHON" "$SHOP_ROOT/scripts/build_index.py" --output "$SHOP_SEARCH_INDEX"
  fi
  run_step 01_index_verify "$PYTHON" "$HELPER" index
}

if [[ "$MODE" == --prepare-only ]]; then
  prepare_inputs
  log "CPU 准备完成；未启动商店、模型、训练或正式评测。"
  exit 0
fi

wait_for_idle
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
# 从此保持低频监测；GPU 阶段发现外部进程或查询不确定时，只终止自己的流水线。
(
  while true; do
    if [[ -e "$GPU_ACTIVE" ]]; then
      if ! "$PYTHON" "$HELPER" gpu guard >> "$RUN_DIR/00_gpu_watch.log" 2>&1; then
        if [[ -e "$GPU_ACTIVE" ]]; then
          log "GPU 出现外部占用或无法确认状态；停止自己的实验，详见 00_gpu_watch.log。"
          kill -TERM "$MAIN_PID"
          exit 1
        fi
      fi
    fi
    sleep 2
  done
) 9>&- &
WATCH_PID=$!

prepare_inputs
RAY_TMP="$(mktemp -d "${TMPDIR:-/tmp}/shopping-grpo-ray.XXXXXXXX")"
# 仅隔离 Ray 集群和临时目录；不追加 Hydra overrides，不覆盖任何原仓库资源/训练参数。
export RAY_ADDRESS=local RAY_TMPDIR="$RAY_TMP"

log "3.2 启动本实验的商店。"
SHOP_TOKEN="$RUN_TOKEN:02_shop"
start_job 02_shop "$SHOP_TOKEN" bash scripts/start_environment.sh
SHOP_PID=$STARTED_PID
wait_service shop "$SHOP_PID" "$SHOP_TOKEN"

log "3.3 Baseline。"
start_model 03_baseline_serve "$BASE_MODEL"
run_step 03_baseline_eval bash scripts/baseline.sh
run_step 03_baseline_verify "$PYTHON" "$HELPER" eval baseline
stop_model

log "3.4 SFT 训练及自动合并。"
wait_for_idle
touch "$GPU_ACTIVE"
run_step 04_sft_train_and_merge bash scripts/sft.sh
rm -f "$GPU_ACTIVE"
run_step 04_sft_verify "$PYTHON" "$HELPER" model sft-merged

log "3.5 SFT 评测。"
start_model 05_sft_serve outputs/models/sft-merged
run_step 05_sft_eval bash scripts/evaluate.sh sft
run_step 05_sft_verify "$PYTHON" "$HELPER" eval sft
stop_model

log "3.6 GRPO dry-run 后按默认配置训练 500 步。"
run_step 06_grpo_dry_run bash scripts/grpo.sh --dry-run
wait_for_idle
touch "$GPU_ACTIVE"
run_step 06_grpo_train bash scripts/grpo.sh
rm -f "$GPU_ACTIVE"

log "3.7 按 GRPO 验证集平均 Reward 选择已保存的 checkpoint，同分选较早步数。"
run_step 07_select_checkpoint "$PYTHON" "$HELPER" choose
ACTOR_CHECKPOINT="$(cat "$RUN_DIR/.selected_actor")"
test -d "$ACTOR_CHECKPOINT"
wait_for_idle
touch "$GPU_ACTIVE"
run_step 07_grpo_export bash scripts/export_grpo.sh "$ACTOR_CHECKPOINT" outputs/models/grpo-merged
rm -f "$GPU_ACTIVE"
run_step 07_grpo_verify "$PYTHON" "$HELPER" model grpo-merged

log "3.8 GRPO 评测。"
start_model 08_grpo_serve outputs/models/grpo-merged
run_step 08_grpo_eval bash scripts/evaluate.sh grpo
run_step 08_grpo_verify "$PYTHON" "$HELPER" eval grpo
stop_model

log "3.2–3.8 全部完成；三份评测见 outputs/evaluation/{baseline,sft,grpo}/；日志：$RUN_DIR"
