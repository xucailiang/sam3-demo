#!/usr/bin/env bash
set -e

# SAM3 Demo 启动脚本 — 同时启动后端和前端开发服务器

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

cleanup() {
  echo ""
  echo "正在停止服务..."
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
  wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
  echo "已停止"
}
trap cleanup EXIT INT TERM

# ---------- 后端 ----------
echo "▶ 启动后端 (http://localhost:8000) ..."

if [ ! -d "$ROOT_DIR/backend/.venv" ]; then
  echo "  创建虚拟环境..."
  uv venv "$ROOT_DIR/backend/.venv"
fi

(
  cd "$ROOT_DIR/backend"
  source .venv/bin/activate
  uv pip install -q -r requirements.txt
  # 禁用 Ultralytics 运行时自动 pip install（避免调用系统 pip 报错）
  export YOLO_AUTOINSTALL=false
  # 指定 SAM3 模型文件路径（通过 huggingface-cli download facebook/sam3 下载）
  export SAM3_MODEL_PATH="${SAM3_MODEL_PATH:-$ROOT_DIR/models/sam3/sam3.pt}"
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
) &
BACKEND_PID=$!

# ---------- 前端 ----------
echo "▶ 启动前端 (http://localhost:3000) ..."

if ! command -v pnpm &>/dev/null; then
  echo "  安装 pnpm..."
  npm install -g pnpm
fi

(cd "$ROOT_DIR/frontend" && pnpm install --frozen-lockfile && pnpm dev) &
FRONTEND_PID=$!

echo ""
echo "✔ 后端: http://localhost:8000  (API 文档: http://localhost:8000/docs)"
echo "✔ 前端: http://localhost:3000"
echo "按 Ctrl+C 停止所有服务"

wait
