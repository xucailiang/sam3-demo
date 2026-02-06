# SAM3 Demo

基于 SAM3 (Segment Anything Model 3) 的图像分割演示应用，支持文本提示、点击提示、边界框提示和图像范例四种分割方式。

## 项目结构

```
├── frontend/          # React + TypeScript 前端应用
├── backend/           # FastAPI 后端服务
├── models/sam3/       # SAM3 模型文件
└── docker-compose.yml # Docker 编排配置
```

## 功能特性

- **四种分割模式**：文本提示、点击提示、边界框提示、图像范例
- **Canvas 交互**：支持点击添加前景/背景点、拖拽绘制边界框、掩码悬停高亮
- **批量处理**：多文件上传，统一提示批量分割，ZIP 打包导出
- **结果导出**：PNG 掩码、叠加图、JSON 数据

## 快速开始

### 使用 Docker Compose（推荐）

```bash
docker compose up --build
```

- 前端：http://localhost:3000
- 后端 API：http://localhost:8000
- API 文档：http://localhost:8000/docs

### 本地开发

#### 后端

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 前端

```bash
cd frontend
pnpm install
pnpm dev
```

## 环境要求

- Python 3.11+
- Node.js 20+
- CUDA GPU（推荐，CPU 也可运行但较慢）
- GPU 显存 16GB+（推荐）

## 技术栈

| 组件 | 技术 |
|------|------|
| 前端 | React, TypeScript, Vite, Canvas API |
| 后端 | FastAPI, Ultralytics SAM3, PyTorch |
| 部署 | Docker, Nginx |

## 相关文档

- [前端文档](frontend/README.md)
- [后端文档](backend/README.md)
- [SAM3 最佳实践](SAM3_BEST_PRACTICES.md)
