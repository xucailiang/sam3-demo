# SAM3 Demo — 后端服务

基于 FastAPI 的图像分割 API 服务，集成 Ultralytics SAM3 模型，支持文本提示、点击提示、边界框提示和图像范例四种分割方式。

## 架构

```
backend/
├── app/
│   ├── main.py              # FastAPI 应用入口，CORS / 异常处理 / 健康检查
│   ├── api/
│   │   ├── segment.py       # 单图分割路由 (/api/segment/*)
│   │   └── batch.py         # 批量分割路由 (/api/batch/*)
│   ├── models/
│   │   └── schemas.py       # Pydantic 数据模型
│   └── services/
│       ├── model_manager.py          # SAM3 模型管理器（单例，延迟加载）
│       └── segmentation_service.py   # 分割业务逻辑
├── requirements.txt
└── Dockerfile
```

### 核心设计

- **双推理接口**：SAM3SemanticPredictor 处理文本/边界框/图像范例（概念级分割），SAM 模型处理点击提示（实例级分割）
- **单例模型管理**：SAM3ModelManager 以单例模式管理模型生命周期，支持延迟加载和 GPU 半精度推理
- **统一错误处理**：全局异常中间件，400（参数错误）/ 500（推理失败）/ 503（GPU OOM）

### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/segment/text` | 文本提示分割 |
| POST | `/api/segment/points` | 点击提示分割 |
| POST | `/api/segment/boxes` | 边界框提示分割 |
| POST | `/api/segment/image-example` | 图像范例分割 |
| POST | `/api/batch/text` | 批量文本分割 |
| POST | `/api/batch/image-example` | 批量图像范例分割 |

## 环境要求

- Python 3.11+
- CUDA GPU（推荐，CPU 也可运行但较慢）

## 安装与启动

### 本地开发

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动服务（默认端口 8000）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `SAM3_MODEL_PATH` | 模型文件路径 | `sam3.pt`（自动下载） |

### Docker

```bash
docker build -t sam3-backend .
docker run --gpus all -p 8000:8000 sam3-backend
```

启动后访问 `http://localhost:8000/docs` 查看自动生成的 OpenAPI 文档。
