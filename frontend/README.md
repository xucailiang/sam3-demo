# SAM3 Demo — 前端应用

基于 React + TypeScript 的图像分割交互界面，通过 Canvas 画布实现图像预览、提示交互和分割结果可视化。

## 架构

```
frontend/src/
├── App.tsx                    # 主应用组件，状态管理与组件组装
├── main.tsx                   # 应用入口
├── api/
│   └── client.ts              # 后端 API 客户端（axios）
├── components/
│   ├── ImageCanvas.tsx        # Canvas 画布：图像渲染、掩码叠加、点击/拖拽交互
│   ├── PromptPanel.tsx        # 提示面板：模式切换、文本输入、点/框管理
│   ├── ResultPanel.tsx        # 结果面板：统计信息、透明度控制、导出按钮
│   ├── BatchPanel.tsx         # 批量处理面板：多文件上传、进度条、缩略图网格
│   └── ErrorBanner.tsx        # 错误提示组件
├── hooks/
│   ├── useSegmentation.ts     # 单图分割状态管理 Hook
│   └── useBatchSegmentation.ts # 批量分割状态管理 Hook
├── types/
│   └── index.ts               # TypeScript 类型定义
└── utils/
    ├── validation.ts          # 文件扩展名验证
    ├── canvas.ts              # 图像缩放计算、掩码颜色分配
    ├── promptState.ts         # 提示状态管理（清除操作）
    ├── displayState.ts        # 显示状态管理（透明度、可见性切换）
    ├── resultDisplay.ts       # 结果显示（标签、置信度格式化）
    ├── statistics.ts          # 统计信息计算
    ├── boxDisplay.ts          # 边界框坐标显示
    ├── batchProgress.ts       # 批量进度计算
    ├── batchResults.ts        # 批量结果处理
    └── export.ts              # 导出功能（PNG 掩码、叠加图、JSON、ZIP）
```

### 核心设计

- **Canvas 交互**：支持点击添加前景/背景点（左键/右键）、拖拽绘制边界框、掩码悬停高亮
- **四种分割模式**：文本提示、点击提示、边界框提示、图像范例
- **批量处理**：多文件上传，统一提示批量分割，缩略图网格展示，ZIP 打包导出
- **属性测试**：使用 fast-check 对核心工具函数进行属性测试，验证正确性不变量

## 环境要求

- Node.js 20+
- pnpm（推荐）或 npm

## 安装与启动

### 本地开发

```bash
# 安装依赖
pnpm install

# 启动开发服务器（端口 3000，API 请求代理到 localhost:8000）
pnpm dev
```

开发模式下 Vite 会将 `/api` 请求代理到后端 `http://localhost:8000`，需先启动后端服务。

### 运行测试

```bash
pnpm test
```

### 构建生产版本

```bash
pnpm build
```

构建产物输出到 `dist/` 目录。

### Docker

```bash
docker build -t sam3-frontend .
docker run -p 3000:80 sam3-frontend
```

生产环境使用 Nginx 托管静态文件，并将 `/api` 请求反向代理到后端服务。

## 整体启动（Docker Compose）

在项目根目录运行：

```bash
docker compose up --build
```

- 前端：`http://localhost:3000`
- 后端：`http://localhost:8000`（API 文档：`http://localhost:8000/docs`）
