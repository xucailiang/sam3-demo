# 实现计划: SAM3 演示项目

## 概述

本实现计划基于已批准的需求和设计文档，将 SAM3 图像分割演示项目分解为可执行的编码任务。项目采用前后端分离架构，后端使用 Python FastAPI + Ultralytics SAM3，前端使用 React + TypeScript。

## 任务列表

- [x] 1. 项目初始化和基础设施搭建
  - [x] 1.1 创建后端项目结构
    - 创建 `backend/` 目录结构：`app/`, `app/api/`, `app/services/`, `app/models/`
    - 创建 `requirements.txt` 包含 fastapi, uvicorn, ultralytics, python-multipart, pydantic
    - 创建 `app/main.py` FastAPI 应用入口，配置 CORS 中间件
    - _Requirements: 6.1, 6.5_

  - [x] 1.2 创建前端项目结构
    - 使用 Vite 创建 React + TypeScript 项目
    - 安装依赖：react, typescript, axios
    - 创建目录结构：`src/components/`, `src/hooks/`, `src/types/`, `src/utils/`
    - _Requirements: 1.2, 1.3_

  - [x] 1.3 创建 Docker 配置
    - 创建 `backend/Dockerfile` 用于后端服务
    - 创建 `frontend/Dockerfile` 用于前端构建
    - 创建 `docker-compose.yml` 编排前后端服务
    - _Requirements: 6.6_

- [x] 2. 后端数据模型和核心服务
  - [x] 2.1 实现数据模型
    - 创建 `app/models/schemas.py` 定义 Pydantic 模型
    - 实现 PointPrompt, BoxPrompt, MaskData, SegmentationResult 等模型
    - 实现 SegmentationResponse, BatchSegmentationResponse 响应模型
    - _Requirements: 3.1, 4.2, 6.1_

  - [x] 2.2 实现 SAM3 模型管理器
    - 支持配置模型下载或者存放路径
    - 创建 `app/services/model_manager.py`
    - 实现 SAM3ModelManager 单例类，负责模型加载和生命周期管理
    - 实现 `get_semantic_predictor()` 和 `get_point_model()` 方法
    - 实现启动时预加载模型到 GPU
    - _Requirements: 6.6_

  - [x] 2.3 实现分割服务核心逻辑
    - 创建 `app/services/segmentation_service.py`
    - 实现 `segment_with_text()` 文本提示分割
    - 实现 `segment_with_points()` 点击提示分割
    - 实现 `segment_with_boxes()` 边界框提示分割
    - _Requirements: 2.1, 3.4, 4.3_

  - [ ]* 2.4 编写属性测试：文本提示解析
    - **Property 3: 文本提示解析**
    - 使用 hypothesis 测试逗号分隔文本的解析逻辑
    - **Validates: Requirements 2.2**

- [x] 3. 后端 API 路由实现
  - [x] 3.1 实现文本分割 API
    - 创建 `app/api/segment.py` 路由模块
    - 实现 `POST /api/segment/text` 端点
    - 处理 multipart/form-data 请求，解析图像和文本提示
    - _Requirements: 2.1, 2.2_

  - [x] 3.2 实现点击分割 API
    - 实现 `POST /api/segment/points` 端点
    - 解析点坐标数组和标签数组
    - _Requirements: 3.4_

  - [x] 3.3 实现边界框分割 API
    - 实现 `POST /api/segment/boxes` 端点
    - 解析边界框坐标数组
    - _Requirements: 4.3_

  - [x] 3.4 实现健康检查 API
    - 实现 `GET /api/health` 端点
    - 返回服务状态和模型加载状态
    - _Requirements: 6.1_

  - [ ]* 3.5 编写属性测试：错误请求返回 400
    - **Property 13: 错误请求返回 400**
    - 使用 hypothesis 生成无效请求数据测试错误处理
    - **Validates: Requirements 6.3**

- [x] 4. 检查点 - 后端核心功能验证
  - 确保所有测试通过，如有问题请询问用户。

- [x] 5. 前端类型定义和工具函数
  - [x] 5.1 定义 TypeScript 类型
    - 创建 `src/types/index.ts`
    - 定义 InteractionMode, PointPrompt, BoxPrompt, MaskData, SegmentationResult 等类型
    - _Requirements: 3.1, 4.2_

  - [x] 5.2 实现文件验证工具
    - 创建 `src/utils/validation.ts`
    - 实现 `validateFileExtension()` 函数验证 jpg, jpeg, png, webp 扩展名
    - _Requirements: 1.1_

  - [ ]* 5.3 编写属性测试：文件扩展名验证
    - **Property 1: 文件扩展名验证**
    - 使用 fast-check 测试文件扩展名验证逻辑
    - **Validates: Requirements 1.1**

  - [x] 5.4 实现图像缩放计算工具
    - 创建 `src/utils/canvas.ts`
    - 实现 `calculateScaledSize()` 函数保持宽高比缩放
    - _Requirements: 1.5_

  - [x] 5.5 编写属性测试：图像缩放保持宽高比
    - **Property 2: 图像缩放保持宽高比**
    - 使用 fast-check 测试缩放计算逻辑
    - **Validates: Requirements 1.5**

  - [x] 5.6 实现颜色分配工具
    - 在 `src/utils/canvas.ts` 中实现 `assignMaskColors()` 函数
    - 为多个掩码分配不同颜色
    - _Requirements: 5.1_

  - [x] 5.7 编写属性测试：掩码颜色唯一性
    - **Property 9: 掩码颜色唯一性**
    - 使用 fast-check 测试颜色分配唯一性
    - **Validates: Requirements 5.1**

- [x] 6. 前端 API 客户端
  - [x] 6.1 实现 API 客户端
    - 创建 `src/api/client.ts`
    - 实现 `segmentWithText()`, `segmentWithPoints()`, `segmentWithBoxes()` 函数
    - 处理 FormData 构建和响应解析
    - _Requirements: 2.1, 3.4, 4.3_

  - [x] 6.2 实现 useSegmentation Hook
    - 创建 `src/hooks/useSegmentation.ts`
    - 管理分割请求状态、加载状态和错误状态
    - 实现 `segmentWithText`, `segmentWithPoints`, `segmentWithBoxes` 方法
    - _Requirements: 2.3, 3.4, 4.3_

- [-] 7. 前端核心组件实现
  - [x] 7.1 实现 ImageCanvas 组件
    - 创建 `src/components/ImageCanvas.tsx`
    - 实现图像渲染、掩码叠加渲染
    - 实现点击事件处理（左键前景点、右键背景点）
    - 实现边界框拖拽绘制
    - _Requirements: 1.2, 3.1, 3.2, 3.3, 4.1_

  - [x] 7.2 编写属性测试：点击坐标记录
    - **Property 5: 点击坐标记录**
    - 测试坐标转换逻辑的正确性
    - **Validates: Requirements 3.1**

  - [x] 7.3 编写属性测试：边界框坐标记录
    - **Property 7: 边界框坐标记录**
    - 测试边界框坐标规范化逻辑
    - **Validates: Requirements 4.2**

  - [x] 7.4 实现 PromptPanel 组件
    - 创建 `src/components/PromptPanel.tsx`
    - 实现模式切换（文本/点击/边界框）
    - 实现文本输入框和提交按钮
    - 实现清除点/清除框按钮
    - _Requirements: 2.1, 3.5, 4.4_

  - [x] 7.5 编写属性测试：清除操作重置状态
    - **Property 6: 清除操作重置状态**
    - 测试清除操作后状态为空数组
    - **Validates: Requirements 3.5, 4.4**

  - [x] 7.6 实现 ResultPanel 组件
    - 创建 `src/components/ResultPanel.tsx`
    - 显示检测统计信息（对象数量、处理耗时）
    - 实现透明度滑块控制
    - 实现掩码显示/隐藏切换
    - _Requirements: 5.3, 5.4, 5.5_

  - [x] 7.7 编写属性测试：掩码可见性切换和透明度范围
    - **Property 10: 掩码可见性切换**
    - **Property 11: 透明度值范围约束**
    - 测试状态切换和值范围约束
    - **Validates: Requirements 5.3, 5.4**

  - [x] 7.8 编写属性测试：统计信息完整性
    - **Property 12: 统计信息完整性**
    - 测试统计信息包含必要字段
    - **Validates: Requirements 5.5**

  - [x] 7.9 编写属性测试：结果显示包含标签和置信度
    - **Property 4: 结果显示包含标签和置信度**
    - 测试渲染输出包含标签和置信度
    - **Validates: Requirements 2.4**

- [x] 8. 检查点 - 前端核心功能验证
  - 确保所有测试通过，如有问题请询问用户。

- [x] 9. 前端主应用组装
  - [x] 9.1 实现 App 主组件
    - 创建 `src/App.tsx`
    - 组装 ImageCanvas, PromptPanel, ResultPanel 组件
    - 实现应用状态管理
    - _Requirements: 1.2, 1.3, 2.3, 5.2_

  - [x] 9.2 实现图像上传功能
    - 在 App 中实现文件选择和验证
    - 实现图像预览渲染
    - 实现分割按钮状态控制
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [x] 9.3 实现掩码悬停高亮
    - 在 ImageCanvas 中实现鼠标悬停检测
    - 高亮显示悬停的掩码并显示标签
    - _Requirements: 5.2_

- [x] 10. 导出功能实现
  - [x] 10.1 实现导出工具函数
    - 创建 `src/utils/export.ts`
    - 实现 `exportMaskAsPNG()` 导出二值掩码
    - 实现 `exportOverlayAsPNG()` 导出叠加图
    - 实现 `exportResultAsJSON()` 导出 JSON 数据
    - _Requirements: 9.1, 9.2, 9.3_

  - [x] 10.2 编写属性测试：JSON 导出 Round-Trip
    - **Property 14: JSON 导出 Round-Trip**
    - 测试 JSON 序列化和反序列化的一致性
    - **Validates: Requirements 9.3**

  - [x] 10.3 在 ResultPanel 中集成导出按钮
    - 添加导出掩码、导出叠加图、导出 JSON 按钮
    - 连接导出工具函数
    - _Requirements: 9.1, 9.2, 9.3_

- [x] 11. 图像范例分割功能
  - [x] 11.1 后端实现图像范例分割 API
    - 在 `app/services/segmentation_service.py` 中实现 `segment_with_image_example()`
    - 在 `app/api/segment.py` 中实现 `POST /api/segment/image-example` 端点
    - _Requirements: 7.1, 7.2_

  - [x] 11.2 前端实现图像范例模式
    - 在 PromptPanel 中添加图像范例模式选项
    - 实现示例图像上传和示例区域选择
    - 在 useSegmentation Hook 中添加 `segmentWithImageExample` 方法
    - _Requirements: 7.1, 7.3, 7.4_

- [x] 12. 批量处理功能
  - [x] 12.1 后端实现批量分割 API
    - 在 `app/services/segmentation_service.py` 中实现 `batch_segment_with_text()` 和 `batch_segment_with_image_example()`
    - 在 `app/api/batch.py` 中实现 `POST /api/batch/text` 和 `POST /api/batch/image-example` 端点
    - _Requirements: 8.2, 8.3_

  - [x] 12.2 前端实现批量处理面板
    - 创建 `src/components/BatchPanel.tsx`
    - 实现多文件上传和文件列表显示
    - 实现进度条和当前处理图像显示
    - _Requirements: 8.1, 8.4_

  - [ ]* 12.3 编写属性测试：批量文件验证
    - **Property 16: 批量文件验证**
    - 测试批量文件扩展名验证逻辑
    - **Validates: Requirements 8.1**

  - [x] 12.4 实现 useBatchSegmentation Hook
    - 创建 `src/hooks/useBatchSegmentation.ts`
    - 管理批量处理状态和进度
    - 实现 `batchSegmentWithText` 和 `batchSegmentWithImageExample` 方法
    - _Requirements: 8.2, 8.3, 8.4_

  - [x] 12.5 编写属性测试：批量处理进度一致性
    - **Property 17: 批量处理进度一致性**
    - 测试进度信息的一致性约束
    - **Validates: Requirements 8.4**

  - [x] 12.6 编写属性测试：批量结果数量一致性
    - **Property 18: 批量结果数量一致性**
    - 测试返回结果数量与输入图像数量一致
    - **Validates: Requirements 8.2, 8.3**

  - [x] 12.7 实现批量结果展示
    - 在 BatchPanel 中实现缩略图网格展示
    - 实现点击缩略图查看详细结果
    - _Requirements: 8.5, 8.6_

  - [x] 12.8 实现批量导出功能
    - 实现 ZIP 打包导出所有分割结果
    - _Requirements: 8.7_

- [x] 13. 检查点 - 批量功能验证
  - 确保所有测试通过，如有问题请询问用户。

- [x] 14. 边界框坐标显示
  - [x] 14.1 实现边界框坐标显示
    - 在 PromptPanel 或 ImageCanvas 中显示已绘制边界框的坐标值
    - _Requirements: 4.5_

  - [x] 14.2 编写属性测试：边界框坐标显示一致性
    - **Property 8: 边界框坐标显示一致性**
    - 测试显示坐标与内部状态一致
    - **Validates: Requirements 4.5**

- [x] 15. 错误处理完善
  - [x] 15.1 后端错误处理
    - 实现统一的异常处理中间件
    - 实现 400/500/503 错误响应格式
    - _Requirements: 6.3, 6.4_

  - [x] 15.2 前端错误处理
    - 实现错误提示组件
    - 处理网络错误、API 错误、文件格式错误
    - 实现"未找到匹配对象"提示
    - _Requirements: 1.4, 2.5, 7.4_

- [x] 16. 最终检查点 - 全功能验证
  - 确保所有测试通过，如有问题请询问用户。

## 备注

- 标记 `*` 的任务为可选任务，可跳过以加快 MVP 开发
- 每个任务都引用了具体的需求编号以便追溯
- 属性测试验证设计文档中定义的正确性属性
- 检查点用于阶段性验证，确保增量开发的稳定性
