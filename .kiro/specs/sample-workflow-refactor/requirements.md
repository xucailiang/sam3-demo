# 需求文档

## 简介

本文档描述 SAM3 演示项目中 SampleWorkflow 功能的重构需求。现有实现使用 `inference_features()` 进行跨图像分割，但 SAM3 的 `bboxes` 提示仅在同一图像内有效，导致跨图像分割无法正常工作。

本次重构将移除所有失效的跨图像代码，替换为两种新的批量分割模式：
- **模式 A（图像拼接 + BBox）**：将样品图和目标图水平拼接为一张图像（左边为样本图，右边为目标图），在拼接图上运行 SAM3 bbox 推理，利用视觉相似性在目标区域找到匹配对象
- **模式 B（纯文本批量）**：用户输入文本描述，对每张目标图独立运行 SAM3 文本推理

## 术语表

- **Segmentation_Service**: 后端分割服务组件，基于 FastAPI 构建，负责处理图像分割请求
- **Frontend_App**: 前端 React 应用组件，提供用户交互界面和结果可视化
- **Stitched_Image**: 拼接图像，将样品图和目标图水平拼接为一张组合图像，宽度为 W1+W2，高度为 max(H1,H2)
- **Sample_Image**: 样品图像，用户上传的包含目标对象的参考图像
- **Target_Image**: 目标图像，用户上传的需要进行批量分割的图像
- **Sample_BBox**: 样品边界框，用户在样品图上绘制的标记目标对象的边界框 [x1, y1, x2, y2]
- **Mode_Switcher**: 模式切换器，前端组件，允许用户在模式 A（拼接+BBox）和模式 B（文本）之间切换
- **Coordinate_Offset**: 坐标偏移，将拼接图中目标区域的坐标减去样品图宽度 W1，还原为目标图的原始坐标

## 需求

### 需求 1：移除失效的跨图像分割代码

**用户故事:** 作为开发者，我希望移除所有基于 `inference_features()` 的跨图像分割代码，以便代码库保持干净且无失效逻辑。

#### 验收标准

1. WHEN 重构完成 THEN THE Segmentation_Service SHALL 不包含 `segment_with_image_example()` 方法
2. WHEN 重构完成 THEN THE Segmentation_Service SHALL 不包含 `batch_segment_with_image_example()` 方法
3. WHEN 重构完成 THEN THE Segmentation_Service SHALL 不包含 `_process_inference_features_results()` 方法
4. WHEN 重构完成 THEN THE Segmentation_Service SHALL 不包含 `/api/segment/image-example` 端点
5. WHEN 重构完成 THEN THE Segmentation_Service SHALL 不包含 `/api/batch/image-example` 端点
6. WHEN 重构完成 THEN THE Frontend_App SHALL 不包含 `segmentWithImageExample` API 调用函数
7. WHEN 重构完成 THEN THE Frontend_App SHALL 不包含 `batchSegmentWithImageExample` API 调用函数

### 需求 2：图像拼接分割服务（模式 A 后端）

**用户故事:** 作为用户，我希望后端能够通过图像拼接方式实现跨图像的视觉相似性分割，以便在目标图中找到与样品图标记区域相似的对象。

#### 验收标准

1. WHEN Segmentation_Service 接收样品图（W1×H1）和目标图（W2×H2）THEN THE Segmentation_Service SHALL 将两张图水平拼接为一张宽度为 W1+W2、高度为 max(H1,H2) 的 Stitched_Image
2. WHEN Stitched_Image 生成完成 THEN THE Segmentation_Service SHALL 在 Stitched_Image 上使用原始 Sample_BBox 坐标调用 SAM3 `predictor(bboxes=...)` 进行推理
3. WHEN SAM3 返回推理结果 THEN THE Segmentation_Service SHALL 过滤掉边界框完全落在样品图区域内（bbox 的 x2 坐标小于等于 W1）的掩码
4. WHEN 过滤完成后 THEN THE Segmentation_Service SHALL 将目标区域掩码的边界框 x 坐标减去 W1 进行 Coordinate_Offset 还原
5. WHEN 过滤完成后 THEN THE Segmentation_Service SHALL 将掩码数组裁剪为仅包含目标图区域（从 x=W1 开始，宽度为 W2）
6. IF 拼接后的图像尺寸超出合理范围 THEN THE Segmentation_Service SHALL 返回 HTTP 400 状态码和错误描述

### 需求 3：拼接分割 API 端点

**用户故事:** 作为前端开发者，我希望有新的 API 端点支持拼接分割请求，以便前端能够调用模式 A 的分割功能。

#### 验收标准

1. THE Segmentation_Service SHALL 提供 `POST /api/segment/stitch` 端点，接收目标图文件、样品图文件和 Sample_BBox 参数
2. THE Segmentation_Service SHALL 提供 `POST /api/batch/stitch` 端点，接收多个目标图文件、样品图文件和 Sample_BBox 参数
3. WHEN `/api/batch/stitch` 接收多张目标图 THEN THE Segmentation_Service SHALL 对每张目标图独立执行拼接分割并返回结果数组
4. IF 请求缺少必要参数（目标图、样品图或 Sample_BBox）THEN THE Segmentation_Service SHALL 返回 HTTP 400 状态码和错误描述

### 需求 4：纯文本批量分割（模式 B）

**用户故事:** 作为用户，我希望能够输入文本描述对多张图片进行批量分割，以便无需样品图即可快速处理大量图像。

#### 验收标准

1. WHEN 用户输入文本描述并提交多张目标图 THEN THE Segmentation_Service SHALL 对每张目标图独立调用 SAM3 `predictor(text=[description])` 进行推理
2. WHEN 文本批量分割完成 THEN THE Segmentation_Service SHALL 返回与输入图像数量相同的结果数组
3. IF 文本描述为空 THEN THE Segmentation_Service SHALL 返回 HTTP 400 状态码和错误描述

### 需求 5：前端模式切换

**用户故事:** 作为用户，我希望在 SampleWorkflow 界面中切换模式 A（拼接+BBox）和模式 B（文本），以便根据场景选择合适的批量分割方式。

#### 验收标准

1. WHEN SampleWorkflow 组件加载 THEN THE Frontend_App SHALL 显示 Mode_Switcher 组件，默认选中模式 A
2. WHEN 用户切换到模式 A THEN THE Frontend_App SHALL 显示样品图上传、边界框绘制和批量目标图上传界面
3. WHEN 用户切换到模式 B THEN THE Frontend_App SHALL 显示文本输入框和批量目标图上传界面，隐藏样品图相关界面
4. WHEN 用户切换模式 THEN THE Frontend_App SHALL 清除当前模式的所有状态（已上传文件、分割结果、进度）

### 需求 6：模式 A 前端工作流

**用户故事:** 作为用户，我希望在模式 A 中完成从上传样品图到查看批量分割结果的完整工作流。

#### 验收标准

1. WHEN 用户上传样品图 THEN THE Frontend_App SHALL 在 Canvas 画布上渲染样品图预览
2. WHEN 用户在样品图上拖拽绘制边界框 THEN THE Frontend_App SHALL 实时渲染矩形边界框预览并记录 Sample_BBox 坐标
3. WHEN 用户点击"预览分割效果"按钮 THEN THE Frontend_App SHALL 调用现有的边界框分割 API 在样品图上预览分割效果
4. WHEN 用户上传多张目标图并点击"开始批量分割"按钮 THEN THE Frontend_App SHALL 调用 `/api/batch/stitch` 端点执行批量拼接分割
5. WHEN 批量分割完成 THEN THE Frontend_App SHALL 以缩略图网格形式展示所有分割结果，并支持点击查看详情
6. WHEN 批量分割任务进行中 THEN THE Frontend_App SHALL 显示进度条和当前处理状态

### 需求 7：模式 B 前端工作流

**用户故事:** 作为用户，我希望在模式 B 中通过文本描述完成批量分割工作流。

#### 验收标准

1. WHEN 用户在模式 B 中输入文本描述 THEN THE Frontend_App SHALL 记录文本描述内容
2. WHEN 用户上传多张目标图并点击"开始批量分割"按钮 THEN THE Frontend_App SHALL 调用 `/api/batch/text` 端点执行批量文本分割
3. WHEN 批量分割完成 THEN THE Frontend_App SHALL 以缩略图网格形式展示所有分割结果
4. IF 文本描述为空且用户点击"开始批量分割"按钮 THEN THE Frontend_App SHALL 显示提示消息要求输入文本描述

### 需求 8：前端 API 客户端更新

**用户故事:** 作为前端开发者，我希望 API 客户端提供新的拼接分割调用函数，以便前端组件能够调用新的后端端点。

#### 验收标准

1. WHEN 前端需要调用拼接分割 THEN THE Frontend_App SHALL 提供 `segmentWithStitch` 函数，发送目标图、样品图和 Sample_BBox 到 `/api/segment/stitch`
2. WHEN 前端需要调用批量拼接分割 THEN THE Frontend_App SHALL 提供 `batchSegmentWithStitching` 函数，发送多张目标图、样品图和 Sample_BBox 到 `/api/batch/stitch`
3. THE Frontend_App SHALL 移除 `segmentWithImageExample` 函数
4. THE Frontend_App SHALL 移除 `batchSegmentWithImageExample` 函数
