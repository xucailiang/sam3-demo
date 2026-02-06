# 需求文档

## 简介

SAM3（Segment Anything Model 3）演示项目是一个全栈 Web 应用，用于展示 Meta 最新发布的 SAM3 模型的图像分割能力。该项目提供直观的用户界面，支持文本提示和视觉提示（点击、边界框）等多种交互方式。

**注意：本项目仅支持图像分割功能，不包含视频分割与跟踪功能。**

### 技术栈选择

基于 SAM3 最佳实践研究，本项目采用以下技术栈：

- **后端**: Python FastAPI + Ultralytics SAM3 集成（推荐方式，更简单稳定）
- **前端**: React + TypeScript + Canvas API
- **部署**: Docker + Docker Compose
- **模型**: Ultralytics SAM3 (sam3.pt)，支持半精度推理优化

## 术语表

- **SAM3**: Segment Anything Model 3，Meta 发布的第三代分割基础模型，参数量约 848M
- **PCS**: Promptable Concept Segmentation，可提示概念分割，SAM3 的核心能力
- **Segmentation_Service**: 后端分割服务组件，基于 FastAPI 构建，负责处理图像分割请求
- **Frontend_App**: 前端 React 应用组件，提供用户交互界面和结果可视化
- **Text_Prompt**: 文本提示，使用自然语言短语描述要分割的目标概念
- **Visual_Prompt**: 视觉提示，包括点击坐标和边界框两种方式
- **Point_Prompt**: 点击提示，包含坐标位置和标签（前景点=1，背景点=0）
- **Box_Prompt**: 边界框提示，包含左上角和右下角坐标 [x1, y1, x2, y2]
- **Mask**: 分割掩码，表示分割结果的二值图像数组
- **Confidence_Score**: 置信度分数，表示分割结果的可信程度，范围 0-1
- **Image_Example**: 图像范例，使用示例图像区域作为视觉提示进行分割
- **Batch_Processing**: 批量处理，对多张图像使用统一提示进行分割

## 需求

### 需求 1：图像上传与预览

**用户故事:** 作为用户，我希望能够上传图像并预览，以便对图像进行分割操作。

**验收标准 1.1-1.6:**

1. WHEN 用户选择图像文件 THEN THE Frontend_App SHALL 验证文件扩展名为 jpg、jpeg、png 或 webp
2. WHEN 用户上传有效图像文件 THEN THE Frontend_App SHALL 在 Canvas 画布上渲染图像预览
3. WHEN 图像渲染完成 THEN THE Frontend_App SHALL 将分割功能按钮状态设置为可用
4. IF 用户上传的文件扩展名不在允许列表中 THEN THE Frontend_App SHALL 显示错误提示消息并阻止上传
5. WHEN 图像显示在 Canvas 画布上 THEN THE Frontend_App SHALL 按原始宽高比缩放图像以适应画布尺寸

### 需求 2：文本提示分割

**用户故事:** 作为用户，我希望通过输入文本描述来分割图像中的目标对象，以便快速定位特定概念。

**验收标准 2.1-2.5:**

1. WHEN 用户输入 Text_Prompt 并点击提交按钮 THEN THE Segmentation_Service SHALL 调用 SAM3 模型并返回所有匹配该概念的 Mask 数组
2. WHEN 用户输入包含逗号分隔的多个 Text_Prompt THEN THE Segmentation_Service SHALL 解析为独立概念列表并分别返回每个概念的分割结果
3. WHEN Segmentation_Service 返回分割结果 THEN THE Frontend_App SHALL 在原图上叠加渲染彩色半透明 Mask
4. WHEN 分割结果包含检测对象 THEN THE Frontend_App SHALL 显示每个对象的标签和 Confidence_Score
5. IF Segmentation_Service 返回空 Mask 数组 THEN THE Frontend_App SHALL 显示"未找到匹配对象"提示消息

### 需求 3：点击提示分割

**用户故事:** 作为用户，我希望通过点击图像上的位置来分割目标对象，以便精确选择特定对象。

**验收标准 3.1-3.5:**

1. WHEN 用户在 Canvas 画布上点击 THEN THE Frontend_App SHALL 记录点击的像素坐标并在该位置渲染标记点
2. WHEN 用户使用鼠标左键点击 THEN THE Frontend_App SHALL 将该 Point_Prompt 的标签设置为 1（前景点）
3. WHEN 用户使用鼠标右键点击 THEN THE Frontend_App SHALL 将该 Point_Prompt 的标签设置为 0（背景点）
4. WHEN 用户提交 Point_Prompt 列表 THEN THE Segmentation_Service SHALL 返回包含所有前景点且排除所有背景点区域的 Mask
5. WHEN 用户点击"清除点"按钮 THEN THE Frontend_App SHALL 从状态中移除所有 Point_Prompt 并清除画布上的标记点

### 需求 4：边界框提示分割

**用户故事:** 作为用户，我希望通过绘制边界框来分割目标对象，以便框选特定区域进行分割。

**验收标准 4.1-4.5:**

1. WHEN 用户在 Canvas 画布上按下鼠标并拖拽 THEN THE Frontend_App SHALL 实时渲染矩形边界框预览
2. WHEN 用户释放鼠标完成边界框绘制 THEN THE Frontend_App SHALL 记录 Box_Prompt 坐标 [x1, y1, x2, y2]
3. WHEN 用户提交 Box_Prompt THEN THE Segmentation_Service SHALL 返回边界框区域内主要对象的 Mask
4. WHEN 用户点击"清除框"按钮 THEN THE Frontend_App SHALL 从状态中移除所有 Box_Prompt 并清除画布上的边界框
5. WHEN 边界框绘制完成 THEN THE Frontend_App SHALL 在界面上显示边界框的像素坐标值

### 需求 5：分割结果可视化

**用户故事:** 作为用户，我希望清晰地查看分割结果，以便理解模型的输出。

**验收标准 5.1-5.5:**

1. WHEN Segmentation_Service 返回多个 Mask THEN THE Frontend_App SHALL 为每个 Mask 分配不同的颜色进行渲染
2. WHEN 用户将鼠标悬停在某个 Mask 区域上 THEN THE Frontend_App SHALL 高亮显示该 Mask 并显示对象标签
3. WHEN 用户点击"显示/隐藏掩码"切换按钮 THEN THE Frontend_App SHALL 切换所有 Mask 的可见性状态
4. WHEN 用户拖动透明度滑块 THEN THE Frontend_App SHALL 更新所有 Mask 的透明度值（范围 0-1）
5. WHEN 分割操作完成 THEN THE Frontend_App SHALL 显示统计信息包括检测对象数量和处理耗时毫秒数

### 需求 6：后端 API 服务

**用户故事:** 作为开发者，我希望有稳定的后端 API 服务，以便前端能够调用分割功能。

**验收标准 6.1-6.6:**

1. THE Segmentation_Service SHALL 提供符合 OpenAPI 规范的 RESTful API 接口
2. WHEN Segmentation_Service 接收单张图像分割请求 THEN THE Segmentation_Service SHALL 在 5000 毫秒内返回响应
3. IF Segmentation_Service 接收格式错误的请求 THEN THE Segmentation_Service SHALL 返回 HTTP 400 状态码和错误描述
4. IF Segmentation_Service 处理过程发生内部错误 THEN THE Segmentation_Service SHALL 返回 HTTP 500 状态码和错误描述
5. THE Segmentation_Service SHALL 配置 CORS 中间件以允许前端域名的跨域请求
6. WHEN Segmentation_Service 启动 THEN THE Segmentation_Service SHALL 预加载 SAM3 模型到 GPU 内存

### 需求 7：图像范例分割

**用户故事:** 作为用户，我希望通过上传示例图像来分割目标对象，以便使用视觉示例定位相似概念。

**验收标准 7.1-7.4:**

1. WHEN 用户上传示例图像并选择示例区域 THEN THE Segmentation_Service SHALL 使用该区域作为视觉提示进行分割
2. WHEN 用户提交图像范例提示 THEN THE Segmentation_Service SHALL 返回目标图像中所有与示例相似的对象的 Mask 数组
3. WHEN Segmentation_Service 返回图像范例分割结果 THEN THE Frontend_App SHALL 在原图上叠加渲染彩色半透明 Mask
4. IF Segmentation_Service 未找到与示例匹配的对象 THEN THE Frontend_App SHALL 显示"未找到匹配对象"提示消息

### 需求 8：批量图片分割

**用户故事:** 作为用户，我希望能够批量上传多张图片并使用统一提示进行分割，以便高效处理大量图像。

**验收标准 8.1-8.7:**

1. WHEN 用户选择多张图像文件 THEN THE Frontend_App SHALL 验证所有文件扩展名并显示文件列表
2. WHEN 用户输入统一的 Text_Prompt 并点击批量分割按钮 THEN THE Segmentation_Service SHALL 对所有图像执行分割并返回结果数组
3. WHEN 用户使用图像范例模式并点击批量分割按钮 THEN THE Segmentation_Service SHALL 使用示例图像对所有目标图像执行分割并返回结果数组
4. WHEN 批量分割任务开始 THEN THE Frontend_App SHALL 显示进度条和当前处理的图像索引
5. WHEN 批量分割完成 THEN THE Frontend_App SHALL 以缩略图网格形式展示所有分割结果
6. WHEN 用户点击某个缩略图 THEN THE Frontend_App SHALL 显示该图像的详细分割结果
7. WHEN 用户点击"批量导出"按钮 THEN THE Frontend_App SHALL 触发下载包含所有分割结果的 ZIP 压缩包

### 需求 9：结果导出

**用户故事:** 作为用户，我希望能够导出分割结果，以便在其他应用中使用。

**验收标准 9.1-9.3:**

1. WHEN 用户点击"导出掩码"按钮 THEN THE Frontend_App SHALL 触发下载 PNG 格式的二值 Mask 图像文件
2. WHEN 用户点击"导出叠加图"按钮 THEN THE Frontend_App SHALL 触发下载带 Mask 叠加渲染的原图 PNG 文件
3. WHEN 用户点击"导出 JSON"按钮 THEN THE Frontend_App SHALL 触发下载包含 Mask 轮廓坐标、边界框和元数据的 JSON 文件
