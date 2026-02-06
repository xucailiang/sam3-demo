# 实现计划：SampleWorkflow 重构

## 概述

将 SampleWorkflow 从失效的跨图像 `inference_features()` 方案重构为图像拼接方案，同时新增纯文本批量分割模式。分为后端核心逻辑、API 端点、前端组件三个阶段实施。

## Tasks

- [x] 1. 实现后端拼接核心逻辑
  - [x] 1.1 在 `backend/app/services/segmentation_service.py` 中实现 `stitch_images()` 工具函数
    - 接收两张 numpy 数组图像，水平拼接，高度不同时底部填充黑色
    - 返回拼接图、sample_width、target_width
    - _Requirements: 2.1_
  - [x] 1.2 在 `backend/app/services/segmentation_service.py` 中实现 `filter_and_offset_results()` 工具函数
    - 过滤 bbox.x2 <= sample_width 的结果
    - 偏移保留结果的 x 坐标 -= sample_width
    - 裁剪掩码为目标图区域
    - _Requirements: 2.3, 2.4, 2.5_
  - [ ]* 1.3 为 `stitch_images` 编写属性测试
    - **Property 1: 拼接图像尺寸正确性**
    - **Validates: Requirements 2.1**
  - [ ]* 1.4 为 `filter_and_offset_results` 编写属性测试
    - **Property 2: 过滤 + 偏移 + 裁剪正确性**
    - **Validates: Requirements 2.3, 2.4, 2.5**

- [x] 2. 实现后端拼接分割服务方法
  - [x] 2.1 在 `SegmentationService` 中实现 `segment_with_stitch()` 方法
    - 调用 stitch_images 拼接，set_image 设置拼接图，predictor(bboxes=sample_box) 推理
    - 调用 filter_and_offset_results 后处理
    - 拼接图尺寸超限时抛出 ValueError
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_
  - [x] 2.2 在 `SegmentationService` 中实现 `batch_segment_with_stitch()` 方法
    - 逐张目标图调用 segment_with_stitch，收集结果数组
    - 单张失败时返回空结果，继续处理其余图片
    - _Requirements: 3.3_
  - [ ]* 2.3 为 `batch_segment_with_stitch` 编写属性测试（mock SAM3）
    - **Property 3: 批量拼接分割结果数量一致性**
    - **Validates: Requirements 3.3**

- [x] 3. 移除旧代码并新增 API 端点
  - [x] 3.1 从 `SegmentationService` 中移除 `segment_with_image_example()`、`batch_segment_with_image_example()`、`_process_inference_features_results()` 方法
    - _Requirements: 1.1, 1.2, 1.3_
  - [x] 3.2 从 `backend/app/api/segment.py` 中移除 `/api/segment/image-example` 端点
    - _Requirements: 1.4_
  - [x] 3.3 从 `backend/app/api/batch.py` 中移除 `/api/batch/image-example` 端点
    - _Requirements: 1.5_
  - [x] 3.4 在 `backend/app/api/segment.py` 中新增 `POST /api/segment/stitch` 端点
    - 接收 file（目标图）、sample_file（样品图）、sample_box（JSON）
    - 参数验证：缺少参数返回 400
    - _Requirements: 3.1, 3.4_
  - [x] 3.5 在 `backend/app/api/batch.py` 中新增 `POST /api/batch/stitch` 端点
    - 接收 files（目标图[]）、sample_file（样品图）、sample_box（JSON）
    - 参数验证：缺少参数返回 400
    - _Requirements: 3.2, 3.3, 3.4_

- [x] 4. Checkpoint - 后端验证
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. 更新前端 API 客户端和类型
  - [x] 5.1 在 `frontend/src/api/client.ts` 中新增 `segmentWithStitch()` 和 `batchSegmentWithStitch()` 函数
    - _Requirements: 8.1, 8.2_
  - [x] 5.2 从 `frontend/src/api/client.ts` 中移除 `segmentWithImageExample()` 和 `batchSegmentWithImageExample()` 函数
    - _Requirements: 8.3, 8.4, 1.6, 1.7_
  - [x] 5.3 更新 `frontend/src/types/index.ts`，从 `InteractionMode` 中移除 `'image-example'`，新增 `WorkflowMode` 类型
    - _Requirements: 5.1_
  - [x] 5.4 更新 `frontend/src/hooks/useBatchSegmentation.ts`，将 `batchSegmentWithImageExample` 替换为 `batchSegmentWithStitch`
    - _Requirements: 8.2_

- [x] 6. 重构 SampleWorkflow 前端组件
  - [x] 6.1 在 `SampleWorkflow` 组件中添加 Mode_Switcher（模式 A / 模式 B 切换），默认模式 A
    - 切换模式时清除所有状态
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  - [x] 6.2 重构模式 A 工作流：保留样品图上传和 BBox 绘制，批量分割改为调用 `batchSegmentWithStitch`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  - [x] 6.3 实现模式 B 工作流：文本输入框 + 批量目标图上传 + 调用 `batchSegmentWithText`
    - 文本为空时禁用按钮并显示提示
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  - [ ]* 6.4 为模式切换状态清除编写属性测试
    - **Property 5: 模式切换清除状态**
    - **Validates: Requirements 5.4**

- [ ] 7. Final checkpoint - 全部验证
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- 后端使用 Python + FastAPI，前端使用 TypeScript + React
- 属性测试后端使用 `hypothesis`，前端使用 `fast-check`
- 所有旧的跨图像代码在 Task 3 中统一移除，确保无残留引用
