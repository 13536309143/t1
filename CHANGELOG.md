# 变更日志

本文档记录项目中的重要变更。

## 记录格式

每条记录统一采用以下结构：

```md
## YYYY-MM-DD

### 变更标题

类型：功能 | 性能 | 缺陷修复 | 其他

涉及文件：
- `path/to/file1`
- `path/to/file2`

变更内容：
- 变更说明 1
- 变更说明 2

验证：
- 验证命令或验证步骤
- 验证结果
```

## 2026-05-06

### 场景零件选择、模型树、高亮与键盘平移交互

类型：功能 | 性能 | 缺陷修复

涉及文件：
- `src/resources.hpp`
- `src/lodclusters.hpp`
- `src/lodclusters.cpp`
- `src/lodclusters_ui.cpp`
- `src/renderer.hpp`
- `src/renderer.cpp`
- `src/renderer_clusters_lod.cpp`
- `shaders/render_instance_bbox.mesh.glsl`
- `shaders/render_instance_bbox.frag.glsl`

变更内容：
- 接回鼠标拾取流程，支持在 viewport 中点击场景内单个 `Instance`，并在 UI 中显示对应 instance、geometry、三角形数、顶点数、最高细节 cluster 数与总 LOD cluster 数。
- 在 `Statistics` 面板新增 `Model Tree`，按 geometry 分组列出场景 instance，点击树中条目可选中并同步黄色高亮边框。
- 为选中模型新增黄色 bbox 高亮；未开启全局 `Instance BBoxes` 时只绘制选中 instance 的边框。
- 新增 `Interactive Mode`，支持使用小键盘 `8/4/2/6` 沿相机上/左/下/右方向平移选中零件，并提供 `Move Speed`、`Reset Selected`、`Reset All` 控制。
- 增加 CPU 侧 instance transform 脏标记与 GPU `RenderInstance` 局部同步路径，避免移动单个零件时重建 renderer 或重传全部实例。
- 修复选中高亮导致帧率大幅下降的问题：selected-only 高亮路径现在只 dispatch 一个 bbox，不再遍历所有 instance 后通过 fragment discard 过滤。
- 交互模式下强制捕获键盘输入，避免方向键被相机控制器同时消费；随后将移动键位改为小键盘，进一步规避相机快捷键冲突。

验证：
- `cmake --build build --config Release`
- `_bin\Release\t1.exe --headless --headlessframes 1 --scene _downloaded_resources\bunny_v2\bunny.gltf`
- 构建通过，headless 场景启动与 shader 编译通过。

## 2026-04-29
### 混合软硬光栅、自适应分流与目标占比反馈控制器

类型：功能 | 性能

涉及文件：
- `src/renderer.hpp`
- `src/resources.hpp`
- `src/lodclusters.hpp`
- `src/lodclusters.cpp`
- `src/lodclusters_ui.cpp`
- `src/renderer_clusters_lod.cpp`
- `shaders/shaderio_building.h`
- `shaders/traversal_run_separate_groups.comp.glsl`

变更内容：
- 将原来的 compute raster 用户入口整理为面向 UI 的混合软硬光栅工作流。
- 为 separate-groups 光栅遍历新增 `Adaptive Routing` 自适应分流模式。
- 重写 SW 路由启发式，使其优先命中真正“小簇 + 小三角 + 高密度”的 cluster，而不再只依赖单一密度阈值。
- 新增可配置的 SW 分流参数：`SW Max Extent` 与 `SW Min Tri Density`。
- 新增运行时有效阈值：`swRasterThresholdEffective` 与 `swRasterTriangleDensityThresholdEffective`。
- 在 CPU 侧加入上一帧统计反馈与 EMA 平滑逻辑。
- 引入目标占比控制器，使有效 SW 分流阈值围绕目标 `SW triangle share` 自动收敛。
- 在 UI 中新增并展示控制器选项与实时状态，包括 `Feedback Auto-Tune`、`SW Target Tri Share`、`SW Effective Extent`、`SW Effective Density`、`SW EMA Cluster Share`、`SW EMA Tri Share` 与 `SW EMA Tri/Cluster`。
- 将 effective 阈值接入 `SceneBuilding`，使 shader 实际使用反馈调节后的值，而不是仅使用 UI 原始基线值。

验证：
- `cmake --build build --config Release`
- 构建通过。

## 2026-04-28

### 相似几何体的 ClusterLOD 复用

类型：功能

涉及文件：
- `src/scene.hpp`
- `src/scene.cpp`

变更内容：
- 在 `processGeometry` 阶段新增几何体相似性查找，用于在同一次场景加载过程中复用已构建的 `ClusterLOD` 结果。
- 将相似性判断拆分为粗筛候选与严格哈希确认两层。
- 在命中完全一致的情况下，直接复用先前构建好的 `groupData`、`groupInfos`、`lodLevels`、`lodNodes` 与 `lodNodeBboxes`。
- 为复用路径补齐 histogram 与构建统计，保证 UI、缓存统计与下游限制计算保持一致。
- 增加复用日志输出，直接报告 similarity lookups 与 hits。
- 采用保守复用策略，仅在完全一致时才复用。

验证：
- `cmake --build build --config Release`
- `.\_bin\Release\t1.exe --processingonly 1 --autoloadcache 0 --autosavecache 0 --scene _downloaded_resources\bunny_v2\bunny_similarity_test.gltf`
- 构建通过，测试场景正确报告了相似几何体复用命中。

## 2026-04-27

### 连续式 ClusterLOD 过渡平滑

类型：功能

涉及文件：
- `shaders/shaderio_building.h`
- `shaders/traversal.glsl`
- `shaders/traversal_init.comp.glsl`
- `shaders/traversal_run.comp.glsl`
- `shaders/traversal_run_separate_groups.comp.glsl`
- `src/lodclusters.cpp`
- `src/lodclusters_ui.cpp`
- `src/renderer_clusters_lod.cpp`
- `src/resources.hpp`

变更内容：
- 将原先基于硬阈值的 ClusterLOD 遍历判定替换为更平滑的屏幕空间误差过渡带。
- 新增一条运行时参数链路，将类似 `lodTransitionWidth` 的配置打通到 frame config、build 参数、UI 与命令行。
- 更新遍历 shader，使 coarse/fine LOD 过渡期使用稳定 decision key，减少可见跳变。

验证：
- `cmake --build build --config Release`
- `.\_bin\Release\t1.exe --headless --headlessframes 1 --scene _downloaded_resources\bunny_v2\bunny.gltf`
- 构建通过，headless 启动通过。

## 2026-04-21

### LOD 健壮性改进

类型：缺陷修复

涉及文件：
- `src/meshlod_build.h`
- `src/clusterlod.cpp`

变更内容：
- 移除“最高 LOD 层必须只有一个 cluster”的强制假设。
- 允许最高 LOD 层包含多个 clusters 与 groups。
- 修复复杂网格在该路径下的处理问题。
- 通过合并所有 cluster bounds 的方式，改进最高 LOD 层的包围体计算。

验证：
- 工程构建成功。

### 并行处理优化

类型：性能 | 缺陷修复

涉及文件：
- `src/streaming.cpp`
- `src/streamutils.cpp`
- `shaders/shaderio.h`
- `shaders/stream_agefilter_groups.comp.glsl`

变更内容：
- 优化任务队列管理与批量处理逻辑。
- 改进内存分配行为，降低碎片。
- 调整 compute shader 的并行处理粒度。
- 优化异步传输处理与命令缓冲区使用方式。
- 修复 `stream_agefilter_groups.comp.glsl` 中的 shader 侧函数调用问题。

验证：
- 工程构建成功。

## 2026-04-20

### LOD 本地索引与包围体优化

类型：性能

涉及文件：
- `src/lod.h`
- `src/meshlod_local_indices.h`

变更内容：
- 优化 local indices 的哈希缓存行为。
- 减少 LOD cluster 处理中的不必要内存抖动。
- 优化 bounds merge 相关逻辑。
- 将这些优化同步到 meshlod 实现路径。

验证：
- 工程构建成功。

### LOD 过渡运行时支持

类型：功能

涉及文件：
- `shaders/shaderio.h`
- `src/lodclusters.cpp`
- `shaders/traversal.glsl`
- `shaders/traversal_run.comp.glsl`
- `shaders/clusters.mesh.glsl`

变更内容：
- 为 `FrameConstants` 增加时间相关字段。
- 更新运行时帧逻辑，计算并传递 `time` 与 `deltaTime`。
- 添加 shader 侧过渡评估辅助函数，用于更平滑的 LOD 混合。
- 将过渡评估逻辑接入 traversal 与 mesh shader 路径。

验证：
- 工程构建成功。

### 渲染管线优化

类型：性能

涉及文件：
- `src/renderer_clusters_lod.cpp`
- `src/resources.cpp`
- `src/resources.hpp`
- `shaders/traversal_run.comp.glsl`

变更内容：
- 优化 compute traversal 的执行行为与内存访问模式。
- 减少不必要的同步与 barrier 开销。
- 改进资源与缓冲区分配流程。
- 优化 shader 编译与 pipeline 创建流程。
- 修复 SPIR-V 兼容性问题。

验证：
- 工程构建成功。

### Hi-Z 修复与优化

类型：性能 | 缺陷修复

涉及文件：
- `shaders/hiz.comp.glsl`
- `src/hiz.cpp`
- `src/resources.cpp`

变更内容：
- 修复 Hi-Z 路径中的 shader 声明与初始化问题。
- 改进 Hi-Z compute shader 的边界处理与 subgroup 使用方式。
- 优化 Hi-Z 生成过程中的同步与资源使用。
- 确保双 Hi-Z 路径在 two-pass culling 下正确初始化。

验证：
- 工程构建成功。

### 资源与内存管理整理

类型：性能

涉及文件：
- `src/resources.hpp`
- `src/resources.cpp`

变更内容：
- 增加面向重复分配场景的内存池式资源复用思路。
- 改进命令缓冲区的复用与分配行为。
- 减少不必要的 framebuffer 与资源重建。
- 增加更完整的内存使用统计与报告。
- 修复 `VkCommandBufferAllocateInfo` 成员顺序问题。

验证：
- 工程构建成功。

## 2026-04-19

### 双 Hi-Z Buffer 支持

类型：功能

涉及文件：
- `src/hiz.cpp`
- `src/hiz.hpp`

变更内容：
- 新增双 Hi-Z buffer 支持，改善时间一致性表现。
- 优化 Hi-Z 更新逻辑。
- 修复相关内存管理问题。

验证：
- 工程构建成功。

## 2026-04-18

### UI 与场景初始化整理

类型：其他

涉及文件：
- `src/lodclusters.cpp`

变更内容：
- 增加更多可视化选项与参数控制。
- 改进场景加载与初始化流程。
- 优化相机控制与场景导航体验。

验证：
- 工程构建成功。

## 模板

```md
## YYYY-MM-DD

### 变更标题

类型：功能 | 性能 | 缺陷修复 | 其他

涉及文件：
- `path/to/file1`
- `path/to/file2`

变更内容：
- 变更说明 1
- 变更说明 2

验证：
- 验证命令或验证步骤
- 验证结果
```
