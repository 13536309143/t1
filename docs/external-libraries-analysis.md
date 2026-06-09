# 项目外部库功能使用分析

本文分析当前项目中外部库承担的功能，以及这些功能在代码里的主要落点。分析范围包括 `CMakeLists.txt`、`cmake/FindNvproCore2.cmake`、`src`、`shaders` 和 `thirdparty`。

## 依赖总体结构

项目在根 `CMakeLists.txt` 中直接查找并链接 `nvpro_core2`，同时链接或编入以下外部库：

- `nvpro2::nvapp`
- `nvpro2::nvgui`
- `nvpro2::nvutils`
- `nvpro2::nvvk`
- `nvpro2::nvvkglsl`
- `meshoptimizer`
- `cgltf`
- `thirdparty/vulkan_radix_sort/src/vk_radix_sort.cc`

`cmake/FindNvproCore2.cmake` 会优先查找本地 `nvpro_core2/cmake/Setup.cmake`，找不到时可下载 `nvpro-samples/nvpro_core2.git`。当前脚本把默认 `NVPRO_GIT_TAG` 固定到提交 `03297bee1f997208951cc104abb5ecc3e1f987ed`，这意味着项目依赖的是一个特定版本的 nvpro2 组件集合，而不是随意跟踪 main 分支。

根 `CMakeLists.txt` 明确关闭了一些 nvpro2 可选组件：

- `NVPRO2_ENABLE_nvgl OFF`
- `NVPRO2_ENABLE_nvgpu_monitor OFF`
- `NVPRO2_ENABLE_nvslang OFF`
- `NVPRO2_ENABLE_nvvkgltf OFF`

这很重要：项目没有使用 nvpro2 的 `nvvkgltf` 来导入 glTF，而是自己用 `cgltf` 解析 glTF，并结合 `meshoptimizer` 解压 `EXT_meshopt_compression`。

## nvpro2

nvpro2 是本项目最大的基础设施依赖。它并不是只提供一个工具函数，而是覆盖了应用框架、Vulkan 初始化、资源管理、shader 编译、UI 辅助、日志、参数系统、线程池和 profiler。

### nvapp：应用框架和生命周期

主要文件：

- `src/app/main.cpp`
- `src/app/lodclusters.hpp`
- `src/app/lodclusters_lifecycle.cpp`

项目通过 `nvapp::Application` 和 `nvapp::ApplicationCreateInfo` 建立窗口应用。`main.cpp` 中先填充 `ApplicationCreateInfo`，包括应用名、菜单、VSync、headless、Vulkan instance/device/physical device/queues 和 ImGui docking 布局，然后调用 `app.init(appInfo)`、`app.addElement(...)`、`app.run()`、`app.deinit()`。

`LodClusters` 继承 `nvapp::IAppElement`，把项目主逻辑接入 nvapp 生命周期。它实现或使用的回调包括：

- `onAttach`
- `onDetach`
- `onPreRender`
- `onRender`
- `onUIRender`
- `onUIMenu`
- `onResize`
- `onFileDrop`

项目还使用 nvapp 的内置 UI/功能 element：

- `nvapp::ElementLogger`：把 `nvutils::Logger` 输出接到应用内 Log 面板。
- `nvapp::ElementProfiler`：显示 CPU/GPU profiler 结果。
- `nvapp::ElementCamera`：把 `CameraManipulator` 接入交互控制。
- `nvapp::ElementDefaultWindowTitle`：默认窗口标题。
- `nvapp::ElementSequencer`：驱动参数序列和自动化测试/截图流程。

这些功能让项目入口不需要直接管理 ImGui 主循环、窗口生命周期和常见面板，而是把核心渲染系统包装成一个 app element。

### nvgui：UI 辅助控件、相机和文件对话框

主要文件：

- `src/app/lodclusters_ui.cpp`
- `src/app/lodclusters_scene.cpp`
- `src/app/lodclusters_runtime.cpp`
- `src/app/lodclusters_lifecycle.cpp`

`nvgui` 主要用于让 ImGui UI 更结构化。项目中最常见的是：

- `nvgui::PropertyEditor`，别名为 `PE`，用于 Settings/Misc/Debug 面板里的属性表格。
- `nvgui::EnumRegistry`，保存在 `LodClusters::m_ui`，用于注册和显示枚举选项，例如渲染模式、visualize 模式、cluster 配置 preset。
- `nvgui::CameraWidget`，在 Misc Settings 中显示和编辑相机参数。
- `nvgui::skySimpleParametersUI`，用于编辑 sky/sun 参数。
- `nvgui::SetHomeCamera`、`nvgui::AddCamera`、`nvgui::SetCameraJsonFile`，用于注册 glTF 相机、默认相机和相机配置文件。
- `nvgui::isWindowHovered`，用于 viewport 中的交互判断。
- `nvgui::windowOpenFileDialog`，用于菜单栏打开模型文件。

UI 层同时直接使用 ImGui 和 ImPlot，但 `nvgui` 负责把常用 property/editor 风格控件和相机/天空参数编辑封装起来。

### nvutils：日志、参数、计时、文件、线程和通用工具

主要文件：

- `src/app/main.cpp`
- `src/app/lodclusters_config.cpp`
- `src/app/lodclusters_lifecycle.cpp`
- `src/app/lodclusters_scene.cpp`
- `src/core/cache.cpp`
- `src/scene/scene.cpp`
- `src/scene/scene_gltf.cpp`
- `src/scene/clusterlod.cpp`
- `src/streaming/streamutils.cpp`

项目使用了 `nvutils` 的多个子功能。

日志：

- `LOGI`、`LOGW`、`LOGE` 宏贯穿 scene loading、cache、renderer、streaming。
- `nvutils::Logger::getInstance().setLogCallback(...)` 在 `main.cpp` 中把日志转发给 `nvapp::ElementLogger`。
- `parameterSequenceCallback` 中也通过 logger 输出 memory report。

参数系统：

- `nvutils::ParameterRegistry` 注册命令行/配置参数。
- `nvutils::ParameterParser` 解析 `argc/argv` 和配置文件。
- `nvutils::ParameterSequencer` 支持脚本式参数序列，配合 profiler、截图和统计输出。

计时和 profiler：

- `nvutils::ProfilerManager`、`nvutils::ProfilerTimeline` 驱动 CPU/GPU 时间线。
- `nvutils::ScopedTimer` 用于记录 Vulkan context 创建等耗时。
- `nvutils::PerformanceTimer` 用于应用运行时间和帧时间相关逻辑。

文件和路径：

- `nvutils::utf8FromPath` 把 `std::filesystem::path` 转成日志、cgltf、cache 使用的 UTF-8 字符串。
- `nvutils::getExecutablePath` 用于构造 shader、资源和默认模型搜索路径。
- `nvutils::findFile` 用于查找默认 `a.glb`。
- `nvutils::FileReadMapping` 用于 memory mapped 读取 cache 和 glTF buffer。
- `nvutils::FileReadOverWriteMapping` 用于保存 cache。

并行任务：

- `nvutils::parallel_batches` 和 `nvutils::parallel_batches_pooled<1>` 用于并行保存 cache、导入多个 geometry、构建 LOD、构建层次结构。
- `nvutils::get_thread_pool().reset(...)` 根据 scene processing 配置调整线程池规模。

通用工具：

- `nvutils::align_up` 大量用于 CPU/GPU 共享结构的对齐、buffer range 规划、streaming buffer 内部布局。
- `nvutils::IDPool` 用于 streaming 中 group 和 cluster resident 资源 ID 分配。
- `nvutils::dumpSpirv` 在 shader 编译失败或开启 dump 时导出 SPIR-V。

### nvvk：Vulkan 上下文、资源、描述符、管线和同步

主要文件：

- `src/app/main.cpp`
- `src/renderer/resources.hpp`
- `src/renderer/resources.cpp`
- `src/renderer/renderer.cpp`
- `src/renderer/renderer_clusters_lod.cpp`
- `src/renderer/preloaded.cpp`
- `src/streaming/streaming.cpp`
- `src/streaming/streamutils.cpp`

`nvvk` 是项目 Vulkan 侧的主要封装。

上下文和设备：

- `nvvk::ContextInitInfo` 收集 instance/device extensions、队列需求、validation 设置和强制 GPU index。
- `nvvk::Context` 创建 instance、选择 physical device、创建 logical device。
- `nvvk::addSurfaceExtensions` 自动加入窗口 surface 所需 instance extensions。
- `nvvk::ValidationSettings` 配置 validation layer preset 和 message filter。
- `nvvk::DebugUtil` 和 `NVVK_DBG_NAME` 给 Vulkan 对象设置 debug name。
- `NVVK_CHECK` 统一检查 Vulkan 返回值。

资源分配：

- `nvvk::ResourceAllocator` 是本项目的核心 GPU 资源分配器，底层配合 VMA。
- `nvvk::Buffer`、`nvvk::LargeBuffer`、`nvvk::BufferTyped<T>` 用于 storage/uniform/transfer/indirect/readback buffer。
- `nvvk::Image` 用于 color、resolved color、depth-stencil、raster atomic、Hi-Z image。
- `nvvk::SamplerPool` 管理 sampler，供 framebuffer、Hi-Z 和 ImGui viewport texture 使用。
- `nvvk::StagingUploader` 支持 CPU 到 GPU buffer 上传。项目在 `Resources::BatchedUploader` 中用 `appendBufferMapping`、`cmdUploadAppended`、`releaseStaging` 组合批量上传。

描述符：

- `nvvk::DescriptorBindings` 声明 descriptor set layout 的 bindings。
- `nvvk::DescriptorPack` 管理 descriptor set layout、pool 和 set。
- `nvvk::WriteSetContainer` 收集 `VkWriteDescriptorSet`，用于 `vkUpdateDescriptorSets`。

管线：

- `nvvk::GraphicsPipelineState` 保存默认 graphics pipeline state。
- `nvvk::GraphicsPipelineCreator` 用于创建 mesh shader/fragment shader 管线、fullscreen 管线、debug bbox 管线。
- `nvvk::createPipelineLayout` 创建带 descriptor set layout 和 push constant range 的 pipeline layout。

同步和命令：

- `nvvk::SemaphoreState` 和项目自定义 `QueueState` 一起管理 timeline semaphore wait/signal。
- `nvvk::ManagedCommandPools` 用于 streaming transfer 队列中的多 slot command buffer 管理。
- `nvvk::cmdMemoryBarrier`、`nvvk::cmdImageMemoryBarrier` 辅助插入 Vulkan synchronization2 风格 barrier。
- `nvvk::ImageMemoryBarrierParams` 简化 image layout transition。

物理设备信息：

- `nvvk::PhysicalDeviceInfo` 保存 Vulkan 1.1/1.2/1.3 设备属性和特性。
- `nvvk::findDepthStencilFormat` 选择 depth-stencil 格式。

### nvvkglsl 和 shaderc：GLSL 编译到 SPIR-V

主要文件：

- `src/renderer/resources.hpp`
- `src/renderer/resources.cpp`
- `src/renderer/renderer.cpp`
- `src/renderer/renderer_clusters_lod.cpp`
- `src/renderer/hiz.cpp`
- `src/streaming/streaming.cpp`

项目运行时编译 GLSL shader，而不是只依赖离线编译产物。

`Resources` 内持有 `nvvkglsl::GlslCompiler m_glslCompiler`，初始化时：

- 添加 shader 搜索路径，包括 `shaders`、`shaders/interface`、`shaders/common`、`shaders/render`、`shaders/debug`、`shaders/post`、`shaders/streaming`、`shaders/traversal`、`shaders/build` 和 nvshaders 路径。
- 调用 `defaultOptions()`、`defaultTarget()`。
- 开启 `SetGenerateDebugInfo()`。

编译流程：

- `Resources::makeCompilerOptions()` 返回 `shaderc::CompileOptions`。
- renderer/streaming 根据配置添加宏，例如 `SUBGROUP_SIZE`、`USE_16BIT_DISPATCH`、`USE_EXT_MESH_SHADER`、`USE_STREAMING`、`CLUSTER_VERTEX_COUNT`、`CLUSTER_TRIANGLE_COUNT`、`USE_SORTING`、`USE_RENDER_STATS`。
- `Resources::compileShader(...)` 调用 `m_glslCompiler.compileFile(...)`。
- `nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(...)` 把编译结果转换为 `VkShaderModuleCreateInfo`。
- `nvvkglsl::GlslCompiler::getSpirvData(...)` 把 SPIR-V 数据传给 `GraphicsPipelineCreator::addShader`。

这套机制让 renderer 配置变化时可以重编 shader，并通过宏控制同一个 shader 文件中的多种路径。

## meshoptimizer

meshoptimizer 在项目里承担两个不同层级的职责：一是 glTF 导入阶段解压 `EXT_meshopt_compression`，二是 CPU 侧 Cluster LOD 构建。

### glTF EXT_meshopt_compression 解压

主要文件：

- `src/scene/scene_gltf.cpp`

当 cgltf 发现 `cgltf_buffer_view::has_meshopt_compression` 时，项目会把对应 buffer view 放入 `compressedViews`，然后在读取属性和索引之前调用 `loadCompressedViewsGLTF(...)`。

使用的 meshoptimizer API 包括：

- `meshopt_decodeVertexVersion`
- `meshopt_decodeVertexBuffer`
- `meshopt_decodeIndexVersion`
- `meshopt_decodeIndexBuffer`
- `meshopt_decodeIndexSequence`
- `meshopt_decodeFilterOct`
- `meshopt_decodeFilterQuat`
- `meshopt_decodeFilterExp`

处理逻辑：

- 根据 `cgltf_meshopt_compression_mode_attributes` 解压顶点属性。
- 根据 `cgltf_meshopt_compression_mode_triangles` 解压三角形索引。
- 根据 `cgltf_meshopt_compression_mode_indices` 解压 index sequence。
- 检查 glTF 规范期望的版本：vertex version 0、index version 1。
- 根据 filter 类型执行 octahedral、quaternion、exponential 后处理。
- 解压后的内存写回 `bufferView->data`，让后续 `cgltf_buffer_view_data` 和 accessor 读取逻辑可以像处理普通 buffer 一样处理压缩数据。

这说明本项目没有把 meshopt 压缩 glTF 当成特殊导入格式，而是在 buffer view 层把它还原成普通 glTF 数据。

### 顶点去重和重映射

主要文件：

- `src/scene/scene.cpp`

项目在导入 geometry 后用 meshoptimizer 做顶点 remap，减少重复顶点并重写 index buffer。

使用的 API 包括：

- `meshopt_Stream`
- `meshopt_generateVertexRemapMulti`
- `meshopt_generateVertexRemap`
- `meshopt_remapVertexBuffer`
- `meshopt_remapIndexBuffer`

逻辑上分两种情况：

- 如果 geometry 有多个属性流，使用 `meshopt_generateVertexRemapMulti`，把位置和属性都纳入唯一顶点判断。
- 如果只有位置流，则使用 `meshopt_generateVertexRemap`。

随后项目把 `vertexPositions`、`vertexAttributes` 和 `triangles` 按 remap 结果重排。这一步影响后续 LOD 构建、cluster 生成和 cache 数据大小。

### meshlet/cluster 切分

主要文件：

- `src/meshlod/lod.h`
- `src/meshlod/meshlod_clustering.h`
- `src/scene/clusterlod.cpp`

项目的 LOD 构建以 meshoptimizer 的 meshlet 构建能力为基础。关键 API：

- `meshopt_buildMeshletsBound`
- `meshopt_buildMeshletsSpatial`
- `meshopt_buildMeshletsFlex`
- `meshopt_Meshlet`
- `meshopt_optimizeMeshlet`

`clusterize(...)` 先用 `meshopt_buildMeshletsBound` 估计 meshlet 数量，再根据配置选择：

- `meshopt_buildMeshletsSpatial`：偏空间局部性的 cluster 切分。
- `meshopt_buildMeshletsFlex`：按 fill weight、split factor 和三角形/顶点上限做更灵活的切分。

可选调用 `meshopt_optimizeMeshlet` 优化 cluster 内部三角形和局部顶点顺序。项目随后把 meshoptimizer 的 meshlet 输出转换成自己的 `Cluster`，每个 cluster 记录局部三角形索引、顶点数、group/refined 信息。

### cluster/group 分区和空间排序

主要文件：

- `src/meshlod/lod.h`
- `src/meshlod/meshlod_clustering.h`

关键 API：

- `meshopt_partitionClusters`
- `meshopt_spatialSortRemap`

`partition(...)` 会把 pending clusters 按目标 group size 重新分区。启用 `partition_spatial` 时，它把顶点位置传给 `meshopt_partitionClusters`，让分区尽量保持空间局部性。启用 `partition_sort` 时，它用 cluster bounds center 调用 `meshopt_spatialSortRemap`，把分区顺序按空间排序重排。

这一步直接影响：

- group 内 cluster 的空间局部性。
- 后续 traversal 层次结构的访问局部性。
- streaming/preload 时 group 数据布局。

### 几何简化和误差度量

主要文件：

- `src/meshlod/lod.h`
- `src/meshlod/meshlod_simplify.h`

关键 API 和常量：

- `meshopt_simplifyWithAttributes`
- `meshopt_simplifySloppy`
- `meshopt_simplifyScale`
- `meshopt_SimplifySparse`
- `meshopt_SimplifyErrorAbsolute`
- `meshopt_SimplifyPermissive`
- `meshopt_SimplifyRegularize`
- `meshopt_SimplifyVertex_Protect`

项目用 `meshopt_simplifyWithAttributes` 生成低 LOD 索引，并把 normal、texcoord、tangent 等属性权重传给 meshoptimizer。配置项来自 `SceneConfig`，包括：

- `simplifyNormalWeight`
- `simplifyTexCoordWeight`
- `simplifyTangentWeight`
- `simplifyTangentSignWeight`

如果常规简化没有达到目标三角形数量，项目可启用：

- permissive fallback：追加 `meshopt_SimplifyPermissive`。
- sloppy fallback：调用 `meshopt_simplifySloppy`，然后用 `meshopt_simplifyScale` 把误差缩放到模型尺度。

`meshopt_SimplifyVertex_Protect` 用于保护边界、特征点或其它不能轻易合并的顶点。项目还在自己的简化逻辑中增加了 feature-aware 参数，例如 curvature、feature edge、perceptual weight、silhouette preservation，这些参数影响传给 meshoptimizer 的锁定点和权重。

### bounds 和层次结构辅助

主要文件：

- `src/meshlod/lod.h`
- `src/meshlod/meshlod_bounds.h`
- `src/scene/clusterlod.cpp`

关键 API：

- `meshopt_computeClusterBounds`
- `meshopt_computeSphereBounds`
- `meshopt_spatialClusterPoints`

用途：

- `meshopt_computeClusterBounds` 计算 cluster 的 bounding sphere、cone 或误差信息。
- `meshopt_computeSphereBounds` 合并多个 bounds，生成 group、node 或 LOD root 的包围球。
- `meshopt_spatialClusterPoints` 在构建 LOD node hierarchy 时对节点做空间聚类，减少 traversal 树的空间混乱。

这些结果会进入 `shaderio::Node::traversalMetric`，被 GPU traversal shader 用于 LOD 判定、视锥/遮挡剔除和层次遍历。

## cgltf

cgltf 是项目的 glTF/glb 解析库。项目没有使用 nvpro2 的 `nvvkgltf`，而是通过 `src/vendor/cgltf.cpp` 定义 `CGLTF_IMPLEMENTATION` 并包含 `<cgltf.h>`，让 cgltf 的单文件实现进入链接。

主要文件：

- `src/vendor/cgltf.cpp`
- `src/scene/scene_gltf.cpp`
- `src/scene/scene.hpp`

### 文件读取回调

项目没有让 cgltf 直接用默认文件 IO，而是提供了自定义回调：

- `cgltf_read`
- `cgltf_release`

回调内部通过 `FileMappingList` 管理 `nvutils::FileReadMapping`。这样做的效果是：

- glTF 主文件和外部 buffer 文件可以用 memory mapping 读取。
- 同一路径可复用 mapping，并通过 ref count 管理生命周期。
- cgltf 仍然通过标准 `cgltf_file_options` 接口访问数据。

### glTF 解析、校验和 buffer 加载

使用的 cgltf API：

- `cgltf_parse_file`
- `cgltf_validate`
- `cgltf_load_buffers`
- `cgltf_free`

项目用 `std::unique_ptr<cgltf_data, decltype(&cgltf_free)>` 管理 `cgltf_data` 生命周期。导入流程为：

1. 填充 `cgltf_options`，安装自定义 file callbacks。
2. 调用 `cgltf_parse_file` 解析 `.gltf` 或 `.glb`。
3. 如果返回 `cgltf_result_legacy_gltf`，直接拒绝 legacy glTF。
4. 调用 `cgltf_validate` 做结构校验。
5. 如果没有使用已有 cache，调用 `cgltf_load_buffers` 加载 buffer 数据。

### mesh、primitive、attribute 和 accessor 读取

项目遍历：

- `gltf->meshes`
- `cgltf_mesh::primitives`
- `cgltf_primitive::attributes`
- `cgltf_primitive::indices`

只处理 `cgltf_primitive_type_triangles`。属性按 glTF 名称识别：

- `POSITION`
- `NORMAL`
- `TANGENT`
- `TEXCOORD_0`
- `TEXCOORD_1`

accessor 读取分为快路径和通用路径：

- 快路径：当 `component_type == cgltf_component_type_r_32f`、`type` 匹配、`stride == sizeof(T)` 时，直接通过 `cgltf_buffer_view_data(accessor->buffer_view) + accessor->offset` 读取。
- 通用路径：调用 `cgltf_accessor_read_float`，让 cgltf 处理类型转换、stride、component layout。
- 索引快路径：如果是 32-bit scalar 且 offsetVertices 为 0，直接 memcpy。
- 索引通用路径：调用 `cgltf_accessor_read_index`。

读取过程中，项目会对位置和 UV 做可配置的 float mantissa drop bit 量化，并用 GLM 更新 bbox。

### scene graph、node transform、camera 和 material

使用的 cgltf 功能：

- `cgltf_node_transform_world`
- `cgltf_node_transform_local`
- `cgltf_camera_type_perspective`
- `cgltf_material`
- `cgltf_alpha_mode_blend`

项目会读取默认 scene 或 root nodes，然后递归 `addInstancesFromNodeGLTF(...)`：

- 用 `cgltf_node_transform_local` 得到 node local transform，再乘上 parent transform。
- mesh node 转成项目内部 `Instance`。
- 读取 material 的 alpha mode，影响透明/双面等实例属性。
- 读取 perspective camera，转换成项目内部相机列表，并用于 UI 的 home camera / camera presets。

### cgltf 与 meshoptimizer 的交界

cgltf 解析出 `cgltf_buffer_view::has_meshopt_compression` 和 `cgltf_meshopt_compression` 元数据，meshoptimizer 负责真正解压。也就是说：

- cgltf 负责理解 glTF 结构和扩展元信息。
- meshoptimizer 负责把扩展压缩 payload 解码为普通 buffer view 数据。
- 解压完成后，项目继续通过 cgltf accessor API 读取属性和索引。

## vulkan_radix_sort

项目在 `thirdparty/vulkan_radix_sort` 中包含了一个本地第三方库。根 `CMakeLists.txt` 没有 `add_subdirectory(thirdparty/vulkan_radix_sort)`，而是直接把 `thirdparty/vulkan_radix_sort/src/vk_radix_sort.cc` 编进主 executable，同时加入 `thirdparty/vulkan_radix_sort/include`。

主要文件：

- `thirdparty/vulkan_radix_sort/include/vk_radix_sort.h`
- `thirdparty/vulkan_radix_sort/src/vk_radix_sort.cc`
- `src/renderer/resources.hpp`
- `src/renderer/resources.cpp`
- `src/renderer/renderer.cpp`
- `src/renderer/renderer_clusters_lod.cpp`

### sorter 生命周期

`Resources` 持有 `VrdxSorter m_vrdxSorter`。初始化和释放：

- `vrdxCreateSorter`
- `vrdxDestroySorter`

`Resources::init` 中把 `VkDevice`、`VkPhysicalDevice`、`pipelineCache` 填入 `VrdxSorterCreateInfo`。这会创建 radix sort 所需 pipeline。`Resources::deinit` 中销毁 sorter。

### 存储需求和辅助 buffer

renderer 在启用 instance sorting 时调用：

- `vrdxGetSorterKeyValueStorageRequirements`

它根据 `m_renderInstances.size()` 查询 key-value sort 的临时存储需求，并创建 `m_sortingAuxBuffer`。这说明 sort 是 GPU 端执行，且需要额外 scratch/aux storage。

### key-value 排序命令

`RendererRasterClustersLod::render` 中启用 `m_config.useSorting` 时调用：

- `vrdxCmdSortKeyValue`

项目先通过 traversal presort compute shader 写出：

- `instanceSortKeys`
- `instanceSortValues`

然后 radix sort 对 key/value 进行排序。排序后 renderer/traversal 后续 pass 可以按排序后的 instance 顺序处理，以改善渲染或遍历的局部性。`USE_SORTING` shader 宏控制相关 shader 路径。

### shader 生成方式

`thirdparty/vulkan_radix_sort/CMakeLists.txt` 原本定义了用 `glslangValidator --vn` 生成 shader header 的流程，包括：

- `upsweep.comp`
- `spine.comp`
- `downsweep.comp`
- `downsweep_key_value_comp`

当前项目根 CMake 直接编入 `.cc`，而 `thirdparty/vulkan_radix_sort/src/generated` 下已经存在生成好的 header。因此项目依赖这些 generated shader header 已经存在，或者由第三方库自己的 CMake 流程预先生成。

## Vulkan SDK、volk 和 VMA

这些库/SDK 不是根 CMake 中单独链接的显式 target，但项目直接使用它们的能力，通常由 nvpro2/Vulkan SDK 提供。

### Vulkan API

主要文件：

- `src/app/main.cpp`
- `src/renderer/resources.*`
- `src/renderer/renderer.*`
- `src/renderer/renderer_clusters_lod.cpp`
- `src/renderer/hiz.*`
- `src/streaming/streaming.*`
- `src/streaming/streamutils.*`

项目大量直接调用 Vulkan API。核心用途包括：

- instance/device/swapchain 所需扩展配置。
- mesh shader 扩展：`VK_EXT_MESH_SHADER_EXTENSION_NAME`、`VK_NV_MESH_SHADER_EXTENSION_NAME`、`vkCmdDrawMeshTasksEXT`、`vkCmdDrawMeshTasksNV`、indirect mesh task draw。
- NVIDIA cluster acceleration structure：`VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME` 和相关 feature struct。
- shader clock、atomic float、fragment shading rate、barycentric、64-bit image atomic 等扩展。
- dynamic rendering、descriptor update、pipeline creation、command buffer、queue submit、timeline semaphore。
- storage buffer、uniform buffer、indirect buffer、storage image、Hi-Z image、framebuffer image。

项目不是把 Vulkan 完全封装到 nvvk 里；它使用 nvvk 管理常用对象，但仍直接调用底层 Vulkan 命令来控制渲染和同步。

### volk

主要文件：

- `src/app/main.cpp`
- 多个 renderer/streaming 源文件包含 `<volk.h>`

`main.cpp` 调用 `volkInitialize()`。由于项目定义/处理 `VK_NO_PROTOTYPES`，volk 负责加载 Vulkan 函数指针，使后续 `vk*` 调用可用。

### VMA

主要文件：

- `src/app/main.cpp`
- `src/renderer/resources.hpp`
- `src/renderer/resources.cpp`
- `src/streaming/streamutils.cpp`

项目在 `main.cpp` 定义 `VMA_IMPLEMENTATION`，并在 debug 下定义 `VMA_LEAK_LOG_FORMAT`。资源创建时使用：

- `VmaAllocatorCreateInfo`
- `VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT`
- `VmaMemoryUsage`
- `VmaAllocationCreateFlags`

项目通常不直接调用 VMA 的裸 API，而是通过 `nvvk::ResourceAllocator` 调用 `createBuffer`、`createImage`、`destroyBuffer`、`destroyImage`。VMA 的实际作用是承担 GPU/CPU memory allocation policy，尤其是 device address buffer、CPU visible readback/upload buffer 和 device-local storage buffer。

## GLM

主要文件：

- `src/scene/*`
- `src/app/lodclusters_runtime.cpp`
- `src/app/lodclusters_scene.cpp`
- `src/renderer/*`
- `src/streaming/streamutils.cpp`

GLM 是项目 CPU 侧数学库。主要用途：

- `glm::vec2`、`glm::vec3`、`glm::vec4`、`glm::uvec2`、`glm::uvec3` 表示位置、法线、UV、颜色、viewport、dispatch grid。
- `glm::mat3`、`glm::mat4`、`glm::mat4x3` 表示实例矩阵、view/projection、normal matrix、shader 常量矩阵。
- `glm::perspectiveRH_ZO` 构造 Vulkan 风格 reversed-Z projection。
- `glm::inverse`、`glm::determinant`、`glm::normalize`、`glm::length`、`glm::distance`、`glm::dot`、`glm::min`、`glm::max` 处理相机、bbox、剔除参数和 cluster bounds。
- `glm::packUnorm4x8` 把实例颜色压缩为 shader 使用的 packed color。
- `glm::value_ptr` 把 `glm::mat4` 传给 cgltf transform 函数。
- `glm::radians`、`glm::degrees` 处理相机 fovy。

GLM 数据类型也贯穿 CPU/GPU 共享结构，例如 `FrameConstants`、`GeometryStorage`、`Instance` 和 `shaderio` 相关布局。

## ImGui 和 ImPlot

主要文件：

- `src/app/main.cpp`
- `src/app/lodclusters_ui.cpp`
- `src/app/lodclusters_lifecycle.cpp`

ImGui 用于整个应用 UI：

- docking 布局：`ImGui::DockBuilderSplitNode`、`ImGui::DockBuilderDockWindow`。
- viewport texture：`ImGui::Image` 显示渲染结果。
- 菜单和快捷键：`BeginMenu`、`MenuItem`、`IsKeyChordPressed`。
- 设置面板：`CollapsingHeader`、`BeginTable`、`Checkbox`、`SliderFloat`、`InputFloat`、`BeginCombo`、`Selectable`。
- loading modal：`OpenPopup`、`BeginPopupModal`、`ProgressBar`。
- debug/readback 面板：表格、tooltip、数值显示。
- 与 Vulkan backend 交互：`ImGui_ImplVulkan_AddTexture` 把 framebuffer image view 注册成 ImGui texture。

ImPlot 用于绘制统计曲线，例如 streaming memory 历史图。项目调用：

- `ImPlot::BeginPlot`
- `ImPlot::SetupLegend`
- `ImPlot::SetupAxes`
- `ImPlot::SetupAxesLimits`
- `ImPlot::PlotShaded`
- `ImPlot::GetPlotMousePos`

## fmt

主要文件：

- `src/app/lodclusters_lifecycle.cpp`
- `src/app/lodclusters_ui.cpp`
- `src/scene/scene_gltf.cpp`
- `src/renderer/renderer.cpp`
- `src/renderer/renderer_clusters_lod.cpp`
- `src/streaming/streaming.cpp`

`fmt::format` 主要用于：

- 构造 UI/日志字符串。
- 构造 profiler/sequencer memory report。
- 构造 cluster preset 文本，例如 `"{}T_{}V"`。
- 构造 shader macro 值，例如 subgroup size、meshlet triangle/vertex count、feature flags。
- 构造 glTF mesh 去重 identifier，把 accessor 指针组合成字符串。

fmt 的作用比较轻量，但出现在 shader 编译宏和日志统计中，间接影响 renderer 配置变更后的 shader 编译路径。

## shaderc

shaderc 通过 `nvvkglsl` 使用，也在代码中显式出现：

- `shaderc::CompileOptions`
- `shaderc::SpvCompilationResult`

主要用途：

- 每个 renderer/streaming/Hi-Z 模块保存自己的 `SpvCompilationResult`。
- 初始化 pipeline 前通过 `CompileOptions::AddMacroDefinition` 注入编译宏。
- Hi-Z 模块通过 `appendShaderDefines` 给不同 shader variant 添加宏。

shaderc 的输出由 `nvvkglsl::GlslCompiler` 转换给 Vulkan shader module 创建流程。

## 依赖之间的数据流关系

可以把外部库在项目中的关系概括为以下流水线：

```text
cgltf
  解析 glTF/glb 结构、accessor、node、camera、material
  |
  v
meshoptimizer
  解压 EXT_meshopt_compression
  顶点去重/remap
  meshlet/cluster 构建
  cluster/group 分区
  属性感知简化
  bounds 和空间层次辅助
  |
  v
nvutils
  并行处理、cache 文件映射、日志、参数、计时
  |
  v
nvvk + Vulkan + VMA
  创建 GPU buffer/image/descriptors/pipelines
  上传 preload/streaming 数据
  运行 traversal、raster、Hi-Z、streaming compute
  |
  v
vulkan_radix_sort
  可选对 instance sort key/value 执行 GPU radix sort
  |
  v
nvapp + nvgui + ImGui/ImPlot
  应用生命周期、相机、菜单、参数 UI、profiler、统计面板
```

## 需要注意的实现细节

- 项目显式关闭 `nvvkgltf`，所以 glTF 解析完全走 `cgltf + 自定义 Scene::loadGLTF`。
- `meshoptimizer` 同时用于导入阶段和 LOD 构建阶段，不能只把它理解成“压缩 glTF 解码库”。
- `vulkan_radix_sort` 不是通过它自己的 CMake target 链接，而是直接把 `.cc` 编入主程序；generated shader header 必须可用。
- `nvvk::ResourceAllocator` 隐藏了大量 VMA 细节，但 `VMA_IMPLEMENTATION` 在 `main.cpp` 中启用，说明内存分配实际由 VMA 支撑。
- shader 编译依赖运行时配置宏，renderer 或 streaming 配置改变可能需要重新编译 shader/pipeline。
- `nvutils::align_up`、`nvvk::BufferTyped<T>`、`shaderio` 共享头共同决定 CPU/GPU 数据布局，外部库调用不能脱离这些布局约束理解。
- 项目直接调用 Vulkan API 的比例很高；nvvk 是辅助封装，不是完整抽象层。
