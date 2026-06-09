# Vulkan Cluster LOD Renderer

这是一个基于 Vulkan 的 Cluster LOD 渲染实验项目。项目从 `.gltf` / `.glb` 模型导入几何数据，在 CPU 侧构建 cluster、group 和多级 LOD 层次结构，然后在 GPU 侧通过 compute shader、mesh shader、Hi-Z、可选 streaming 和可选 software raster 路径完成遍历、剔除、排序与渲染。

项目目标不是做通用模型查看器，而是验证和调试大规模几何场景的 Cluster LOD 数据组织、运行时遍历、可见性剔除、显存驻留和流式加载策略。

## 功能概览

- glTF / GLB 场景导入，支持读取 mesh、node、material、camera 和实例层级。
- 支持 `EXT_meshopt_compression` 数据解压。
- CPU 侧构建 cluster / group / LOD hierarchy。
- 基于 meshoptimizer 的 meshlet 切分、cluster 分区、属性感知简化和 bounds 计算。
- 支持 scene cache，避免每次启动重复做离线 LOD 构建。
- 支持 processing-only 模式，用于提前生成 cache。
- 支持 preload 和 streaming 两种几何驻留模式。
- 支持 Vulkan mesh shader 渲染路径，优先使用 EXT mesh shader，也可使用 NV mesh shader 路径。
- 支持 Hi-Z、视锥/遮挡剔除、two-pass culling、primitive culling 等遍历优化。
- 支持可选 GPU radix sort，对实例 sort key/value 做排序。
- 支持可选 hybrid software/hardware raster 路径。
- 提供 ImGui / ImPlot 调试界面，用于调节渲染、LOD、streaming、压缩、cache 和统计参数。

## 目录结构

```text
.
+-- CMakeLists.txt
+-- cmake/
|   +-- FindNvproCore2.cmake
+-- docs/
|   +-- external-libraries-analysis.md
|   +-- key-functions-blog.md
|   +-- src-shaders-reference.md
+-- shaders/
|   +-- build/
|   +-- common/
|   +-- debug/
|   +-- interface/
|   +-- post/
|   +-- render/
|   +-- streaming/
|   +-- traversal/
+-- src/
|   +-- app/
|   +-- core/
|   +-- meshlod/
|   +-- renderer/
|   +-- scene/
|   +-- streaming/
|   +-- vendor/
+-- thirdparty/
    +-- vulkan_radix_sort/
```

主要模块：

- `src/app`：应用入口、参数注册、生命周期、UI、场景加载和每帧调度。
- `src/core`：cache 序列化和文件映射相关逻辑。
- `src/scene`：glTF 导入、几何预处理、cluster LOD 构建、压缩和场景实例化。
- `src/meshlod`：基于 meshoptimizer 的 LOD 构建核心逻辑。
- `src/renderer`：Vulkan 资源、framebuffer、Hi-Z、preload renderer 和 Cluster LOD renderer。
- `src/streaming`：几何流式加载、驻留管理、传输任务和 streaming shader 管线。
- `shaders/interface`：CPU/GPU 共享结构定义。
- `shaders/render`、`shaders/traversal`、`shaders/streaming`、`shaders/post`：主要 GPU 侧逻辑。

## 外部依赖

项目使用 CMake 构建，核心依赖由 `nvpro_core2` 提供。`cmake/FindNvproCore2.cmake` 会优先查找本地 `nvpro_core2`，找不到时默认下载固定提交：

```text
03297bee1f997208951cc104abb5ecc3e1f987ed
```

主要依赖：

- Vulkan SDK `>= 1.4.309.0`
- CMake `>= 3.22`
- C++20 编译器
- Git
- nvpro2：`nvapp`、`nvgui`、`nvutils`、`nvvk`、`nvvkglsl`
- meshoptimizer
- cgltf
- GLM
- ImGui / ImPlot
- shaderc
- volk
- Vulkan Memory Allocator
- `thirdparty/vulkan_radix_sort`

项目明确关闭了 nvpro2 的部分可选模块：

```cmake
NVPRO2_ENABLE_nvgl OFF
NVPRO2_ENABLE_nvgpu_monitor OFF
NVPRO2_ENABLE_nvslang OFF
NVPRO2_ENABLE_nvvkgltf OFF
```

因此 glTF 导入走项目自己的 `cgltf + meshoptimizer` 路径，而不是 `nvvkgltf`。

更完整的外部库使用分析见 [docs/external-libraries-analysis.md](docs/external-libraries-analysis.md)。

## 构建

在 Windows + Visual Studio 环境中，可以使用：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

也可以使用 Ninja：

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

当前 CMake target 名来自目录名。因此在当前路径 `t1` 下，生成的可执行目标通常叫 `t1`。

如果不想让 CMake 自动下载 nvpro2，可以提前把 `nvpro_core2` 放在以下任一路径：

- `build/_deps/nvpro_core2`
- 当前项目目录下的 `nvpro_core2`
- 当前项目上级或上上级目录下的 `nvpro_core2`

也可以通过 CMake cache 覆盖：

```powershell
cmake -S . -B build -DNVPROCORE2_DOWNLOAD=OFF
```

## 运行

构建后从生成目录运行可执行文件。项目默认会尝试在运行目录、源码资源目录或复制到 runtime 的资源目录里查找 `a.glb`。

指定模型：

```powershell
.\build\Release\t1.exe --scene path\to\model.glb
```

加载配置文件：

```powershell
.\build\Release\t1.exe --scene path\to\scene.cfg
```

只生成 cache，不打开窗口：

```powershell
.\build\Release\t1.exe --scene path\to\model.glb --processingonly 1
```

允许 processing-only 断点续跑：

```powershell
.\build\Release\t1.exe --scene path\to\model.glb --processingonly 1 --processingpartial 1
```

启用 streaming：

```powershell
.\build\Release\t1.exe --scene path\to\model.glb --streaming 1
```

指定 GPU：

```powershell
.\build\Release\t1.exe --device 0
```

启用 validation：

```powershell
.\build\Release\t1.exe --validation 1
```

## 常用参数

基础：

- `--scene`：加载 `.gltf`、`.glb` 或 `.cfg`。
- `--renderer`：选择 renderer 类型。
- `--verbose`：输出更多日志。
- `--debugui`：显示 Debug 面板。
- `--vsync`：启用或关闭 VSync。
- `--device`：按 index 强制选择 Vulkan 设备。
- `--headless`：无窗口运行。
- `--headlessframes`：headless 模式运行帧数。
- `--dumpspirv`：把编译后的 SPIR-V 输出到工作目录。

场景和 LOD：

- `--clusterconfig`：选择 cluster preset。
- `--clustergroupsize`：设置每组 cluster 数量。
- `--loderror`：设置屏幕空间 LOD pixel error。
- `--lodnodewidth`：设置 LOD hierarchy node width。
- `--loddecimationfactor`：设置 LOD 递减系数。
- `--meshoptfillweight`：传给 meshoptimizer cluster 构建的 fill weight。
- `--simplifyuvweight`：UV 在属性感知简化中的权重。
- `--simplifynormalweight`：normal 在属性感知简化中的权重。
- `--simplifytangentweight`：tangent 在属性感知简化中的权重。
- `--simplifytangentsignweight`：tangent sign 在属性感知简化中的权重。
- `--curvatureadaptive`、`--curvaturewindow`、`--featureedge`、`--perceptualweight`、`--silhouettepreserve`：feature-aware simplification 参数。

cache 和预处理：

- `--autosavecache`：加载后自动保存 cache。
- `--autoloadcache`：启动时自动加载已有 cache。
- `--mappedcache`：使用 memory mapped cache。
- `--processingonly`：只处理并保存 cache，随后退出。
- `--processingpartial`：processing-only 模式下允许部分保存和续跑。
- `--processingmode`：控制外层/内层并行策略，`0` 自动，`-1` inner，`1` outer。
- `--processingthreadpct`：初始加载和处理时使用的线程比例。
- `--forcepreprocessmegabytes`：超过指定规模时要求先预处理。
- `--cachesuffix`：cache 文件后缀，默认 `.zippp`。

压缩：

- `--compressed`：启用项目内部 group 数据压缩。
- `--compressedpositionbits`：position mantissa drop bits。
- `--compressedtexcoordbits`：texcoord mantissa drop bits。

渲染和剔除：

- `--visualize`：选择可视化模式。
- `--culling`：启用视锥和遮挡剔除。
- `--twopassculling`：启用 two-pass culling。
- `--primitiveculling`：在 NV mesh shader 路径启用 primitive culling。
- `--forcedinvisculling`：强制 invisible culling。
- `--separategroups`：使用 separate groups kernel。
- `--instancesorting`：启用 instance sorting。
- `--renderstats`：启用渲染统计。
- `--extmeshshader`：使用 EXT mesh shader 路径。
- `--facetshading`：启用 facet shading。
- `--flipwinding`：翻转 winding。
- `--forcetwosided`：强制双面。

streaming：

- `--streaming`：启用 streaming。
- `--maxtransfermegabytes`：每帧 transfer buffer 预算。
- `--maxblascachingmegabytes`：BLAS cache 预算。
- `--maxgeomegabytes`：几何数据预算。
- `--maxresidentgroups`：最大驻留 group 数。
- `--maxframeloadrequests`：每帧最大加载请求数。
- `--maxframeunloadrequests`：每帧最大卸载请求数。

software raster：

- `--swraster`：启用 compute software raster 路径。
- `--adaptiveraster`：启用自适应 SW/HW raster routing。
- `--swrasterdensity`：software raster triangle density 阈值。
- `--swrasterfeedback`：启用 feedback auto-tune。
- `--swrastertargetshare`：目标 software raster triangle share。

相机和光照：

- `--camerastring`：从字符串恢复相机。
- `--cameraspeed`：设置相机移动速度。
- `--sundirection`：设置太阳方向。
- `--suncolor`：设置太阳颜色。
- `--shadowray`：启用或关闭 shadow ray。

## 数据流程

```text
glTF / GLB / cache
  -> Scene::init
  -> cgltf 解析场景结构
  -> meshoptimizer 解压 EXT_meshopt_compression
  -> 顶点读取、量化、去重和 remap
  -> meshoptimizer / meshlod 构建 cluster、group 和 LOD
  -> Scene cache 保存或加载
  -> RenderScene 选择 preload 或 streaming
  -> Resources 创建 Vulkan buffer、image、descriptor 和 pipeline
  -> renderer 执行 traversal、culling、Hi-Z、sorting 和 mesh shader render
  -> ImGui / ImPlot 显示 viewport、设置和统计
```

## Shader

shader 文件在运行时通过 `nvvkglsl::GlslCompiler` 和 shaderc 编译。编译时会根据 renderer、streaming、mesh shader 类型、subgroup size、cluster 大小等配置注入宏。

主要 shader 目录：

- `shaders/interface`：CPU/GPU 共享结构和 binding 定义。
- `shaders/common`：剔除、Hi-Z、属性编码等公共逻辑。
- `shaders/render`：mesh shader 渲染和 fragment shading。
- `shaders/traversal`：LOD traversal、presort、init、run。
- `shaders/streaming`：streaming setup、update scene、age filter。
- `shaders/build`：scene build setup。
- `shaders/post`：fullscreen、background、atomic raster、Hi-Z。
- `shaders/debug`：instance / cluster bbox debug rendering。

源码和 shader 文件说明见 [docs/src-shaders-reference.md](docs/src-shaders-reference.md)。

## UI 使用

运行程序后主要面板包括：

- `Viewport`：显示渲染结果。
- `Settings`：渲染、LOD、cluster、streaming 和压缩参数。
- `Misc Settings`：相机、光照和高级参数。
- `Statistics`：scene、traversal、memory 和 cluster 统计。
- `Streaming memory`：streaming 驻留和内存曲线。
- `Profiler`：CPU/GPU profiler 视图。
- `Log`：运行日志。
- `Debug`：shader readback 和调试值。

菜单支持打开模型、重新加载、保存/删除 cache、重编 shader、切换 VSync 和退出。

## 开发备注

- `build`、`_bin` 被 `.gitignore` 忽略。
- `thirdparty/vulkan_radix_sort` 的 `.cc` 被直接编入主可执行文件，不是通过它自己的 CMake target 链接。
- `thirdparty/vulkan_radix_sort/src/generated` 下的 shader header 需要存在。
- 项目要求 Vulkan header version 至少为 309，对应 Vulkan SDK `>= 1.4.309.0`。
- CMake 会把 `shaders` 和 nvshaders 相关文件复制到 runtime/install 目录。
- 修改 `shaders/interface` 中 CPU/GPU 共享结构时，需要同步检查 C++ 侧结构、buffer layout、descriptor binding 和 shader 使用点。
- 修改 LOD/cluster 配置会触发 scene 或 renderer 重建，较大模型可能耗时明显，建议先使用 processing-only 生成 cache。

## 相关文档

- [docs/external-libraries-analysis.md](docs/external-libraries-analysis.md)：外部库功能使用分析。
- [docs/src-shaders-reference.md](docs/src-shaders-reference.md)：源码和 shader 文件职责说明。
- [docs/key-functions-blog.md](docs/key-functions-blog.md)：从关键函数视角理解项目主流程。
