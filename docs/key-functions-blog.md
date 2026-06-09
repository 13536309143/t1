# 从函数视角理解 Vulkan Cluster LOD 项目

这篇文章按项目主链路介绍常用且关键的函数。这里的“常用”不是简单的调用次数统计，而是指函数在数据流中反复承担核心职责：离线构建 Cluster LOD、加载和缓存场景、压缩 group 数据、创建 Vulkan 资源、执行流式加载，以及在 shader 中做 LOD 判断、剔除、属性解码和着色。

项目整体可以概括为一条流水线：

```text
glTF/缓存输入
  -> Scene 解析与 GeometryStorage 构建
  -> meshlod 生成 cluster/group/LOD 层级
  -> group 压缩或运行时布局
  -> RenderScene 选择 preload 或 streaming
  -> Renderer 录制 traversal/render/Hi-Z 命令
  -> GLSL shader 根据可见性、误差和驻留状态输出最终 cluster
```

下面按模块介绍函数的表达式和作用。

## 一、Mesh LOD 构建函数

### 1. `clodDefaultConfig(size_t max_triangles)`

位置：`src/meshlod/meshlod_build.h`

表达式：

```cpp
clodConfig config = {};
config.max_vertices = max_triangles;
config.min_triangles = max_triangles / 3;
config.max_triangles = max_triangles;
config.partition_size = 16;
config.simplify_ratio = 0.5f;
config.simplify_threshold = 0.85f;
```

作用：生成 Cluster LOD 构建的默认参数。它决定单个 cluster 的三角形/顶点预算、group 分区规模、简化比例、fallback 策略以及特征保护强度。后续 `clodBuild`、`clusterize`、`partition`、`simplify` 都依赖这些参数。

### 2. `clodBuild(...)`

位置：`src/meshlod/meshlod_build.h`

表达式：

```cpp
context.clusters = clusterize(config, mesh, mesh.indices, mesh.index_count, &context.feature_importance);
while (context.pending.size() > 1) {
  context.groups = partition(config, mesh, context.clusters, context.pending, context.remap);
  lockBoundary(...);
  clodBuild_iterationTask(...);
  context.depth++;
}
```

作用：这是 CPU 侧 LOD 构建入口。它先把原始三角形切成 cluster，再反复把待处理 cluster 分组、锁定边界、简化成更粗层级，直到只剩根 group。它把普通 mesh 转换为运行时可遍历的层次化 Cluster LOD 数据。

### 3. `clodBuild_iterationTask(...)`

位置：`src/meshlod/meshlod_build.h`

表达式：

```cpp
target_size = size_t((merged.size() / 3) * config.simplify_ratio) * 3;
simplified = simplify(config, mesh, merged, locks, feature_importance, target_size, &error);
bounds.error = max(previous_error, error) + error * additive_factor;
refined = outputGroup(...);
split = clusterize(config, mesh, simplified.data(), simplified.size(), &feature_importance);
```

作用：处理一组 group 的单次 LOD 迭代。它合并 group 内 cluster 的索引，按目标三角形数量简化，再把简化结果重新切成新 cluster，作为下一层级输入。并行构建时，这个函数就是每个任务的工作单元。

### 4. `clusterize(...)`

位置：`src/meshlod/meshlod_clustering.h`

表达式：

```cpp
max_meshlets = meshopt_buildMeshletsBound(index_count, max_vertices, min_triangles);
meshlets = meshopt_buildMeshletsSpatial(...) 或 meshopt_buildMeshletsFlex(...);
cluster.indices[j] = meshlet_vertices[meshlet.vertex_offset + meshlet_triangles[...]];
```

作用：把一段 index buffer 切成 meshlet/cluster。每个 cluster 保持有限的顶点数和三角形数，使 GPU mesh shader 或软件光栅阶段可以用固定规模的工作组处理。

### 5. `clusterFeatureImportance(...)`

位置：`src/meshlod/meshlod_clustering.h`

表达式：

```cpp
importance = min(1, avg * 0.5 + max_feature * 0.3 + strong_ratio * 0.2);
```

作用：把顶点级特征重要性汇总成 cluster 级重要性。它影响后续分区和简化策略，使边界、尖锐边、薄片、小孔等视觉敏感区域不被过度合并。

### 6. `partition(...)`

位置：`src/meshlod/meshlod_clustering.h`

表达式：

```cpp
partition_count = meshopt_partitionClusters(..., config.partition_size);
feature_pressure = avg_feature * 0.45 + max_feature * 0.35 + strong_ratio * 0.2;
feature_limit = partition_size * (1 - 0.45 * strength * feature_pressure);
```

作用：把待处理 cluster 列表切成 group。默认按空间和拓扑关系分组；如果特征压力较高，会缩小 group 尺寸，减少简化跨越强特征区域的概率。

### 7. `lockBoundary(...)`

位置：`src/meshlod/meshlod_clustering.h`

表达式：

```cpp
locks[r] |= locks[r] >> 7;
locks[r] |= 1 << 7;
locks[i] = (locks[r] & 1) | (locks[i] & meshopt_SimplifyVertex_Protect);
```

作用：标记 group 边界顶点，防止简化时破坏 group 之间的拓扑连续性。它还会合并外部传入的 `vertex_lock` 和属性保护锁。

### 8. `computeFeatureImportance(...)`

位置：`src/meshlod/meshlod_simplify.h`

表达式：

```cpp
weighted =
  boundary * 0.28 +
  hole_loop * 0.18 +
  non_manifold * 0.22 +
  sharp_edge * 0.22 +
  normal_variation * 0.18 +
  thin_wall * 0.14;
importance[v] = max(importance[v], clamp01(weighted));
```

作用：计算每个顶点的视觉/拓扑重要性。它会检测边界、非流形边、法线变化、细长三角形、小尺度边、受保护属性差异等因素，是项目中“特征感知简化”的核心。

### 9. `simplify(...)`

位置：`src/meshlod/meshlod_simplify.h`

表达式：

```cpp
adaptive_target = featureAdaptiveTarget(config, indices, feature_importance, target_count);
lod = meshopt_simplifyWithAttributes(..., adaptive_target, ...);
if (lod.size() > adaptive_target) simplifyFallback(...);
*error = perceptualError(...) * perceptual_weight + *error * (1 - perceptual_weight);
*error *= feature_error_scale;
```

作用：执行真正的 mesh 简化。它不仅按几何误差简化，还考虑顶点属性权重、保护锁、特征区域自适应目标数量、fallback 策略和感知误差修正。

### 10. `featureAdaptiveTarget(...)`

位置：`src/meshlod/meshlod_simplify.h`

表达式：

```cpp
pressure = clamp01(avg_feature * 0.7 + max_feature * 0.3);
strength = clamp01(curvature_adaptive_strength + silhouette_preservation);
relaxed = target_count + (indices.size() - target_count) * preserve * strength * pressure;
```

作用：根据特征强度放宽简化目标。特征越强，目标三角形数越接近原始数量，从而保留更多形状细节。

### 11. `perceptualError(...)`

位置：`src/meshlod/meshlod_simplify.h`

表达式：

```cpp
perceptual = geometric_error * pow(vertex_count / original_count, 0.3f);
```

作用：把几何误差转换为更接近视觉感知的误差。简化后顶点越少，同样的几何误差在 LOD 判断中会被更谨慎地处理。

### 12. `boundsCompute(...)`

位置：`src/meshlod/meshlod_bounds.h`

表达式：

```cpp
meshopt_Bounds b = meshopt_computeClusterBounds(indices, positions, ...);
result = { b.center, b.radius, error };
```

作用：计算 cluster 的包围球和误差。这个结果最终进入 `TraversalMetric`，供 shader 判断某个 LOD 节点是否需要继续细分。

### 13. `boundsMerge(...)`

位置：`src/meshlod/meshlod_bounds.h`

表达式：

```cpp
merged = meshopt_computeSphereBounds(child_centers, child_radii);
result.error = max(child.bounds.error);
```

作用：把多个 child cluster/group 的 bounds 合成父级 bounds。合并后的包围球用于父节点可见性和误差判断。

### 14. `outputGroup(...)`

位置：`src/meshlod/meshlod_build.h`

表达式：

```cpp
clodGroup group = { depth, simplified_bounds };
output_callback(output_context, group, clusters, cluster_count, task_index, thread_index);
```

作用：把构建阶段内部的 group 和 cluster 转换为对外 API 的输出格式。上层 `Scene::buildGeometryLod` 通过回调接收这些结果。

### 15. `clodLocalIndices(...)`

位置：`src/meshlod/meshlod_local_indices.h`

表达式：

```cpp
for each global index v:
  local = first vertices[local] == v
  if not found: vertices[unique++] = v
  triangles[i] = uint8_t(local)
```

作用：把 cluster 内的全局索引转换为局部顶点表和 8-bit 三角形索引。这样 group 数据更紧凑，shader 访问也更适合固定上限布局。

## 二、Scene、缓存与压缩函数

### 16. `Scene::init(...)`

位置：`src/scene/scene.cpp`

表达式：

```cpp
openCache();
loadGLTF(processingInfo, filePath);
processGeometry(...);
saveCache();
updateSceneGrid(...);
```

作用：场景加载总入口。它负责 cache 判断、glTF 解析、geometry 处理、LOD 构建、压缩、实例和统计信息建立。应用层加载新模型时最终都会落到这个函数。

### 17. `Scene::loadGLTF(...)`

位置：`src/scene/scene_gltf.cpp`

表达式：

```cpp
cgltf_parse_file(...);
cgltf_validate(...);
cgltf_load_buffers(...);
loadGeometryGLTF(...);
addInstancesFromNodeGLTF(...);
```

作用：把 `.gltf` 或 `.glb` 解析为项目内部的 geometry、material、camera 和 instance。它是外部资产进入项目数据结构的主要入口。

### 18. `Scene::loadGeometryGLTF(...)`

位置：`src/scene/scene_gltf.cpp`

表达式：

```cpp
readAttributesGLTF(...);
indices -> geometry.triangleIndices;
positions/normals/tangents/texcoords -> geometry vertex arrays;
```

作用：加载 glTF mesh primitive。它读取索引、顶点位置、法线、切线、纹理坐标和材质映射，并为后续 LOD 构建准备 `GeometryStorage`。

### 19. `Scene::processGeometry(...)`

位置：`src/scene/scene.cpp`

表达式：

```cpp
if (isCached) loadCachedGeometry(...);
else {
  buildGeometryDedupVertices(...);
  buildGeometryLod(...);
  compressGroup(...);
}
```

作用：处理单个 geometry。它决定是否复用缓存；如果不能复用，则执行顶点去重、LOD 构建、group 压缩和统计更新。

### 20. `Scene::buildGeometryLod(...)`

位置：`src/scene/clusterlod.cpp`

表达式：

```cpp
clodBuild(config, mesh, output_context, output_callback, iteration_callback);
group/cluster outputs -> Scene::GroupStorage / Node / LodLevel
```

作用：把 `meshlod` 输出接入 `Scene` 的 runtime 布局。它把通用 LOD 构建器生成的 group、cluster、bounds 和 error 转换为本项目 traversal shader 能读取的数据。

### 21. `Scene::fillGroupRuntimeData(...)`

位置：`src/scene/scene.cpp`

表达式：

```cpp
shaderio::Group header;
shaderio::Cluster clusters[];
vertex/index/attribute payload;
```

作用：把 CPU 侧 group view 写成 shader 侧运行时布局。preload 和 streaming 两种路径最终都需要得到这种统一布局。

### 22. `Scene::compressGroup(...)`

位置：`src/scene/scene_cluster_compression.cpp`

表达式：

```cpp
raw group data -> quantized vertices / packed attributes / bitstream
if compressed_size >= raw_size: keep uncompressed
```

作用：压缩 group 数据，减少 cache、系统内存和 streaming 上传量。它会在压缩收益不足时回退到未压缩布局，避免为了压缩牺牲空间或复杂度。

### 23. `Scene::decompressGroup(...)`

位置：`src/scene/scene_cluster_compression.cpp`

表达式：

```cpp
if compressed:
  read bitstream -> reconstruct shaderio::Group layout
else:
  copy raw group data
```

作用：在上传或运行时访问前，把压缩 group 还原为 shader 消费的标准布局。streaming 加载 group 时会调用它生成 GPU buffer 内容。

### 24. `serialization::getCachedSize(...)`

位置：`src/core/serialization.hpp`

表达式：

```cpp
cached_size = align16(view.size_bytes()) + 16;
```

作用：计算 span 写入缓存后的字节数。额外 16 字节用于保存元素数量，数据区按 16 字节对齐，便于后续 memory-mapped cache 直接读取。

### 25. `serialization::storeAndAdvance(...)`

位置：`src/core/serialization.hpp`

表达式：

```cpp
write count as 16-byte header;
write data bytes;
dataAddress = align16(dataAddress + data_size);
```

作用：把连续数组写入 cache buffer，并推进写游标。它维护缓存布局的对齐约束，是 `Scene::storeCached` 的基础工具。

### 26. `serialization::loadAndAdvance(...)`

位置：`src/core/serialization.hpp`

表达式：

```cpp
count = *reinterpret_cast<uint64_t*>(dataAddress);
view = span(basePointer, count);
dataAddress = align16(dataAddress + sizeof(T) * count);
```

作用：从 cache buffer 恢复只读 span，并推进读游标。它不复制数据，span 生命周期依赖底层缓存映射。

## 三、应用层与渲染资源函数

### 27. `LodClusters::initScene(...)`

位置：`src/app/lodclusters_scene.cpp`

表达式：

```cpp
std::thread([=] {
  Scene::init(filePath, sceneConfig, loaderConfig);
});
```

作用：应用层异步加载场景。它把 UI/命令行中的路径和配置传入 `Scene::init`，并在加载完成后触发 render scene 和 renderer 重新初始化。

### 28. `LodClusters::handleChanges(...)`

位置：`src/app/lodclusters_runtime.cpp`

表达式：

```cpp
if scene/config changed: deinit/init scene or render scene
if renderer config changed: deinit/init renderer
if framebuffer size changed: initFramebuffer(...)
```

作用：集中处理运行时配置变化。UI、命令行、拖拽文件、renderer 切换、streaming 参数变化都会在这里被转换为明确的重建动作。

### 29. `LodClusters::onRender(...)`

位置：`src/app/lodclusters_runtime.cpp`

表达式：

```cpp
Resources::beginFrame(...);
update FrameConstants;
renderer->render(cmd, resources, frame, renderScene);
Resources::postProcessFrame(...);
```

作用：每帧 CPU 渲染入口。它更新 camera、projection、LOD threshold、Hi-Z 参数、readback 参数等 frame constants，然后驱动 renderer 录制 GPU 命令。

### 30. `Resources::initFramebuffer(...)`

位置：`src/renderer/resources.cpp`

表达式：

```cpp
renderSize = windowSize * supersample_scale;
create color/depth/resolved/rasterAtomic/hiz images;
updateFramebufferRenderSizeDependent(...);
```

作用：根据窗口大小和超采样设置创建 framebuffer 相关 Vulkan image。renderer、Hi-Z、后处理和 ImGui viewport 都依赖这些资源。

### 31. `Resources::compileShader(...)`

位置：`src/renderer/resources.cpp`

表达式：

```cpp
shaderc options + include paths + macros -> SPIR-V
```

作用：统一编译 GLSL shader。renderer 会给它传入 cluster 大小、streaming、sorting、culling、mesh shader 类型等宏，生成对应变体。

### 32. `Resources::cmdBuildHiz(...)`

位置：`src/renderer/resources.cpp`

表达式：

```cpp
m_hizUpdate.update source depth;
NVHizVK::cmdUpdateHiz(cmd, update, descriptorSet);
```

作用：录制 Hi-Z 生成命令，把当前深度 buffer 归约成多级最大/最小深度纹理。遍历 shader 使用它进行遮挡剔除。

### 33. `RenderScene::init(...)`

位置：`src/renderer/renderer.cpp`

表达式：

```cpp
if (useStreaming) sceneStreaming.init(...);
else scenePreloaded.init(...);
```

作用：统一封装两种 GPU scene 驻留模式。renderer 不需要关心 geometry 是一次性 preload 还是按需 streaming，只通过 `RenderScene` 获取 shader buffer 和统计信息。

### 34. `Renderer::initBasics(...)`

位置：`src/renderer/renderer.cpp`

表达式：

```cpp
create render instance buffer;
upload instance matrices/materials/geometry mapping;
create optional sorting buffers;
```

作用：创建 renderer 共享基础资源。不同 renderer 变体都需要 instance 数据、基础 descriptor、debug bbox 和 fullscreen pass。

### 35. `RendererRasterClustersLod::render(...)`

位置：`src/renderer/renderer_clusters_lod.cpp`

表达式：

```text
upload FrameConstants/SceneBuilding
clear traversal/readback buffers
streaming pre-traversal
traversal init/run/build setup
optional SW raster
mesh shader draw
optional Hi-Z build
streaming post/end
```

作用：Cluster LOD 渲染器的核心命令录制函数。它串起 GPU 遍历、间接绘制准备、硬件 mesh shader 渲染、可选软件光栅、Hi-Z 和 streaming 状态更新。

## 四、Streaming 函数

### 36. `SceneStreaming::init(...)`

位置：`src/streaming/streaming.cpp`

表达式：

```cpp
create streaming buffers;
init requests/resident/allocator/updates/storage;
initGeometries(...);
initShadersAndPipelines();
```

作用：初始化 streaming 模式下的 GPU scene。它建立 request、resident、allocator、storage、update 等子系统，使 GPU traversal 能按需请求 group。

### 37. `SceneStreaming::cmdBeginFrame(...)`

位置：`src/streaming/streaming.cpp`

表达式：

```cpp
consume completed tasks;
upload finished group data;
prepare shader request/update state;
```

作用：每帧 streaming 前置处理。它接收前几帧异步任务的完成结果，并把 CPU 处理后的加载数据同步给 GPU。

### 38. `SceneStreaming::handleCompletedRequest(...)`

位置：`src/streaming/streaming.cpp`

表达式：

```cpp
read GPU load/unload requests;
for each load:
  allocate resident ID/storage
  decompress group
  append transfer
for each unload:
  release resident/storage
```

作用：处理 GPU traversal 产生的 streaming 请求。它是 GPU 可见性决策和 CPU/GPU 数据驻留状态之间的桥梁。

### 39. `SceneStreaming::cmdPreTraversal(...)`

位置：`src/streaming/streaming.cpp`

表达式：

```cpp
dispatch stream_setup.comp;
update scene geometry addresses before traversal;
```

作用：在 traversal 前运行 streaming setup，使 shader 能看到本帧最新的 geometry/group 地址、resident 状态和 pending update。

### 40. `SceneStreaming::cmdPostTraversal(...)`

位置：`src/streaming/streaming.cpp`

表达式：

```cpp
compact load/unload requests;
optional age filter;
copy request data for CPU readback;
```

作用：在 traversal 后整理 GPU 生成的请求，包括按年龄过滤、压缩请求列表和准备 CPU readback。

### 41. `StreamingResident::canAllocateGroup(...)`

位置：`src/streaming/streamutils.cpp`

表达式：

```cpp
return groupAllocator.has(1) && clusterAllocator.has(numClusters);
```

作用：判断 resident 表是否还能容纳一个 group 及其 cluster 地址。它防止 GPU resident ID 或 cluster address 表溢出。

### 42. `StreamingResident::addGroup(...)`

位置：`src/streaming/streamutils.cpp`

表达式：

```cpp
groupResidentID = allocate group ID;
clusterResidentID = allocate cluster range;
map[geometryGroup.key] = groupResidentID;
```

作用：把一个 geometry/group 标记为 GPU resident。它维护 geometry group 到 resident ID 的稳定映射，并分配 cluster 地址范围。

### 43. `StreamingResident::removeGroup(...)`

位置：`src/streaming/streamutils.cpp`

表达式：

```cpp
erase geometryGroup -> residentID;
free groupResidentID;
free clusterResidentID range;
```

作用：卸载 group 时回收 resident 表项和 cluster 地址范围，同时维护 active group 列表。

### 44. `StreamingStorage::allocate(...)`

位置：`src/streaming/streamutils.cpp`

表达式：

```cpp
handle = m_dataAllocator.subAllocate(sz);
deviceAddress = baseAddress + handle.offset;
```

作用：从 streaming geometry buffer 中分配一段 GPU 可访问空间，用于存放解压后的 group 数据。

### 45. `StreamingStorage::appendTransfer(...)`

位置：`src/streaming/streamutils.cpp`

表达式：

```cpp
dst region -> VkBufferCopy;
return mapped staging pointer;
```

作用：把一次 group 上传追加到 staging transfer 队列中。调用方拿到 host 指针后写入解压数据，之后由 `cmdUploadTask` 统一录制 copy。

### 46. `StreamingTaskQueue::acquireTaskIndex()` / `releaseTaskIndex(...)`

位置：`src/streaming/streamutils.hpp`

表达式：

```cpp
acquire: find first bit in availableTaskBits, clear it
release: availableTaskBits |= 1 << index
```

作用：管理最多 `STREAMING_MAX_ACTIVE_TASKS` 个异步任务槽。它避免 request、upload、update 任务复用同一个未完成 slot。

## 五、Shader 侧关键函数

### 47. `computeUniformScale(...)`

位置：`shaders/traversal/traversal.glsl`

表达式：

```glsl
scale = max(length(transform[0]), length(transform[1]), length(transform[2]));
```

作用：估计实例变换的最大缩放。LOD 误差和包围球半径都要乘以该值，避免非单位缩放导致 LOD 判断错误。

### 48. `testForTraversal(...)`

位置：`shaders/traversal/traversal.glsl`

表达式：

```glsl
errorDistance = max(view.nearPlane, sphereDistance - radius * uniformScale);
errorOverDistance = maxQuadricError * uniformScale / errorDistance;
return errorOverDistance >= threshold * errorScale;
```

作用：判断当前 LOD 节点是否需要继续细分。误差越大、距离越近、缩放越大，越容易继续 traversal。

### 49. `computeLodTransitionFactor(...)`

位置：`shaders/traversal/traversal.glsl`

表达式：

```glsl
transitionStart = threshold * 0.8;
transitionEnd = threshold * 1.2;
return smoothstep(transitionStart, transitionEnd, (currentError + nextError) * 0.5);
```

作用：计算 LOD 过渡因子，减少层级切换时的突变。它以阈值附近的误差区间作为平滑窗口。

### 50. `intersectFrustum(...)`

位置：`shaders/common/culling_frustum.inc`

表达式：

```glsl
for 8 bbox corners:
  hPos = viewProj * world * corner
  bits &= getCullBits(hPos)
return bits == 0
```

作用：判断 AABB 是否与视锥相交，并输出裁剪空间包围矩形。traversal、group culling 和 raster routing 都依赖这个结果。

### 51. `getCullBits(...)`

位置：`shaders/common/culling_frustum.inc`

表达式：

```glsl
bits |= x < -w ? 1 : 0;
bits |= x >  w ? 2 : 0;
bits |= y < -w ? 4 : 0;
bits |= y >  w ? 8 : 0;
bits |= z <  0 ? 16 : 0;
bits |= z >  w ? 32 : 0;
```

作用：为齐次裁剪空间点生成平面外侧 bit mask。8 个角点的 mask 做 AND 后，如果仍有某一位为 1，说明整个 bbox 在同一裁剪平面外。

### 52. `intersectHiz(...)`

位置：`shaders/common/culling_hiz.inc`

表达式：

```glsl
uvRect = clipRect * 0.5 + 0.5;
mip = fastMipmapLevel(max(rectSize) * hizSizeMax);
depth = textureLod(texHizFar, centerUV, mip).r;
visible = clipMax.z >= depth - c_depthNudge;
```

作用：用 Hi-Z 深度金字塔进行遮挡剔除。它根据 bbox 屏幕尺寸选择 mip level，再比较 bbox 最远深度和该区域历史深度。

### 53. `testTriangleSW(...)`

位置：`shaders/common/culling_raster.inc`

表达式：

```glsl
cross = (b.xy - a.xy).y * (c.xy - a.xy).x
      - (b.xy - a.xy).x * (c.xy - a.xy).y;
if (!twoSided && cross < 0) reject;
return testTrianglePixel(...);
```

作用：软件光栅路径的三角形有效性测试。它做背面剔除、像素包围盒检查和退化三角形过滤。

### 54. `testTriangleHW(...)`

位置：`shaders/common/culling_raster.inc`

表达式：

```glsl
if (!isFrontFacingHW(ha, hb, hc)) reject;
project vertices to pixels;
return testTrianglePixel(...);
```

作用：硬件 raster 路径的三角形预剔除逻辑。它和 SW 路径共用像素包围盒判断，保证 routing 判断更一致。

### 55. `vec_to_oct(...)` / `oct_to_vec(...)`

位置：`shaders/common/attribute_encoding.h`

表达式：

```glsl
vec_to_oct:
p = v.xy / (abs(v.x) + abs(v.y) + abs(v.z));
if v.z <= 0: fold lower hemisphere

oct_to_vec:
v = vec3(e.x, e.y, 1 - abs(e.x) - abs(e.y));
if v.z < 0: unfold;
normalize(v)
```

作用：在 3D 单位向量和 2D 八面体编码之间转换。法线压缩、解压都基于这两个函数。

### 56. `normal_pack(...)` / `normal_unpack(...)`

位置：`shaders/common/attribute_encoding.h`

表达式：

```glsl
packed = quantize(vec_to_oct_precise(normal), halfBits) into uint32;
normal = oct_to_vec(dequantize(packed));
```

作用：把法线压缩进 32-bit 字段，并在 shader 中恢复。它降低 vertex attribute 存储和带宽成本。

### 57. `tangent_pack(...)` / `tangent_unpack(...)`

位置：`shaders/common/attribute_encoding.h`

表达式：

```glsl
basis = tangent_orthonormalBasis(normal);
angle = atan(dot(autoTangent, tangent), dot(autoBitangent, tangent)) / PI;
encoded = angleBits << 1 | signBit;
```

作用：用法线派生局部正交基，只保存切线相对该基的角度和 handedness。相比直接保存 vec4 tangent 更节省空间。

### 58. `shading(...)`

位置：`shaders/common/render_shading.glsl`

表达式：

```glsl
ambient = ao * albedo * mix(groundColor, skyColor, up_dot);
flashlight = albedo * (diffuse + specular);
overhead = sunColor * sunIntensity * albedo * (diffuse + specular);
color = ambient + flashlight + overhead;
```

作用：项目默认片元着色函数。它把可视化颜色、环境光、手电光、太阳光和 AO 合成最终颜色。

### 59. `visualizeColor(...)`

位置：`shaders/common/render_shading.glsl`

表达式：

```glsl
if cluster/group/triangle: colorizeID(visData)
if LOD: lodMix(uintBitsToFloat(visData))
if material: decode material color
```

作用：根据 UI 中的 visualize 模式生成调试颜色。LOD、cluster、group、triangle、material 等视图都通过它统一着色。

### 60. `packPickingValue(...)`

位置：`shaders/common/render_shading.glsl`

表达式：

```glsl
z = 1 - clamp(z, 0, 1);
bits = floatBitsToUint(z);
value = (uint64_t(bits) << 32) | uint64_t(object_id);
```

作用：把 picking ID 和深度打包成 64-bit 值，用于 GPU 原子写入 picking/readback buffer。高 32 位保存可排序深度，低 32 位保存对象 ID。

## 结语

这个项目的核心不是单个函数，而是几组函数之间的契约：

1. `meshlod` 负责把输入 mesh 变成可分层选择的 cluster/group。
2. `Scene` 负责把构建结果整理为 shader 可消费的 runtime layout，并通过 cache/压缩降低加载成本。
3. `RenderScene` 和 `SceneStreaming` 把 CPU 数据转换为 GPU 驻留状态。
4. `RendererRasterClustersLod::render` 串起每帧 traversal、render、Hi-Z 和 streaming 更新。
5. Shader 侧通过误差公式、视锥剔除、Hi-Z 剔除和属性解码完成最终选择与绘制。

读源码时建议从 `LodClusters::onRender` 和 `RendererRasterClustersLod::render` 进入运行时路径，再回到 `Scene::init` 和 `clodBuild` 理解离线数据如何生成。这样能同时看到“数据如何构建”和“数据如何被 GPU 消费”。
