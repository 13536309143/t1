#include <volk.h> // 引入 volk 库，用于动态加载 Vulkan API 函数指针
#include <nvutils/alignment.hpp> // 引入 NVIDIA 工具库中的内存对齐工具
#include <fmt/format.h> // 引入 fmt 库，用于高效的字符串格式化
#include "renderer.hpp" // 引入渲染器基类定义
#include "../shaders/shaderio.h" // 引入与着色器 (Shader) 交互的数据结构定义
namespace lodclusters {
// 定义名为 RendererRasterClustersLod 的类，继承自 Renderer 基类
class RendererRasterClustersLod : public Renderer 
{
public:
  // 重写初始化函数，接收资源、渲染场景和渲染器配置参数
  virtual bool init(Resources& res, RenderScene& rscene, const RendererConfig& config) override;
  // 重写渲染核心函数，每帧被调用，负责提交渲染指令
  virtual void render(VkCommandBuffer primary, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler) override;
  // 重写帧缓冲更新函数，当窗口大小改变或帧缓冲重建时调用
  virtual void updatedFrameBuffer(Resources& res, RenderScene& rscene) override;
  // 重写清理函数，释放 Vulkan 资源
  virtual void deinit(Resources& res) override;
private:
  // 私有函数：专门用于初始化和编译各类着色器
  bool initShaders(Resources& res, RenderScene& scene, const RendererConfig& config);
  struct Shaders  // 内部结构体：保存所有编译后的着色器 SPIR-V 字节码结果
  {
    shaderc::SpvCompilationResult graphicsMesh;             // 网格着色器 (Mesh Shader)
    shaderc::SpvCompilationResult graphicsFragment;         // 片元着色器 (Fragment Shader)
    shaderc::SpvCompilationResult computeTraversalPresort;  // 计算着色器：遍历前的预排序
    shaderc::SpvCompilationResult computeTraversalInit;     // 计算着色器：遍历初始化
    shaderc::SpvCompilationResult computeTraversalRun;      // 计算着色器：执行 LOD 遍历和剔除
    shaderc::SpvCompilationResult computeTraversalGroups;   // 计算着色器：分离的组遍历 (可选)
    shaderc::SpvCompilationResult computeBuildSetup;        // 计算着色器：构建和设置间接绘制参数
    shaderc::SpvCompilationResult computeRaster;            // 计算着色器：软件光栅化
  };
  struct Pipelines// 内部结构体：保存所有 Vulkan 渲染和计算管线对象 (VkPipeline)
  {
    VkPipeline graphicsMesh            = nullptr; // 网格着色器图形管线
    VkPipeline graphicsBboxes          = nullptr; // 边界框渲染图形管线 (调试用)
    VkPipeline computeTraversalPresort = nullptr; // 预排序计算管线
    VkPipeline computeTraversalInit    = nullptr; // 初始化遍历计算管线
    VkPipeline computeTraversalRun     = nullptr; // 核心遍历计算管线
    VkPipeline computeTraversalGroups  = nullptr; // 分组遍历计算管线
    VkPipeline computeBuildSetup       = nullptr; // 绘制参数设置计算管线
    VkPipeline computeRaster           = nullptr; // 软件光栅化计算管线
  };
  Shaders            m_shaders;        // 实例化着色器结构体
  Pipelines          m_pipelines;      // 实例化管线结构体
  VkShaderStageFlags m_stageFlags{};   // 着色器阶段标志位掩码
  VkPipelineLayout   m_pipelineLayout{}; // 管线布局 (描述了 Shader 如何访问描述符和推送常量)
  nvvk::DescriptorPack m_dsetPack;     // NVIDIA 工具类：用于打包和管理描述符集 (Descriptor Sets)
  nvvk::Buffer m_sceneBuildBuffer;     // Vulkan 缓冲：存储构建场景所需的数据 (SSBO/UBO)
  nvvk::Buffer m_sceneTraversalBuffer; // Vulkan 缓冲：存储遍历过程中的节点数据
  nvvk::Buffer m_sceneDataBuffer;      // Vulkan 缓冲：存储具体的集群、实例、排序等庞大场景数据
  shaderio::SceneBuilding m_sceneBuildShaderio; // C++ 端映射的着色器结构体，准备传给 GPU
};
// ------------------- 着色器初始化实现 -------------------
bool RendererRasterClustersLod::initShaders(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
    // 检查配置冲突：如果开启软件光栅化，必须关闭光照、开启分组和剔除
  if(config.useComputeRaster && (config.useShading || !config.useSeparateGroups || !config.useCulling))
  {
    LOGW("Hybrid SW/HW raster requires:\n  visualize == visibility buffer or depth only\n  separate groups on\n  culling on\n\n");
    return false;// 配置不合法，退出
  }
  // 调用基类方法初始化基础着色器
  if(!initBasicShaders(res, rscene, config))
  {
    return false;
  }
  // 创建着色器编译选项对象
  shaderc::CompileOptions options = res.makeCompilerOptions();
  // 根据场景支持的最大顶点/三角形数，调整硬件 Meshlet (簇) 的属性
  uint32_t meshletTriangles = shaderio::adjustClusterProperty(rscene.scene->m_maxClusterTriangles);
  uint32_t meshletVertices  = shaderio::adjustClusterProperty(rscene.scene->m_maxClusterVertices);
  LOGI("mesh shader config: %d triangles %d vertices\n", meshletTriangles, meshletVertices);// 打印日志
  
  // 优化：减少着色器编译时间，只启用必要的宏定义
  options.SetOptimizationLevel(shaderc_optimization_level_performance);
  
  // 向 Shader 编译器注入预编译宏定义，根据当前系统属性和用户配置控制 Shader 内部分支
  options.AddMacroDefinition("SUBGROUP_SIZE", fmt::format("{}", res.m_physicalDeviceInfo.properties11.subgroupSize)); // 子组大小 (通常是 32 也就是 NVIDIA warp size)
  options.AddMacroDefinition("USE_16BIT_DISPATCH", fmt::format("{}", res.m_use16bitDispatch ? 1 : 0)); // 是否使用 16 位 Dispatch 指令
  options.AddMacroDefinition("CLUSTER_VERTEX_COUNT", fmt::format("{}", meshletVertices)); // 簇的最大顶点数
  options.AddMacroDefinition("CLUSTER_TRIANGLE_COUNT", fmt::format("{}", meshletTriangles)); // 簇的最大三角形数
  options.AddMacroDefinition("TARGETS_RASTERIZATION", "1"); // 标记目标是光栅化
  options.AddMacroDefinition("USE_STREAMING", rscene.useStreaming ? "1" : "0"); // 是否启用数据流式传输
  options.AddMacroDefinition("USE_SORTING", config.useSorting ? "1" : "0"); // 是否启用实例排序
  options.AddMacroDefinition("USE_CULLING", config.useCulling ? "1" : "0"); // 是否启用遮挡剔除
  
  // 启用细粒度图元剔除 (须在整体 Culling 开启的前提下)
  options.AddMacroDefinition("USE_PRIMITIVE_CULLING", config.useCulling && config.usePrimitiveCulling ? "1" : "0");
  // 启用双遍遮挡剔除 (Two-Pass Culling，第一遍用上一帧 Hi-Z，第二遍用本帧 Hi-Z)
  options.AddMacroDefinition("USE_TWO_PASS_CULLING", config.useCulling && config.useTwoPassCulling ? "1" : "0");
  options.AddMacroDefinition("USE_RENDER_STATS", config.useRenderStats ? "1" : "0"); // 是否统计渲染数据
  options.AddMacroDefinition("USE_SEPARATE_GROUPS", config.useSeparateGroups ? "1" : "0"); // 遍历时是否分离任务组
  options.AddMacroDefinition("USE_EXT_MESH_SHADER", fmt::format("{}", config.useEXTmeshShader ? 1 : 0)); // 使用标准的 EXT_mesh_shader 还是 NV 的
  options.AddMacroDefinition("MESHSHADER_WORKGROUP_SIZE", fmt::format("{}", m_meshShaderWorkgroupSize)); // Mesh Shader 工作组大小
  options.AddMacroDefinition("MESHSHADER_BBOX_COUNT", fmt::format("{}", m_meshShaderBoxes)); // 边界框数量
  
  // 注入顶点属性宏 (法线、切线、UV等)
  options.AddMacroDefinition("ALLOW_VERTEX_NORMALS", rscene.scene->m_hasVertexNormals && res.m_supportsBarycentrics ? "1" : "0");
  options.AddMacroDefinition("ALLOW_VERTEX_TANGENTS", rscene.scene->m_hasVertexTangents && res.m_supportsBarycentrics ? "1" : "0");
  options.AddMacroDefinition("ALLOW_VERTEX_TEXCOORDS", rscene.scene->m_hasVertexTexCoord0 ? "1" : "0");
  
  // 是否允许着色计算 (如果使用软件光栅化，则关闭片元着色)
  options.AddMacroDefinition("ALLOW_SHADING", config.useShading && !config.useComputeRaster ? "1" : "0");
  // 添加深度只渲染模式 (如用于预渲染深度或阴影贴图)
  options.AddMacroDefinition("USE_DEPTH_ONLY", !config.useShading && config.useDepthOnly ? "1" : "0");
  options.AddMacroDefinition("DEBUG_VISUALIZATION", config.useDebugVisualization && res.m_supportsBarycentrics ? "1" : "0"); // 调试可视化
  options.AddMacroDefinition("USE_SW_RASTER", config.useComputeRaster ? "1" : "0"); // 使用 Compute Shader 进行软件光栅化
  options.AddMacroDefinition("USE_ADAPTIVE_SW_RASTER_ROUTING", config.useComputeRaster && config.useAdaptiveRasterRouting ? "1" : "0");
  options.AddMacroDefinition("USE_TWO_SIDED", rscene.scene->m_hasTwoSided && !config.forceTwoSided ? "1" : "0"); // 材质是否有双面
  options.AddMacroDefinition("USE_FORCED_TWO_SIDED", config.forceTwoSided ? "1" : "0"); // 强制双面材质处理
  options.AddMacroDefinition("USE_FORCED_INVISIBLE_CULLING", "0"); // 强制隐藏面剔除 (此处写死为 0)
  
    // 开始编译所有的着色器文件，并传入上方的 options (预定义宏)
    res.compileShader(m_shaders.graphicsMesh, VK_SHADER_STAGE_MESH_BIT_NV, "clusters.mesh.glsl", &options); // 编译 Mesh Shader
    res.compileShader(m_shaders.graphicsFragment, VK_SHADER_STAGE_FRAGMENT_BIT, "frag.glsl", &options); // 编译 Fragment Shader
    res.compileShader(m_shaders.computeTraversalPresort, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_presort.comp.glsl", &options); // 预排序 Compute
    res.compileShader(m_shaders.computeTraversalInit, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_init.comp.glsl", &options); // 初始化 Compute
    res.compileShader(m_shaders.computeTraversalRun, VK_SHADER_STAGE_COMPUTE_BIT, "traversal_run.comp.glsl", &options); // LOD 遍历 Compute
    res.compileShader(m_shaders.computeBuildSetup, VK_SHADER_STAGE_COMPUTE_BIT, "build_setup.comp.glsl", &options); // 间接绘制参数设置 Compute
    // 如果开启了软光栅，单独编译相应的软光栅 Compute Shader
  if(config.useComputeRaster)
  {
    res.compileShader(m_shaders.computeRaster, VK_SHADER_STAGE_COMPUTE_BIT, "SWclusters.comp.glsl", &options);
  }
  // 如果开启了分离遍历任务组，编译相关的 Compute Shader
  if(config.useSeparateGroups)
  {
    res.compileShader(m_shaders.computeTraversalGroups, VK_SHADER_STAGE_COMPUTE_BIT,"traversal_run_separate_groups.comp.glsl", &options);
  }
  // 验证所有 Shader 是否都编译成功
  return res.verifyShaders(m_shaders);
}
// ------------------- 管线和资源初始化实现 -------------------
bool RendererRasterClustersLod::init(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  m_resourceReservedUsage = {}; // 初始化显存用量统计
  m_config                = config; // 保存配置
  m_maxRenderClusters     = 1u << config.numRenderClusterBits; // 通过位移计算最大可渲染的 Cluster 数量
  m_maxTraversalTasks     = 1u << config.numTraversalTaskBits; // 计算最大遍历任务并发数
  // 步骤 1：编译着色器
  if(!initShaders(res, rscene, config))
  {
    return false;
  }
  
  // 优化：预分配管线相关资源，减少动态扩容
  m_pipelines = {}; // 确保管线对象初始化为nullptr
  m_dsetPack.deinit(); // 确保描述符包已清理
  // 步骤 2：初始化基类的通用资源
  initBasics(res, rscene, config);
  // 统计几何和操作内存占用
  m_resourceReservedUsage.geometryMemBytes   = rscene.getGeometrySize(true);
  m_resourceReservedUsage.operationsMemBytes = logMemoryUsage(rscene.getOperationsSize(), "operations", "rscene total");
  {
      // 创建 m_sceneBuildBuffer：既作为存储缓冲(SSBO)，也作为 UBO 和 间接绘制命令缓冲
    res.createBuffer(m_sceneBuildBuffer, sizeof(shaderio::SceneBuilding), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    NVVK_DBG_NAME(m_sceneBuildBuffer.buffer);// 给缓冲命名，方便调试(RenderDoc)
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneBuildBuffer.bufferSize, "operations", "build shaderio");
    // 初始化 C++ 端的 m_sceneBuildShaderio 数据结构
    memset(&m_sceneBuildShaderio, 0, sizeof(m_sceneBuildShaderio));
    m_sceneBuildShaderio.numRenderInstances = uint32_t(m_renderInstances.size()); // 场景实例数量
    m_sceneBuildShaderio.maxRenderClusters = uint32_t(1u << config.numRenderClusterBits); // 集群上限
    m_sceneBuildShaderio.maxTraversalInfos = uint32_t(1u << config.numTraversalTaskBits); // 遍历任务上限
    // 初始化间接分发/绘制的默认组大小参数 (通常由 GPU Compute Shader 在运行时修改这些值)
    m_sceneBuildShaderio.indirectDispatchGroups.gridY  = 1;
    m_sceneBuildShaderio.indirectDispatchGroups.gridZ  = 1;
    m_sceneBuildShaderio.indirectDrawClustersEXT.gridZ = 1;
    m_sceneBuildShaderio.indirectDrawClustersNV.first  = 0;
    m_sceneBuildShaderio.indirectDrawClustersSW.gridY  = 1;
    m_sceneBuildShaderio.indirectDrawClustersSW.gridZ  = 1;
    // 分配大块数据缓冲区所需的偏移量
    BufferRanges mem = {};
    // 将待渲染的集群信息空间追加到管理器，8字节对齐
    m_sceneBuildShaderio.renderClusterInfos = mem.append(sizeof(shaderio::ClusterInfo) * m_sceneBuildShaderio.maxRenderClusters, 8);
    // 如果启用了软光栅，还需要单独开辟软光栅所需的集群信息内存
    if(config.useComputeRaster)
    {
      m_sceneBuildShaderio.renderClusterInfosSW = mem.append(sizeof(shaderio::ClusterInfo) * m_sceneBuildShaderio.maxRenderClusters, 8);
    }
    // 排序使用的 Keys 和 Values
    if(config.useSorting)
    {
      m_sceneBuildShaderio.instanceSortKeys   = mem.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
      m_sceneBuildShaderio.instanceSortValues = mem.append(sizeof(uint32_t) * m_renderInstances.size(), 4);
    }
    // 分离组遍历信息的内存
    if(config.useSeparateGroups)
    {
      m_sceneBuildShaderio.traversalGroupInfos = mem.append(sizeof(uint64_t) * m_sceneBuildShaderio.maxTraversalInfos, 8);
    }
    // 两遍剔除时，需要一个字节数组记录每个 Instance 在第一遍的可见性 (InstanceVisibility)
    if(config.useTwoPassCulling && config.useCulling)
    {
      m_sceneBuildShaderio.instanceVisibility = mem.append(sizeof(uint8_t) * m_renderInstances.size(), 4);
    }
    // 创建 m_sceneDataBuffer，大小为上述所有 mem 追加后累积的总大小
    res.createBuffer(m_sceneDataBuffer, mem.getSize(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    NVVK_DBG_NAME(m_sceneDataBuffer.buffer);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneDataBuffer.bufferSize, "operations", "build data");
    // 将上面计算的偏移量转换为实际的 GPU 显存地址 (Device Address)
    m_sceneBuildShaderio.renderClusterInfos += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceSortKeys += m_sceneDataBuffer.address;
    m_sceneBuildShaderio.instanceSortValues += m_sceneDataBuffer.address;
    if(config.useSeparateGroups)
    {
      m_sceneBuildShaderio.traversalGroupInfos += m_sceneDataBuffer.address;
    }
    if(config.useComputeRaster)
    {
      m_sceneBuildShaderio.renderClusterInfosSW += m_sceneDataBuffer.address;
    }
    // 管理 instanceVisibility 的 GPU 地址
    if(config.useTwoPassCulling && config.useCulling)
    {
      m_sceneBuildShaderio.instanceVisibility += m_sceneDataBuffer.address;
    }
    // 创建单独的缓冲区 m_sceneTraversalBuffer，存放正在遍历的层级 LOD 节点信息
    res.createBuffer(m_sceneTraversalBuffer, sizeof(uint64_t) * m_sceneBuildShaderio.maxTraversalInfos, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    NVVK_DBG_NAME(m_sceneTraversalBuffer.buffer);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sceneTraversalBuffer.bufferSize, "operations", "build traversal");
    m_sceneBuildShaderio.traversalNodeInfos = m_sceneTraversalBuffer.address;
  }
  // 更新基础的 Descriptor Sets
  updateBasicDescriptors(res, rscene, &m_sceneBuildBuffer);
  // 如果启用了流式加载，更新绑定的流数据指针
  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.updateBindings(m_sceneBuildBuffer);
  }
  {
      // 定义该管线需要使用的 Shader 阶段集合
    m_stageFlags = VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    // 定义 Descriptor Bindings (告诉着色器哪些槽位对应哪些缓冲)
    nvvk::DescriptorBindings bindings;
    bindings.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags); // 帧常量
    bindings.addBinding(BINDINGS_READBACK_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags); // 回读数据
    bindings.addBinding(BINDINGS_GEOMETRIES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags); // 几何数据
    bindings.addBinding(BINDINGS_RENDERINSTANCES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags); // 实例数据
    bindings.addBinding(BINDINGS_SCENEBUILDING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags); // 场景构建读写
    bindings.addBinding(BINDINGS_SCENEBUILDING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);  // 场景构建只读
    // HiZ (层级 Z 缓冲) 纹理绑定数量根据是否开启“两遍模式”动态调整
        // HiZ[0]：当前帧构建的 HiZ (或者是用于第二遍剔除的 HiZ)
        // HiZ[1]：上一帧保留的 HiZ (用于第一遍快速视锥和遮挡剔除)
    bindings.addBinding(BINDINGS_HIZ_TEX, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, config.useCulling && config.useTwoPassCulling ? 2 : 1, m_stageFlags);
    bindings.addBinding(BINDINGS_RASTER_ATOMIC, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, m_stageFlags); // 软光栅输出用的原子图像
    if(rscene.useStreaming)// 流式传输用的 Buffer
    {
      bindings.addBinding(BINDINGS_STREAMING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
      bindings.addBinding(BINDINGS_STREAMING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
    }
    // 初始化描述符集管理器
    m_dsetPack.init(bindings, res.m_device);
    // 创建 Vulkan Pipeline Layout (包含所有描述符和 Push Constants 的布局)
    nvvk::createPipelineLayout(res.m_device, &m_pipelineLayout, {m_dsetPack.getLayout()}, {{m_stageFlags, 0, sizeof(uint32_t)}});
    // 填充 Descriptor Sets 的实际内容指针
    nvvk::WriteSetContainer writeSets;
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_FRAME_UBO), res.m_commonBuffers.frameConstants);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_READBACK_SSBO), &res.m_commonBuffers.readBack);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_GEOMETRIES_SSBO), rscene.getShaderGeometriesBuffer());
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_RENDERINSTANCES_SSBO), m_renderInstanceBuffer);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_SCENEBUILDING_SSBO), m_sceneBuildBuffer);
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_SCENEBUILDING_UBO), m_sceneBuildBuffer);
    //    writeSets.append(m_dsetPack.makeWrite(BINDINGS_HIZ_TEX), &res.m_hizUpdate.farImageInfo);
    //绑定两个 HiZ 纹理绑定两个 HiZ 纹理
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_HIZ_TEX, 0, 0), &res.m_hizUpdate[0].farImageInfo);
    if(config.useCulling && config.useTwoPassCulling)
    {
      writeSets.append(m_dsetPack.makeWrite(BINDINGS_HIZ_TEX, 0, 1), &res.m_hizUpdate[1].farImageInfo);
    }
    writeSets.append(m_dsetPack.makeWrite(BINDINGS_RASTER_ATOMIC), &res.m_frameBuffer.imgRasterAtomic);
    if(rscene.useStreaming)
    {
      writeSets.append(m_dsetPack.makeWrite(BINDINGS_STREAMING_SSBO), rscene.sceneStreaming.getShaderStreamingBuffer());
      writeSets.append(m_dsetPack.makeWrite(BINDINGS_STREAMING_UBO), rscene.sceneStreaming.getShaderStreamingBuffer());
    }
    // 批量提交描述符更新至 Vulkan 驱动
    vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);
  }
  // ---------------- 图形管线创建 ----------------
  {
    nvvk::GraphicsPipelineCreator graphicsGen;// NVVK 辅助类创建管线
    nvvk::GraphicsPipelineState   state = res.m_basicGraphicsState; // 继承通用状态
    graphicsGen.pipelineInfo.layout                  = m_pipelineLayout;
    // 配置动态渲染的颜色/深度/模板附件格式
    graphicsGen.renderingState.depthAttachmentFormat = res.m_frameBuffer.pipelineRenderingInfo.depthAttachmentFormat;
    graphicsGen.renderingState.stencilAttachmentFormat = res.m_frameBuffer.pipelineRenderingInfo.stencilAttachmentFormat;
    graphicsGen.colorFormats = {res.m_frameBuffer.colorFormat};
    state.rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; // 逆时针为正面
    if (config.forceTwoSided)
    {
        state.rasterizationState.cullMode = VK_CULL_MODE_NONE; // 如果强制双面，关闭背面剔除
    }
    if (config.useComputeRaster && false) // 这里有一段死代码 (&& false) 禁用了条件块
    {
        state.depthStencilState.depthWriteEnable = VK_FALSE;
        state.depthStencilState.depthTestEnable = VK_FALSE;
        state.depthStencilState.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    }
    // 绑定并创建 Mesh Shader 图形管线
    graphicsGen.addShader(VK_SHADER_STAGE_MESH_BIT_NV, "main", nvvkglsl::GlslCompiler::getSpirvData(m_shaders.graphicsMesh));
    // 深度只(Depth-Only)模式下不编译/绑定片段着色器 (Fragment Shader)
    //    graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", nvvkglsl::GlslCompiler::getSpirvData(m_shaders.graphicsFragment));
    if(!m_config.useDepthOnly)
    {
      graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", nvvkglsl::GlslCompiler::getSpirvData(m_shaders.graphicsFragment));
    }
    // 生成最终的 VkPipeline 对象给 m_pipelines.graphicsMesh
    graphicsGen.createGraphicsPipeline(res.m_device, nullptr, state, &m_pipelines.graphicsMesh);
  }
  // ---------------- 计算管线创建 ----------------
  {
    VkComputePipelineCreateInfo compInfo   = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkShaderModuleCreateInfo    shaderInfo = {};
    compInfo.stage                         = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    compInfo.stage.stage                   = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.pName                   = "main";// 入口函数名都是 main
    compInfo.stage.pNext                   = &shaderInfo;
    compInfo.layout                        = m_pipelineLayout;
    // 创建预排序计算管线
    if(config.useSorting)
    {
      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalPresort);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalPresort);
    }
    // 创建构建准备管线 
    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeBuildSetup);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBuildSetup);
    // 创建遍历初始化管线
    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalInit);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalInit);
    // 创建遍历执行管线
    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalRun);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalRun);
    // 创建软件光栅化计算管线
    if(config.useComputeRaster)
    {
      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeRaster);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeRaster);
    }
    // 创建分组遍历管线
    if(config.useSeparateGroups)
    {
      shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeTraversalGroups);
      vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeTraversalGroups);
    }
  }

  return true; // 初始化成功
}
// 辅助函数：根据线程总数和工作组大小，计算 Compute Shader 所需的 dispatch 次数
static uint32_t getWorkGroupCount(uint32_t numThreads, uint32_t workGroupSize)
{
  return (numThreads + workGroupSize - 1) / workGroupSize;
}
// ------------------- 核心渲染函数 (每帧调用) -------------------
void RendererRasterClustersLod::render(VkCommandBuffer cmd, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler)
{
  VkMemoryBarrier memBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER }; // 准备一个内存屏障结构体

  {
    // 获取屏幕缩放比例并计算当前相机的屏幕误差阈值
    glm::vec2 renderScale = res.getFramebufferWindow2RenderScale();
    float     pixelScale  = std::min(renderScale.x, renderScale.y);
    // clusterLodErrorOverDistance 这一段的作用是计算出一个几何误差度量值，
    // 因为在同一帧中，对于同一个摄像机而言，高度/视角相关参数是一个常数，这里预先算好传入 Shader 以节省 GPU 算力。
    m_sceneBuildShaderio.errorOverDistanceThreshold = clusterLodErrorOverDistance(frame.lodPixelError * pixelScale, frame.frameConstants.fov, frame.frameConstants.viewportf.y);////////////////////////////////////////////////////////////////////////////////////////////////因为在同一帧中，对于同一个摄像机而言，$\frac{H}{2 \times \tan(\frac{FOV}{2})}$ 完完全全是一个常数。
  }
  // 更新用于遍历和剔除的矩阵 (通常比摄像机真实矩阵大一点，作防穿帮处理或延迟剔除)
  m_sceneBuildShaderio.traversalViewMatrix    = frame.traversalViewMatrix;
  m_sceneBuildShaderio.cullViewProjMatrix     = frame.cullViewProjMatrix;
  m_sceneBuildShaderio.cullViewProjMatrixLast = frame.cullViewProjMatrixLast; // 上一帧的视锥矩阵，用于两遍剔除第一遍
  m_sceneBuildShaderio.frameIndex = m_frameIndex; // 帧号递增
  m_sceneBuildShaderio.swRasterThreshold = frame.swRasterThresholdEffective; // 切换到软光栅的屏幕尺寸阈值
  m_sceneBuildShaderio.swRasterTriangleDensityThreshold = frame.swRasterTriangleDensityThresholdEffective;
  // 将帧常量和 SceneBuilding 信息通过 vkCmdUpdateBuffer 拷贝到 GPU UBO/SSBO
  vkCmdUpdateBuffer(cmd, res.m_commonBuffers.frameConstants.buffer, 0, sizeof(shaderio::FrameConstants), (const uint32_t*)&frame.frameConstants);
  vkCmdUpdateBuffer(cmd, m_sceneBuildBuffer.buffer, 0, sizeof(shaderio::SceneBuilding), (const uint32_t*)&m_sceneBuildShaderio);
  // 清零回读用的缓冲
  vkCmdFillBuffer(cmd, res.m_commonBuffers.readBack.buffer, 0, sizeof(shaderio::Readback), 0);
  // 用 ~0 (即全 1 或 uint64 最大值) 填充遍历任务队列，代表初始空状态
  vkCmdFillBuffer(cmd, m_sceneTraversalBuffer.buffer, 0, m_sceneTraversalBuffer.bufferSize, ~0);
  // 如果启用软光栅，清理其输出目标原子纹理
  if(m_config.useComputeRaster)
  {
    VkClearColorValue clearValue;// 黑底/透明 清除颜色
    clearValue.uint32[0] = 0u;
    clearValue.uint32[1] = 0u;
    clearValue.uint32[2] = 0u;
    clearValue.uint32[3] = 0u;

    VkImageSubresourceRange subResource = {};
    subResource.aspectMask              = VK_IMAGE_ASPECT_COLOR_BIT;
    subResource.levelCount              = 1;
    subResource.layerCount              = 1;

    vkCmdClearColorImage(cmd, res.m_frameBuffer.imgRasterAtomic.image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &subResource);
  }
  // 场景流式加载：开启新帧，处理磁盘->显存的数据交换
  if(rscene.useStreaming)
  {
    SceneStreaming::FrameSettings settings;
    settings.ageThreshold = frame.streamingAgeThreshold; // 根据年龄卸载不常用 LOD 数据
    rscene.sceneStreaming.cmdBeginFrame(cmd, res.m_queueStates.primary, res.m_queueStates.transfer, settings, profiler);
  }
  // 插入内存屏障，确保所有 vkCmdUpdateBuffer / fillBuffer 的数据对 Shader 来说是完全准备好且可见的
  // 优化：使用更精确的内存屏障，减少不必要的等待
  memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 1, &memBarrier, 0, nullptr, 0, nullptr);
  // 流式加载遍历前置处理
  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.cmdPreTraversal(cmd, 0, profiler);
    // 屏障保护 Compute Shader 处理流式数据的写入
    // 优化：使用更精确的内存屏障，减少不必要的等待
    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                         VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0, nullptr, 0, nullptr);
  }
  // 核心循环：如果开启双遍剔除，则循环2次 (pass 0 和 pass 1)，否则执行 1 次。
  const uint32_t passCount = m_config.useCulling && m_config.useTwoPassCulling ? 2 : 1;
  const uint32_t lastPass  = passCount - 1;
  for(uint32_t pass = 0; pass < passCount; pass++)
  {
    {
      auto timerSection = profiler.cmdFrameSection(cmd, "Traversal Preparation");
      // 绑定通用的描述符集 (供所有的 Compute Pipelines 使用)
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);
      // 第一遍 并且 开启了排序功能 (深度前到后排序有利于 Early-Z 和剔除)
      if(pass == 0 && m_config.useSorting)
      {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalPresort);
        res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, TRAVERSAL_PRESORT_WORKGROUP));
        // 屏障：等待预排序着色器写入完毕
        // 优化：使用更精确的内存屏障，减少不必要的等待
        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                             VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0, nullptr, 0, nullptr);
        // 调用 vrdx 基数排序算法，根据距离将 Instances 进行排序
        vrdxCmdSortKeyValue(cmd, res.m_vrdxSorter, m_sceneBuildShaderio.numRenderInstances, m_sceneDataBuffer.buffer,
                            m_sceneBuildShaderio.instanceSortKeys - m_sceneDataBuffer.address, m_sceneDataBuffer.buffer,
                            m_sceneBuildShaderio.instanceSortValues - m_sceneDataBuffer.address,
                            m_sortingAuxBuffer.buffer, 0, nullptr, 0);
        // 屏障：等待硬件基数排序完成
        // 优化：使用更精确的内存屏障，减少不必要的等待
        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                             VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0, nullptr, 0, nullptr);
      }
      // ==== 遍历初始化 (将所有可见 Instance 的根节点压入待处理队列) ====
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalInit);
      res.cmdLinearDispatch(cmd, getWorkGroupCount(m_sceneBuildShaderio.numRenderInstances, TRAVERSAL_INIT_WORKGROUP));
      // 屏障：等待 Init 初始化写入完成
      // 优化：使用更精确的内存屏障，减少不必要的等待
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0, nullptr, 0, nullptr);
      // Compute Build Setup：在真正遍历之前，根据 Init 统计出的任务量，设置接下来计算着色器的间接 Dispatch 参数
      uint32_t buildSetupID = BUILD_SETUP_TRAVERSAL_RUN;
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
      vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID); // Push Constants 传递类型标志
      vkCmdDispatch(cmd, 1, 1, 1); // 仅分发 1 个线程去修正 Dispatch 缓冲参数
      // 屏障：等待 BuildSetup 修改 Indirect Argument 完成
      // 优化：使用更精确的内存屏障，减少不必要的等待
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0, nullptr, 0, nullptr);
    }
    {
        auto timerSection = profiler.cmdFrameSection(cmd, "Traversal Run"); // 性能标记区间：执行遍历层级与剔除
        // ==== 核心：执行 LOD 树遍历 ====
        // 开启持久化线程模式 (Persistent Threads)，工作组自我消费队列任务直到为空
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalRun);
        res.cmdLinearDispatch(cmd, getWorkGroupCount(frame.traversalPersistentThreads, TRAVERSAL_RUN_WORKGROUP));
        // 分开处理分组的遍历结果
      if(m_config.useSeparateGroups)
      {
        memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeTraversalGroups);
        vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDispatchGroups));
      }
      // 屏障：等待遍历结果和可见 Clusters 列表收集完毕
      // 优化：使用更精确的内存屏障，减少不必要的等待
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0, nullptr, 0, nullptr);
      // 再次调用 Build Setup：把前面遍历收集到的待渲染 Cluster 数量，转换为 Mesh Shader 绘制指令所需参数
      uint32_t buildSetupID = BUILD_SETUP_DRAW;
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBuildSetup);
      vkCmdPushConstants(cmd, m_pipelineLayout, m_stageFlags, 0, sizeof(uint32_t), &buildSetupID);
      vkCmdDispatch(cmd, 1, 1, 1);
      // 超大屏障：等待即将发生的间接绘制命令、传给 Mesh Shader / Compute Raster 的数据一切就绪
      // 优化：使用更精确的内存屏障，减少不必要的等待
      memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                           VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 
                           VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0, nullptr, 0, nullptr);
    }
    // 后续的流数据处理
    if(pass == lastPass && rscene.useStreaming)
    {
      rscene.sceneStreaming.cmdPostTraversal(cmd, 0, true, profiler);// 最后一遍完成，处理缺失的 LOD 页面请求
    }
    else if(rscene.useStreaming)
    {
      auto timerSection = profiler.cmdFrameSection(cmd, "Stream Post Traversal");
    }
    {
       auto timerSection = profiler.cmdFrameSection(cmd, "Draw"); // 性能标记区间：实际绘制
       // ==== 可选：软件光栅化 (针对极小多边形的优化) ====
      if(m_config.useComputeRaster)
      {
        auto timerSection = profiler.cmdFrameSection(cmd, "SW-Raster");
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeRaster);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);
        // 使用间接派发，根据之前 Build Setup 计算的需要使用软光栅的 Cluster 数量执行 Compute Shader
        vkCmdDispatchIndirect(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDrawClustersSW));
      }
      // 配置 Dynamic Rendering 的 Attachment Load 操作
      // 如果是第二遍渲染 (pass == 1)，必须保留第一遍的颜色/深度 (LOAD_OP_LOAD)
      // 第一遍看需不需要清空
      VkAttachmentLoadOp op = pass == 1 ? VK_ATTACHMENT_LOAD_OP_LOAD : (m_config.useShading ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_CLEAR);
      res.cmdBeginRendering(cmd, false, op, op);// 开启渲染通道
      if(pass == 0 && m_config.useShading)// 绘制天空盒/背景 (只在第一遍执行)
      {
        writeBackgroundSky(cmd);
      }

      {
        auto timerSection = profiler.cmdFrameSection(cmd, "HW-Raster"); // 硬件 Mesh Shader 光栅化
        // 绑定图形描述符集和网格管线
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines.graphicsMesh);
        // 根据配置选择调用标准 EXT 还是 NVIDIA 专有 Mesh Shader 的间接绘制 API
        if(m_config.useEXTmeshShader)
        {
          vkCmdDrawMeshTasksIndirectEXT(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDrawClustersEXT), 1, 0);
        }
        else
        {
          vkCmdDrawMeshTasksIndirectNV(cmd, m_sceneBuildBuffer.buffer, offsetof(shaderio::SceneBuilding, indirectDrawClustersNV), 1, 0);
        }
      }
      // 如果有软光栅输出，将之前软光栅原子写入的贴图数据合并写回真实的渲染目标(Framebuffer)
      if(m_config.useComputeRaster)
      {
        VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        memBarrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;
        memBarrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 1, &memBarrier, 0, 0, 0, nullptr);
        writeAtomicRaster(cmd);// 全屏后处理合并
      }
      // 调试：绘制被渲染的 Cluster 边界框
      if(frame.showClusterBboxes)
      {
        renderClusterBboxes(cmd, m_sceneBuildBuffer);
      }
      // 调试：绘制大 Instance 边界框
      if(pass == lastPass && frame.showInstanceBboxes)
      {
        renderInstanceBboxes(cmd);
      }
      vkCmdEndRendering(cmd);// 结束渲染通道
    }
    // ===== HiZ (层级深度缓冲) 构建 =====
    // 为下一遍剔除或下一帧剔除生成 MipMap 深度图
    //对应代码：shaders/hiz.comp.glsl。它通过 Subgroup 指令高效地对 Depth Texture 进行 Max/Min 规约，生成多个级别的 Mipmap。
    if(!frame.freezeCulling)// 冻结剔除功能时不更新 HiZ，方便 Debug 观察被剔除了什么
    {
      if(m_config.useTwoPassCulling)
      {
        res.cmdBuildHiz(cmd, frame, profiler, pass ^ 1);// 交叉索引更新 (0变1，1变0)
      }
      else
      {
        res.cmdBuildHiz(cmd, frame, profiler, 0);// 单边剔除，更新下标 0
      }
    }
  }
  // 如果使用两通道剔除，在性能分析器中分离两部分的耗时
  if(passCount > 1)
  {
    profiler.getProfilerTimeline()->frameAccumulationSplit();
  }
  /*
    if(!frame.freezeCulling)
  {
    res.cmdBuildHiz(cmd, frame, profiler);
  }
  */
  // 结束帧内的流处理操作 (回收页等)
  if(rscene.useStreaming)
  {
    rscene.sceneStreaming.cmdEndFrame(cmd, res.m_queueStates.primary, profiler);
  }
  // 帧末尾重置并统计显存几何占用
  m_resourceReservedUsage.geometryMemBytes = rscene.getGeometrySize(true);
  m_resourceActualUsage                    = m_resourceReservedUsage;
  m_resourceActualUsage.geometryMemBytes   = rscene.getGeometrySize(false);
  m_frameIndex++;// 自增帧序号供下一帧判断遮挡过期时间使用
}
// ------------------- FrameBuffer 变更事件 -------------------
void RendererRasterClustersLod::updatedFrameBuffer(Resources& res, RenderScene& rscene)
{
  vkDeviceWaitIdle(res.m_device);// 等待 GPU 空闲，避免读写冲突
  //单遍更新代码
  //VkWriteDescriptorSet writes[2];
  //writes[0]            = m_dsetPack.makeWrite(BINDINGS_HIZ_TEX);
  //writes[0].pImageInfo = &res.m_hizUpdate.farImageInfo;
  //writes[1]            = m_dsetPack.makeWrite(BINDINGS_RASTER_ATOMIC);
  //writes[1].pImageInfo = &res.m_frameBuffer.imgRasterAtomic.descriptor;
  // 准备重新绑定 FrameBuffer 相关的 Image 描述符
  VkWriteDescriptorSet writes[3];
  writes[0] = m_dsetPack.makeWrite(BINDINGS_RASTER_ATOMIC); // 软光栅目标
  writes[0].pImageInfo = &res.m_frameBuffer.imgRasterAtomic.descriptor;
  writes[1] = m_dsetPack.makeWrite(BINDINGS_HIZ_TEX, 0, 0); // 绑定新的 HiZ[0]
  writes[1].pImageInfo = &res.m_hizUpdate[0].farImageInfo;
  // 如果启用了两遍剔除，也需要绑定新的 HiZ[1]
  if(m_config.useCulling && m_config.useTwoPassCulling)
  {
    writes[2]            = m_dsetPack.makeWrite(BINDINGS_HIZ_TEX, 0, 1);
    writes[2].pImageInfo = &res.m_hizUpdate[1].farImageInfo;
  }
  //vkUpdateDescriptorSets(res.m_device, 2, writes, 0, nullptr);
  // 更新描述符到 Vulkan
  vkUpdateDescriptorSets(res.m_device, m_config.useCulling && m_config.useTwoPassCulling ? 3 : 2, writes, 0, nullptr);
  Renderer::updatedFrameBuffer(res, rscene);// 调用基类进行基础更新
}
// ------------------- 资源释放 -------------------
void RendererRasterClustersLod::deinit(Resources& res)
{
  // 销毁所有的 Pipeline 对象
  res.destroyPipelines(m_pipelines);
  // 销毁 Pipeline Layout
  vkDestroyPipelineLayout(res.m_device, m_pipelineLayout, nullptr);
  // 销毁并反初始化 Descriptor Pack
  m_dsetPack.deinit();
  // 销毁我们创建的 Vulkan Buffers
  res.m_allocator.destroyBuffer(m_sceneDataBuffer);
  res.m_allocator.destroyBuffer(m_sceneBuildBuffer);
  res.m_allocator.destroyBuffer(m_sceneTraversalBuffer);
  deinitBasics(res);// 调用基类通用销毁逻辑
}
// C 风格工厂函数，用于创建和返回这个渲染器实例的独占指针
std::unique_ptr<Renderer> makeRendererRasterClustersLod()
{
  return std::make_unique<RendererRasterClustersLod>();
}
}
