//==============================================================================
// 文件：src/streaming/streaming.hpp
// 模块定位：SceneStreaming 高层接口声明，统一管理一个场景在 流式加载 模式下的 GPU 数据和每帧调度入口。
// 数据流：Renderer 在遍历前后调用 begin/pre/post/end；SceneStreaming 在这些阶段处理请求、上传、地址更新和统计。
// 方法说明：高层接口把按需驻留机制封装成 renderer 可插拔组件，使预加载和流式加载共享同一遍历/渲染逻辑。
// 正确性约束：每帧调用顺序必须固定；reset 要同时清空 resident 状态和 Geometry 地址；descriptor 更新必须覆盖 traversal 和 render 需要的 缓冲。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#pragma once


// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include "streamutils.hpp"


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace lodclusters {


// 类型：SceneStreaming。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class SceneStreaming
{
public:


  // 函数：init。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  bool init(Resources* res, const Scene* scene, const StreamingConfig& config);


  // 函数：deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinit();


  // 函数：reset。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void reset();


  // 函数：reloadShaders。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
  bool reloadShaders()
  {

    deinitShadersAndPipelines();
    return initShadersAndPipelines();
  }


  // 结构：FrameSettings。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct FrameSettings
  {
    uint32_t ageThreshold = 16;
  };
  void cmdBeginFrame(VkCommandBuffer         cmd,
                     QueueState&             cmdQueueState,
                     QueueState&             asyncQueueState,
                     const FrameSettings&    settings,
                     nvvk::ProfilerGpuTimer& profiler);


  // 函数：cmdPreTraversal。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
  void cmdPreTraversal(VkCommandBuffer cmd, VkDeviceAddress clasScratchBuffer, nvvk::ProfilerGpuTimer& profiler);


  // 函数：cmdPostTraversal。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
  void cmdPostTraversal(VkCommandBuffer cmd, VkDeviceAddress clasScratchBuffer, bool runAgeFilter, nvvk::ProfilerGpuTimer& profiler);


  // 函数：cmdEndFrame。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
  void cmdEndFrame(VkCommandBuffer cmd, QueueState& cmdQueueState, nvvk::ProfilerGpuTimer& profiler);


  // 函数：getStats。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void getStats(StreamingStats& stats) const;
  const nvvk::BufferTyped<shaderio::Geometry>& getShaderGeometriesBuffer() const { return m_shaderGeometriesBuffer; }
  const nvvk::Buffer&                          getShaderStreamingBuffer() const { return m_shaderBuffer; }
  const shaderio::SceneStreaming&              getShaderStreamingData() const { return m_shaderData; }
  const StreamingConfig&                       getStreamingConfig() const { return m_config; }


  // 函数：getGeometrySize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  size_t getGeometrySize(bool reserved) const;
  size_t getOperationsSize() const { return m_operationsSize; }


  // 函数：updateBindings。根据最新状态刷新缓存数据、GPU 地址、描述符或统计信息。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：更新函数负责把“旧状态”推进到“当前状态”，因此要避免部分更新造成 CPU/GPU 视图不一致。
  void updateBindings(const nvvk::Buffer& sceneBuildingBuffer);
#ifndef NDEBUG
  static const int32_t s_defaultDebugFrameLimit = -1;
#else
  static const int32_t s_defaultDebugFrameLimit = -1;
#endif
  int32_t m_debugFrameLimit = s_defaultDebugFrameLimit;

private:
  Resources*   m_resources = nullptr;
  const Scene* m_scene     = nullptr;

  StreamingConfig m_config;
  size_t          m_persistentGeometrySize;
  size_t          m_operationsSize;
  size_t          m_peakGeometrySize = 0;
  uint32_t        m_peakFrameIndex   = ~0;
  uint32_t        m_lastUpdateIndex;
  uint32_t        m_frameIndex;
  StreamingStats  m_stats;


  // 结构：PersistentGeometry。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct PersistentGeometry
  {
    nvvk::BufferTyped<shaderio::Node>     nodes;
    nvvk::BufferTyped<shaderio::BBox>     nodeBboxes;
    nvvk::BufferTyped<shaderio::LodLevel> lodLevels;
    nvvk::BufferTyped<uint64_t>           groupAddresses;
    nvvk::Buffer                          lowDetailGroupsData;
    uint32_t                              lodLevelsCount                                = 0;
    uint32_t                              lodLoadedGroupsCount[SHADERIO_MAX_LOD_LEVELS] = {};
    uint32_t                              lodGroupsCount[SHADERIO_MAX_LOD_LEVELS]       = {};
  };

  std::vector<PersistentGeometry>       m_persistentGeometries;
  std::vector<shaderio::Geometry>       m_shaderGeometries;
  nvvk::BufferTyped<shaderio::Geometry> m_shaderGeometriesBuffer;


  // 函数：initGeometries。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void initGeometries(Resources& res, const Scene* scene);


  // 函数：resetGeometryGroupAddresses。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void resetGeometryGroupAddresses(Resources::BatchedUploader& uploader);
  shaderio::SceneStreaming m_shaderData;
  nvvk::Buffer             m_shaderBuffer;
  StreamingTaskQueue m_requestsTaskQueue;
  StreamingTaskQueue m_storageTaskQueue;
  StreamingTaskQueue m_updatesTaskQueue;
  StreamingRequests m_requests;
  StreamingResident m_resident;
  StreamingAllocator m_clasAllocator;
  StreamingStorage m_storage;
  StreamingUpdates m_updates;
  uint32_t handleCompletedRequest(VkCommandBuffer      cmd,
                                  QueueState&          cmdQueueState,
                                  QueueState&          asyncQueueState,
                                  const FrameSettings& settings,
                                  uint32_t             popRequestIndex);


  // 结构：Shaders。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct Shaders
  {
    shaderc::SpvCompilationResult computeAgeFilterGroups;
    shaderc::SpvCompilationResult computeUpdateSceneRaster;
    shaderc::SpvCompilationResult computeSetup;


    shaderc::SpvCompilationResult computeAllocatorBuildFreeGaps;
    shaderc::SpvCompilationResult computeAllocatorFreeGapsInsert;
    shaderc::SpvCompilationResult computeAllocatorSetupInsertion;
    shaderc::SpvCompilationResult computeAllocatorUnloadGroups;
    shaderc::SpvCompilationResult computeAllocatorLoadGroups;
  };


  // 结构：Pipelines。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct Pipelines
  {
    VkPipeline computeAllocatorBuildFreeGaps  = nullptr;
    VkPipeline computeAllocatorFreeGapsInsert = nullptr;
    VkPipeline computeAllocatorSetupInsertion = nullptr;
    VkPipeline computeAllocatorUnloadGroups   = nullptr;
    VkPipeline computeAllocatorLoadGroups     = nullptr;


    VkPipeline computeAgeFilterGroups   = nullptr;
    VkPipeline computeUpdateSceneRaster = nullptr;
    VkPipeline computeSetup             = nullptr;
  };

  Shaders              m_shaders;
  Pipelines            m_pipelines;
  VkPipelineLayout     m_pipelineLayout{};
  nvvk::DescriptorPack m_dsetPack;


  // 函数：initShadersAndPipelines。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  bool initShadersAndPipelines();


  // 函数：deinitShadersAndPipelines。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinitShadersAndPipelines();
};
}
