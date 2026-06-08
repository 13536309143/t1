//==============================================================================
// 文件：src/streaming/streamutils.hpp
// 模块定位：流式加载基础结构声明，定义请求队列、驻留表、分配器、存储、更新任务和异步任务队列。
// 数据流：GPU 写出 load/unload request；CPU 处理后通过这些结构把结果上传回 GPU 地址表和 resident 状态。
// 方法说明：流式加载把几何驻留视为受限资源分配问题，需要同时管理请求排序、空间分配、传输带宽和可见性保活。
// 正确性约束：residentID 必须稳定映射到 组 状态；任务队列索引不能重复释放；allocator 的 free gap 与 storage 状态必须一致。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#pragma once


// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <queue>
#include <nvutils/logger.hpp>
#include <nvutils/id_pool.hpp>
#include <nvvk/buffer_suballocator.hpp>
#include <nvvk/command_pools.hpp>
#include "scene.hpp"
#include "resources.hpp"
#include "shaderio_streaming.h"


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace lodclusters {

static const uint32_t STREAMING_MAX_ACTIVE_TASKS = 3;
static const uint32_t INVALID_TASK_INDEX         = ~0;


// 结构：StreamingConfig。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct StreamingConfig
{
  bool usePersistentClasAllocator = true;
  bool allowBlasCaching           = true;
  bool useAsyncTransfer           = false;
  bool useDecoupledAsyncTransfer  = false;

  uint32_t maxPerFrameLoadRequests   = 128;
  uint32_t maxPerFrameUnloadRequests = 1024;

  uint32_t maxGroups   = 1 << 16;
  uint32_t maxClusters = 0;

  size_t maxTransferMegaBytes    = 32;
  size_t maxGeometryMegaBytes    = 1024 * 2;
  size_t maxClasMegaBytes        = 1024 * 2;
  size_t maxBlasCachingMegaBytes = 1024;

  VkBuildAccelerationStructureFlagsKHR clasBuildFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  uint32_t                             clasPositionTruncateBits = 0;


  uint32_t clasAllocatorSectorSizeShift = 10;

  uint32_t clasAllocatorGranularityShift = 0;
};


// 结构：StreamingStats。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct StreamingStats
{
  uint32_t residentGroups   = 0;
  uint32_t residentClusters = 0;
  uint32_t maxGroups        = 0;
  uint32_t maxClusters      = 0;

  uint32_t persistentGroups    = 0;
  uint32_t persistentClusters  = 0;
  uint64_t persistentDataBytes = 0;
  uint64_t persistentClasBytes = 0;

  uint64_t maxDataBytes      = 0;
  uint64_t reservedDataBytes = 0;
  uint64_t usedDataBytes     = 0;

  uint64_t reservedClasBytes = 0;
  uint64_t usedClasBytes     = 0;
  uint64_t wastedClasBytes   = 0;
  uint32_t maxSizedLeft      = 0;
  uint32_t maxSizedReserved  = 0;

  uint64_t maxTransferBytes     = 0;
  uint64_t transferBytes        = 0;
  uint32_t transferCount        = 0;
  uint32_t loadCount            = 0;
  uint32_t unloadCount          = 0;
  uint32_t uncompletedLoadCount = 0;
  uint32_t maxLoadCount         = 0;
  uint32_t maxUnloadCount       = 0;

  uint32_t couldNotAllocateGroup = 0;
  uint32_t couldNotAllocateClas  = 0;
  uint32_t couldNotTransfer      = 0;
  uint32_t couldNotStore         = 0;
};

union GeometryGroup
{
  struct
  {
    uint32_t geometryID;
    uint32_t groupID;
  };
  uint64_t key;
};


// 类型：StreamingRequests。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class StreamingRequests
{
public:


  // 结构：TaskInfo。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct TaskInfo
  {
    const shaderio::StreamingRequest* shaderData;
    const GeometryGroup*              loadGeometryGroups;
    const GeometryGroup*              unloadGeometryGroups;
  };


  // 函数：init。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void init(Resources& res, const StreamingConfig& config, uint32_t groupCountAlignment, uint32_t clusterCountAlignment);


  // 函数：deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinit(Resources& res);


  // 函数：getOperationsSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  size_t getOperationsSize() const;


  // 函数：applyTask。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void applyTask(shaderio::StreamingRequest& shaderData, uint32_t taskIndex, uint32_t frameIndex);


  // 函数：cmdRunTask。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
  void cmdRunTask(VkCommandBuffer cmd, const shaderio::StreamingRequest& shaderData, VkBuffer srcBuffer, size_t srcBufferOffset);


  const TaskInfo& getCompletedTask(uint32_t taskIndex) { return m_taskInfos[taskIndex]; }

private:
  nvvk::Buffer m_requestBuffer;
  nvvk::Buffer m_requestHostBuffer;

  uint64_t m_requestSize;
  uint64_t m_shaderDataOffset;

  shaderio::StreamingRequest m_shaderData;
  TaskInfo                   m_taskInfos[STREAMING_MAX_ACTIVE_TASKS];
};


// 类型：StreamingResident。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class StreamingResident
{
public:
  static const uint32_t INVALID_GROUP = ~0;


  // 结构：Group。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct Group
  {
    GeometryGroup             geometryGroup;
    uint32_t                  activeIndex;
    uint32_t                  groupResidentID;
    uint32_t                  clusterResidentID;
    uint16_t                  clusterCount;
    uint16_t                  lodLevel;
    uint64_t                  deviceAddress;
    nvvk::BufferSubAllocation storageHandle;
  };


  // 函数：init。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void init(Resources& res, const StreamingConfig& config, uint32_t groupCountAlignment, uint32_t clusterCountAlignment);
  const Group* initClas(Resources&                   res,
                        const StreamingConfig&       config,
                        shaderio::StreamingResident& shaderData,
                        uint32_t&                    loGroupsCount,
                        uint32_t&                    loClustersCount,
                        uint32_t&                    loMaxGroupClustersCount);


  // 函数：deinitClas。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void         deinitClas(Resources& res);


  // 函数：deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void         deinit(Resources& res);


  // 函数：reset。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void         reset(shaderio::StreamingResident& shaderData);


  // 函数：getOperationsSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  size_t getOperationsSize() const;


  // 函数：getClasOperationsSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  size_t getClasOperationsSize() const;


  // 函数：getStats。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void   getStats(StreamingStats& stats) const;


  // 函数：uploadInitialState。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void                uploadInitialState(Resources::BatchedUploader& uploader, shaderio::StreamingResident& shaderData);
  const nvvk::Buffer& getClasBuffer() const { return m_clasManageBuffer; }


  // 函数：findGroup。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  const StreamingResident::Group* findGroup(GeometryGroup geometryGroup) const;
  const StreamingResident::Group& getGroup(uint32_t groupResidentID) const { return m_groups[groupResidentID]; }


  // 函数：getLoadActiveGroupsOffset。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
  uint32_t getLoadActiveGroupsOffset() const;


  // 函数：getLoadActiveClustersOffset。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
  uint32_t getLoadActiveClustersOffset() const;


  // 函数：canAllocateGroup。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  bool                      canAllocateGroup(uint32_t numClusters) const;


  // 函数：addGroup。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  StreamingResident::Group* addGroup(GeometryGroup geometryGroup, uint32_t clusterCount);


  // 函数：removeGroup。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void                      removeGroup(uint32_t groupResidentID);


  // 函数：cmdUploadTask。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
  size_t cmdUploadTask(VkCommandBuffer cmd, uint32_t taskIndex);


  // 函数：applyTask。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void applyTask(shaderio::StreamingResident& shaderData, uint32_t taskIndex, uint32_t frameIndex);


  // 函数：cmdRunTask。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
  void cmdRunTask(VkCommandBuffer cmd, uint32_t taskIndex);


private:


  // 结构：UpdateRange。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct UpdateRange
  {

    uint32_t lo = uint32_t(~0);
    uint32_t hi = 0;


    // 函数：update。根据最新状态刷新缓存数据、GPU 地址、描述符或统计信息。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：更新函数负责把“旧状态”推进到“当前状态”，因此要避免部分更新造成 CPU/GPU 视图不一致。
    void update(uint32_t index)
    {

      lo = std::min(lo, index);

      hi = std::max(hi, index);
    }

    uint32_t count() const { return hi == 0 && lo == ~0 ? 0 : 1 + hi - lo; }
  };


  // 结构：TaskInfo。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct TaskInfo
  {
    VkBufferCopy                region;
    shaderio::StreamingResident shaderData;
  };


  std::unordered_map<uint64_t, uint32_t> m_mapGeometryGroup2Residency;

  nvutils::IDPool m_groupAllocator;
  nvutils::IDPool m_clusterAllocator;

  uint32_t m_maxClusters;
  uint32_t m_maxGroups;
  size_t   m_maxClasBytes;

  std::vector<Group> m_groups;


  std::vector<uint32_t> m_activeGroupIndices;

  uint32_t m_lowDetailGroupsCount;
  uint32_t m_lowDetailClustersCount;
  uint32_t m_lowDetailMaxGroupClusters;

  uint32_t m_activeGroupsCount;
  uint32_t m_activeClustersCount;

  nvvk::Buffer m_residentBuffer;
  uint64_t     m_residentGroupsOffset;
  uint64_t     m_residentGroupIDsOffset;
  uint64_t     m_residentClustersOffset;
  uint64_t     m_residentActiveOffset;
  uint64_t     m_residentActiveUpdateOffset;

  nvvk::Buffer      m_clasManageBuffer;
  nvvk::LargeBuffer m_clasDataBuffer;

  shaderio::StreamingResident m_shaderData;

  nvvk::BufferTyped<uint32_t> m_residentActiveHostBuffer;
  UpdateRange                 m_groupIndicesUpdateRange;

  TaskInfo m_taskInfos[STREAMING_MAX_ACTIVE_TASKS];
};


// 类型：StreamingAllocator。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class StreamingAllocator
{
public:
  void init(Resources&                    res,
            size_t                        totalMegaBytes,
            uint32_t                      maxAllocationByteSize,
            uint32_t                      granularityByteSize,
            uint32_t                      sectorSizeShift,
            shaderio::StreamingAllocator& shaderData);


  // 函数：deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinit(Resources& res);


  // 函数：getOperationsSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  size_t   getOperationsSize() const;


  // 函数：getMaxSized。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  uint32_t getMaxSized() const;


  // 函数：cmdReset。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
  void cmdReset(VkCommandBuffer cmd);


  // 函数：cmdBeginFrame。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
  void cmdBeginFrame(VkCommandBuffer cmd);

private:
  shaderio::StreamingAllocator m_shaderData;

  nvvk::Buffer m_managementBuffer;
};


// 类型：StreamingUpdates。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class StreamingUpdates
{
public:


  // 结构：TaskInfo。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct TaskInfo
  {
    uint32_t                          loadCount;
    uint32_t                          unloadCount;
    uint32_t                          newClusterCount;
    uint32_t                          loadActiveGroupsOffset;
    uint32_t                          loadActiveClustersOffset;
    uint32_t                          geometryCachedCount;
    uint32_t                          geometryCachedClustersCount;
    shaderio::StreamingPatch*         loadPatches;
    shaderio::StreamingPatch*         unloadPatches;
    nvvk::BufferSubAllocation*        unloadHandles;
    shaderio::StreamingGeometryPatch* geometryPatches;
  };


  // 结构：NewInfo。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct NewInfo
  {
    uint32_t groups   = 0;
    uint32_t clusters = 0;
  };


  // 函数：init。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void init(Resources& res, const StreamingConfig& config, uint32_t geometryCount, uint32_t groupCountAlignment, uint32_t clusterCountAlignment);


  // 函数：initClas。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void initClas(Resources& res, const StreamingConfig& config, const SceneConfig& sceneConfig);


  // 函数：deinitClas。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinitClas(Resources& res);


  // 函数：deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinit(Resources& res);


  // 函数：getOperationsSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  size_t   getOperationsSize() const;


  // 函数：getClasOperationsSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  size_t   getClasOperationsSize() const;


  // 函数：getMaxCachedBlasBuilds。构建派生数据结构，通常用于 LOD、层次结构、间接命令或加速访问。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：构建结果会被后续阶段高频读取，必须保证布局紧凑、索引合法并与共享结构定义一致。
  uint32_t getMaxCachedBlasBuilds() const;


  // 函数：reset。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void reset();


  // 函数：getFutureNew。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  NewInfo getFutureNew(uint64_t frameIndex) const
  {


    NewInfo info = m_pendingNew;


    for(uint32_t i = 0; i < STREAMING_MAX_ACTIVE_TASKS; i++)
    {
      if(m_scheduledNewFrame[i] > frameIndex)
      {
        info.groups += m_scheduledNew[i].groups;
        info.clusters += m_scheduledNew[i].clusters;
      }
    }
    return info;
  }


  // 函数：getNewTask。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  TaskInfo& getNewTask(uint32_t taskIndex);


  // 函数：cmdUploadTask。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
  size_t cmdUploadTask(VkCommandBuffer cmd, uint32_t taskIndex);


  // 函数：applyTask。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void applyTask(shaderio::StreamingUpdate& shaderData, uint32_t taskIndex, uint32_t frameIndex);


  const TaskInfo& getCompletedTask(uint32_t taskIndex) const { return m_taskInfos[taskIndex]; }

private:
  bool m_useBlasCaching = false;

  nvvk::BufferTyped<shaderio::StreamingPatch> m_patchesBuffer;
  nvvk::BufferTyped<shaderio::StreamingPatch> m_patchesHostBuffer;

  std::vector<nvvk::BufferSubAllocation> m_unloadHandles;
  TaskInfo                               m_taskInfos[STREAMING_MAX_ACTIVE_TASKS];

  shaderio::StreamingUpdate m_shaderData;

  uint32_t m_maxCachedBlasBuilds;
  uint32_t m_clusterCountAlignment;
  uint32_t m_scheduleIndex;
  NewInfo  m_pendingNew;
  NewInfo  m_scheduledNew[STREAMING_MAX_ACTIVE_TASKS]      = {};
  uint64_t m_scheduledNewFrame[STREAMING_MAX_ACTIVE_TASKS] = {};


  nvvk::Buffer m_clasBuffer;
};


// 类型：StreamingStorage。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class StreamingStorage
{
public:


  // 结构：TaskInfo。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct TaskInfo
  {
    size_t usedMemory;
    size_t baseOffset;
    size_t regionCount;
  };


  // 函数：init。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void init(Resources& res, const StreamingConfig& config);


  // 函数：deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinit(Resources& res);


  // 函数：reset。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void reset();


  // 函数：free。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void free(nvvk::BufferSubAllocation& handle);


  // 函数：getStats。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void   getStats(StreamingStats& stats) const;


  // 函数：getOperationsSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  size_t getOperationsSize() const;


  // 函数：getMaxDataSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  size_t getMaxDataSize() const;


  // 函数：getNewTask。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  TaskInfo& getNewTask(uint32_t taskIndex);


  // 函数：canTransfer。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  bool canTransfer(const TaskInfo& operation, size_t size) const;


  // 函数：allocate。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  bool allocate(nvvk::BufferSubAllocation& handle, GeometryGroup group, size_t sz, uint64_t& deviceAddress);


  // 函数：appendTransfer。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void* appendTransfer(TaskInfo& operation, const nvvk::BufferSubAllocation& dstHandle);


  // 函数：cmdUploadTask。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
  uint32_t cmdUploadTask(VkCommandBuffer cmd);

  nvvk::ManagedCommandPools m_taskCommandPool;

private:


  // 结构：CopyInfo。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct CopyInfo
  {
    VkBuffer targetBuffer;
    size_t   regionOffset;
    size_t   regionCount;
  };

  size_t m_maxSceneBytes;
  size_t m_maxTransferBytes;
  size_t m_blockBytes;

  nvvk::Buffer                       m_transferHostBuffer;
  nvvk::BufferSubAllocator::InitInfo m_dataInfo;
  nvvk::BufferSubAllocator           m_dataAllocator;
  std::vector<uint32_t>              m_dataQueueFamilies;

  std::vector<CopyInfo>     m_copyInfos;
  std::vector<VkBufferCopy> m_copyRegions;

  TaskInfo m_taskOperations[STREAMING_MAX_ACTIVE_TASKS];
};


// 类型：StreamingTaskQueue。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class StreamingTaskQueue
{
public:

  static_assert(STREAMING_MAX_ACTIVE_TASKS < 32);

  StreamingTaskQueue() { m_availableTaskBits = (1 << STREAMING_MAX_ACTIVE_TASKS) - 1; }


  // 函数：acquireTaskIndex。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  uint32_t acquireTaskIndex()
  {

    for(uint32_t i = 0; i < STREAMING_MAX_ACTIVE_TASKS; i++)
    {
      if(m_availableTaskBits & (1 << i))
      {

        m_availableTaskBits &= ~(1 << i);
        return i;
      }
    }

    return INVALID_TASK_INDEX;
  }


  // 函数：releaseTaskIndex。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void releaseTaskIndex(uint32_t index)
  {
    assert((m_availableTaskBits & (1 << index)) == 0);
    m_availableTaskBits |= (1 << index);
  }


  // 函数：canPop。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  bool canPop(VkDevice device, bool ensureAcquisition)
  {
    if(ensureAcquisition && !m_availableTaskBits && !m_taskQueue.empty())
    {


      if(m_taskQueue.front().semaphoreState.wait(device, ~0ULL) == VK_TIMEOUT)
      {

        LOGE("Failure to wait for semaphore");
        {

          exit(-1);
        }
      }
    }

    return !m_taskQueue.empty() && m_taskQueue.front().semaphoreState.testSignaled(device);
  }


  void push(uint32_t taskIndex, nvvk::SemaphoreState semaphoreState, uint32_t dependentIndex = INVALID_TASK_INDEX)
  {
    Task task = {
        .semaphoreState = semaphoreState,
        .taskIndex      = taskIndex,
        .dependentIndex = dependentIndex,
    };

    m_taskQueue.push(task);
  }


  // 函数：pop。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  uint32_t pop()
  {
    uint32_t taskIndex = m_taskQueue.front().taskIndex;

    assert(taskIndex != INVALID_TASK_INDEX);

    m_taskQueue.pop();
    return taskIndex;
  }


  // 函数：popWithDependent。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  uint32_t popWithDependent(uint32_t& dependentIndex)
  {
    uint32_t taskIndex = m_taskQueue.front().taskIndex;

    assert(taskIndex != INVALID_TASK_INDEX);
    dependentIndex = m_taskQueue.front().dependentIndex;

    m_taskQueue.pop();
    return taskIndex;
  }

private:


  // 结构：Task。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct Task
  {
    nvvk::SemaphoreState semaphoreState;
    uint32_t             taskIndex      = INVALID_TASK_INDEX;
    uint32_t             dependentIndex = INVALID_TASK_INDEX;
  };

  std::queue<Task> m_taskQueue;
  uint32_t         m_availableTaskBits;
};
}
