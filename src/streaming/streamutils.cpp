//==============================================================================
// 文件：src/streaming/streamutils.cpp
// 模块定位：流式加载基础结构实现，管理 GPU 缓冲、驻留 组、几何存储分配、暂存上传和统计汇总。
// 数据流：输入是 SceneStreaming 的任务和配置；输出是可提交的上传命令、地址修补任务和统计信息。
// 方法说明：实现层把 流式加载 的离散事件规约为一组固定 缓冲 中的批处理任务，使 GPU 和 CPU 可以通过有限同步点协作。
// 正确性约束：每次 load/unload 都必须同步更新 resident、storage 和 着色器 数据；上传范围不能超过 transfer budget。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <volk.h>
#include "streamutils.hpp"


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace lodclusters {


// 函数：StreamingRequests::init。初始化本模块所需状态、资源或 GPU 侧绑定。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
void StreamingRequests::init(Resources& res, const StreamingConfig& config, uint32_t groupCountAlignment, uint32_t clusterCountAlignment)
{
  m_shaderData            = {};
  m_shaderData.maxLoads   = config.maxPerFrameLoadRequests;
  m_shaderData.maxUnloads = config.maxPerFrameUnloadRequests;


  BufferRanges ranges = {};
  m_shaderData.loadGeometryGroups =
      ranges.append(sizeof(GeometryGroup) * nvutils::align_up(config.maxPerFrameLoadRequests, groupCountAlignment), 8);
  m_shaderData.unloadGeometryGroups =
      ranges.append(sizeof(GeometryGroup) * nvutils::align_up(config.maxPerFrameUnloadRequests, groupCountAlignment), 8);

  m_requestSize = ranges.getSize();


  m_shaderDataOffset = m_requestSize * STREAMING_MAX_ACTIVE_TASKS;

  std::vector<uint32_t> sharingQueueFamilies;
  if(config.useAsyncTransfer)
  {

    sharingQueueFamilies.push_back(res.m_queueStates.primary.m_familyIndex);

    sharingQueueFamilies.push_back(res.m_queueStates.transfer.m_familyIndex);
  }

  res.createBuffer(m_requestBuffer, (m_requestSize + sizeof(shaderio::StreamingRequest)) * STREAMING_MAX_ACTIVE_TASKS,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, 0, sharingQueueFamilies);

  NVVK_DBG_NAME(m_requestBuffer.buffer);

  res.createBuffer(m_requestHostBuffer, m_requestBuffer.bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                   VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

  NVVK_DBG_NAME(m_requestHostBuffer.buffer);

  for(uint32_t c = 0; c < STREAMING_MAX_ACTIVE_TASKS; c++)
  {
    TaskInfo& task = m_taskInfos[c];
    task           = {};

    task.shaderData = reinterpret_cast<const shaderio::StreamingRequest*>(
        uint64_t(m_requestHostBuffer.mapping) + m_shaderDataOffset + sizeof(shaderio::StreamingRequest) * c);
    task.loadGeometryGroups = reinterpret_cast<const GeometryGroup*>(
        uint64_t(m_requestHostBuffer.mapping) + m_requestSize * c + m_shaderData.loadGeometryGroups);
    task.unloadGeometryGroups = reinterpret_cast<const GeometryGroup*>(
        uint64_t(m_requestHostBuffer.mapping) + m_requestSize * c + m_shaderData.unloadGeometryGroups);
  }
}


// 函数：StreamingRequests::deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
void StreamingRequests::deinit(Resources& res)
{

  res.m_allocator.destroyBuffer(m_requestBuffer);

  res.m_allocator.destroyBuffer(m_requestHostBuffer);
}


// 函数：StreamingRequests::getOperationsSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
size_t StreamingRequests::getOperationsSize() const
{
  return m_requestBuffer.bufferSize;
}


// 函数：StreamingRequests::applyTask。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void StreamingRequests::applyTask(shaderio::StreamingRequest& shaderData, uint32_t taskIndex, uint32_t frameIndex)
{
  shaderData = m_shaderData;
  shaderData.loadGeometryGroups += m_requestBuffer.address + m_requestSize * taskIndex;
  shaderData.unloadGeometryGroups += m_requestBuffer.address + m_requestSize * taskIndex;
  shaderData.taskIndex = taskIndex;


  shaderData.frameIndex = STREAMING_INVALID_ADDRESS_START + frameIndex;
}


// 函数：StreamingRequests::cmdRunTask。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
void StreamingRequests::cmdRunTask(VkCommandBuffer cmd, const shaderio::StreamingRequest& shaderData, VkBuffer buffer, size_t bufferOffset)
{
  uint32_t taskIndex = shaderData.taskIndex;


  VkBufferCopy region;
  region.dstOffset = m_requestSize * taskIndex;
  region.srcOffset = m_requestSize * taskIndex;
  region.size      = m_requestSize;

  vkCmdCopyBuffer(cmd, m_requestBuffer.buffer, m_requestHostBuffer.buffer, 1, &region);


  region.dstOffset = m_shaderDataOffset + (sizeof(shaderio::StreamingRequest) * taskIndex);
  region.srcOffset = bufferOffset;
  region.size      = sizeof(shaderio::StreamingRequest);

  vkCmdCopyBuffer(cmd, buffer, m_requestHostBuffer.buffer, 1, &region);
}


// 函数：StreamingResident::init。初始化本模块所需状态、资源或 GPU 侧绑定。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
void StreamingResident::init(Resources& res, const StreamingConfig& config, uint32_t groupCountAlignment, uint32_t clusterCountAlignment)
{

  m_groupAllocator.init(config.maxGroups);

  m_clusterAllocator.init(config.maxClusters);


  m_maxClusters  = nvutils::align_up(config.maxClusters, clusterCountAlignment);

  m_maxGroups    = nvutils::align_up(config.maxGroups, groupCountAlignment);
  m_maxClasBytes = 0;

  m_lowDetailGroupsCount   = 0;
  m_lowDetailClustersCount = 0;

  m_activeGroupsCount   = 0;
  m_activeClustersCount = 0;

  m_groupIndicesUpdateRange    = {};
  m_groups                     = {};
  m_activeGroupIndices         = {};
  m_mapGeometryGroup2Residency = {};


  m_mapGeometryGroup2Residency.reserve(m_maxGroups);

  m_groups.resize(m_maxGroups);

  m_activeGroupIndices.resize(m_maxGroups);

  BufferRanges ranges      = {};
  m_residentGroupsOffset   = ranges.append(sizeof(shaderio::StreamingGroup) * m_maxGroups, 16);
  m_residentGroupIDsOffset = ranges.append(sizeof(shaderio::uint32_t) * m_maxGroups, 4);
  m_residentClustersOffset = ranges.append(sizeof(shaderio::uint64_t) * m_maxClusters, 8);
  m_residentActiveOffset   = ranges.append(sizeof(shaderio::uint32_t) * m_maxGroups, 4);
  m_residentActiveUpdateOffset = ranges.append(sizeof(shaderio::uint32_t) * m_maxGroups * STREAMING_MAX_ACTIVE_TASKS, 4);

  std::vector<uint32_t> sharingQueueFamilies;
  if(config.useAsyncTransfer)
  {

    sharingQueueFamilies.push_back(res.m_queueStates.primary.m_familyIndex);

    sharingQueueFamilies.push_back(res.m_queueStates.transfer.m_familyIndex);
  }

  res.createBuffer(m_residentBuffer, ranges.getSize(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, 0, sharingQueueFamilies);

  NVVK_DBG_NAME(m_residentBuffer.buffer);

  res.createBufferTyped(m_residentActiveHostBuffer, (m_maxGroups)*STREAMING_MAX_ACTIVE_TASKS,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY,
                        VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

  NVVK_DBG_NAME(m_residentActiveHostBuffer.buffer);

  m_shaderData              = {};
  m_shaderData.groups       = m_residentBuffer.address + m_residentGroupsOffset;
  m_shaderData.groupIDs     = m_residentBuffer.address + m_residentGroupIDsOffset;
  m_shaderData.clusters     = m_residentBuffer.address + m_residentClustersOffset;
  m_shaderData.activeGroups = m_residentBuffer.address + m_residentActiveOffset;
}


// 函数：StreamingResident::deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
void StreamingResident::deinit(Resources& res)
{

  m_groupAllocator.destroyAll();

  m_clusterAllocator.destroyAll();


  res.m_allocator.destroyBuffer(m_residentBuffer);

  res.m_allocator.destroyBuffer(m_residentActiveHostBuffer);

  deinitClas(res);

  *this = {};
}


// 函数：StreamingResident::getOperationsSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
size_t StreamingResident::getOperationsSize() const
{
  return m_residentBuffer.bufferSize;
}

const StreamingResident::Group* StreamingResident::initClas(Resources&                   res,
                                                            const StreamingConfig&       config,
                                                            shaderio::StreamingResident& shaderData,
                                                            uint32_t&                    loGroupsCount,
                                                            uint32_t&                    loClustersCount,
                                                            uint32_t&                    loMaxGroupClustersCount)
{
  m_maxClasBytes = config.maxClasMegaBytes * 1024 * 1024;

  BufferRanges ranges                    = {};
  m_shaderData.clasAddresses             = ranges.append(sizeof(shaderio::uint64_t) * m_maxClusters, 8);
  m_shaderData.clasSizes                 = ranges.append(sizeof(shaderio::uint32_t) * m_maxClusters, 4);
  m_shaderData.clasCompactionUsedSize    = ranges.append(sizeof(shaderio::uint64_t), 8);
  m_shaderData.clasAllocatedMaxSizedLeft = ranges.append(sizeof(shaderio::uint32_t), 4);
  if(config.usePersistentClasAllocator)
  {
    m_shaderData.groupClasSizes = ranges.append(sizeof(glm::uvec2) * m_maxGroups, 8);
  }


  res.createBuffer(m_clasManageBuffer, ranges.getSize(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  NVVK_DBG_NAME(m_clasManageBuffer.buffer);

  m_shaderData.clasAddresses += m_clasManageBuffer.address;
  m_shaderData.clasSizes += m_clasManageBuffer.address;
  m_shaderData.clasCompactionUsedSize += m_clasManageBuffer.address;
  m_shaderData.clasAllocatedMaxSizedLeft += m_clasManageBuffer.address;
  if(config.usePersistentClasAllocator)
  {
    m_shaderData.groupClasSizes += m_clasManageBuffer.address;
  }


  res.createLargeBuffer(m_clasDataBuffer, m_maxClasBytes,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

  NVVK_DBG_NAME(m_clasDataBuffer.buffer);

  m_shaderData.clasBaseAddress = m_clasDataBuffer.address;
  m_shaderData.clasMaxSize     = m_maxClasBytes;

  shaderData = m_shaderData;

  loGroupsCount           = m_lowDetailGroupsCount;
  loClustersCount         = m_lowDetailClustersCount;
  loMaxGroupClustersCount = m_lowDetailMaxGroupClusters;

  return m_groups.data();
}


// 函数：StreamingResident::getClasOperationsSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
size_t StreamingResident::getClasOperationsSize() const
{
  return m_clasManageBuffer.bufferSize;
}


// 函数：StreamingResident::getStats。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void StreamingResident::getStats(StreamingStats& stats) const
{
  stats.residentGroups     = m_activeGroupsCount;
  stats.residentClusters   = m_activeClustersCount;
  stats.persistentGroups   = m_lowDetailGroupsCount;
  stats.persistentClusters = m_lowDetailClustersCount;
}


// 函数：StreamingResident::deinitClas。释放或回收前面初始化的资源，保持生命周期成对管理。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
void StreamingResident::deinitClas(Resources& res)
{

  res.m_allocator.destroyBuffer(m_clasManageBuffer);

  res.m_allocator.destroyLargeBuffer(m_clasDataBuffer);

  m_shaderData.clasBaseAddress           = 0;
  m_shaderData.clasAddresses             = 0;
  m_shaderData.clasSizes                 = 0;
  m_shaderData.clasCompactionUsedSize    = 0;
  m_shaderData.clasAllocatedMaxSizedLeft = 0;
  m_shaderData.groupClasSizes            = 0;
  m_shaderData.clasMaxSize               = 0;
}


// 函数：StreamingResident::reset。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void StreamingResident::reset(shaderio::StreamingResident& shaderData)
{
  for(uint32_t activeGroup = m_lowDetailGroupsCount; activeGroup < m_activeGroupsCount; activeGroup++)
  {
    Group& group = m_groups[m_activeGroupIndices[activeGroup]];


    m_mapGeometryGroup2Residency.erase(group.geometryGroup.key);

    m_groupAllocator.destroyID(group.groupResidentID);

    m_clusterAllocator.destroyRangeID(group.clusterResidentID, group.clusterCount);
  }

  m_activeGroupsCount   = m_lowDetailGroupsCount;
  m_activeClustersCount = m_lowDetailClustersCount;

  m_groupIndicesUpdateRange = {};


  m_shaderData.activeGroupsCount   = m_activeGroupsCount - m_lowDetailGroupsCount;
  m_shaderData.activeClustersCount = m_activeClustersCount - m_lowDetailClustersCount;
  m_shaderData.activeGroups = m_residentBuffer.address + m_residentActiveOffset + sizeof(uint32_t) * m_lowDetailGroupsCount;

  shaderData = m_shaderData;
}


// 函数：StreamingResident::uploadInitialState。初始化本模块所需状态、资源或 GPU 侧绑定。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
void StreamingResident::uploadInitialState(Resources::BatchedUploader& uploader, shaderio::StreamingResident& shaderData)
{


  m_lowDetailGroupsCount      = m_activeGroupsCount;
  m_lowDetailClustersCount    = m_activeClustersCount;
  m_lowDetailMaxGroupClusters = 0;


  uint32_t updatedActiveGroups = m_lowDetailGroupsCount;
#if STREAMING_DEBUG_ADDRESSES
  updatedActiveGroups = m_maxGroups;
#endif

  shaderio::StreamingGroup* shaderGroups =
      uploader.uploadBuffer(m_residentBuffer, m_residentGroupsOffset,
                            sizeof(shaderio::StreamingGroup) * updatedActiveGroups, (shaderio::StreamingGroup*)nullptr);

  uint64_t* shaderClusters = uploader.uploadBuffer(m_residentBuffer, m_residentClustersOffset,
                                                   sizeof(shaderio::uint64_t) * m_activeClustersCount,
                                                   (uint64_t*)nullptr, Resources::DONT_FLUSH);

  for(uint32_t g = 0; g < m_lowDetailGroupsCount; g++)
  {
    const Group& group = m_groups[g];

    assert(group.groupResidentID == g);

    shaderio::StreamingGroup& shaderGroup = shaderGroups[g];
    shaderGroup.age                       = 0x1234;
    shaderGroup.lodLevel                  = group.lodLevel;
    shaderGroup.group                     = group.deviceAddress;
    for(uint32_t c = 0; c < group.clusterCount; c++)
    {
      shaderClusters[group.clusterResidentID + c] = group.deviceAddress + sizeof(shaderio::Group) + sizeof(shaderio::Cluster) * c;
    }
    m_lowDetailMaxGroupClusters = std::max(m_lowDetailMaxGroupClusters, uint32_t(group.clusterCount));
  }
#if STREAMING_DEBUG_ADDRESSES

  for(uint32_t g = m_lowDetailGroupsCount; g < updatedActiveGroups; g++)
  {
    shaderGroups[g].group = STREAMING_INVALID_ADDRESS_START;
  }
#endif


  m_shaderData.activeGroupsCount   = m_activeGroupsCount - m_lowDetailGroupsCount;
  m_shaderData.activeClustersCount = m_activeClustersCount - m_lowDetailClustersCount;
  m_shaderData.activeGroups = m_residentBuffer.address + m_residentActiveOffset + sizeof(uint32_t) * m_lowDetailGroupsCount;

  shaderData = m_shaderData;
}


// 函数：StreamingResident::cmdUploadTask。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
size_t StreamingResident::cmdUploadTask(VkCommandBuffer cmd, uint32_t taskIndex)
{
  TaskInfo& task = m_taskInfos[taskIndex];

  uint32_t taskOffset = m_maxGroups * taskIndex;

  task.region     = {};
  task.shaderData = m_shaderData;

  task.shaderData.activeGroupsCount   = m_activeGroupsCount - m_lowDetailGroupsCount;
  task.shaderData.activeClustersCount = m_activeClustersCount - m_lowDetailClustersCount;


  uint32_t deltaCount = m_groupIndicesUpdateRange.count();
  if(!deltaCount)
  {
    return 0;
  }


  memcpy(m_residentActiveHostBuffer.data() + taskOffset, m_activeGroupIndices.data() + m_groupIndicesUpdateRange.lo,
         sizeof(uint32_t) * deltaCount);


  VkBufferCopy region;
  region.size      = sizeof(uint32_t) * deltaCount;
  region.srcOffset = sizeof(uint32_t) * taskOffset;
  region.dstOffset = m_residentActiveUpdateOffset + sizeof(uint32_t) * (taskOffset);

  vkCmdCopyBuffer(cmd, m_residentActiveHostBuffer.buffer, m_residentBuffer.buffer, 1, &region);


  task.region.size      = region.size;
  task.region.srcOffset = region.dstOffset;

  task.region.dstOffset = m_residentActiveOffset + sizeof(uint32_t) * m_groupIndicesUpdateRange.lo;


  m_groupIndicesUpdateRange = {};

  return region.size;
}


// 函数：StreamingResident::applyTask。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void StreamingResident::applyTask(shaderio::StreamingResident& shaderData, uint32_t taskIndex, uint32_t frameIndex)
{
  shaderData            = m_taskInfos[taskIndex].shaderData;
  shaderData.taskIndex  = taskIndex;
  shaderData.frameIndex = frameIndex;
}


// 函数：StreamingResident::cmdRunTask。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
void StreamingResident::cmdRunTask(VkCommandBuffer cmd, uint32_t taskIndex)
{
  TaskInfo& task = m_taskInfos[taskIndex];
  if(task.region.size)
  {

    vkCmdCopyBuffer(cmd, m_residentBuffer.buffer, m_residentBuffer.buffer, 1, &task.region);
  }
}


// 函数：StreamingResident::getLoadActiveGroupsOffset。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
uint32_t StreamingResident::getLoadActiveGroupsOffset() const
{
  return m_activeGroupsCount - m_lowDetailGroupsCount;
}


// 函数：StreamingResident::getLoadActiveClustersOffset。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
uint32_t StreamingResident::getLoadActiveClustersOffset() const
{
  return m_activeClustersCount - m_lowDetailClustersCount;
}


// 函数：StreamingResident::canAllocateGroup。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
bool StreamingResident::canAllocateGroup(uint32_t numClusters) const
{
  return m_groupAllocator.isRangeAvailable(1) && m_clusterAllocator.isRangeAvailable(numClusters);
}


// 函数：StreamingResident::findGroup。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
const StreamingResident::Group* StreamingResident::findGroup(GeometryGroup geometryGroup) const
{

  auto it = m_mapGeometryGroup2Residency.find(geometryGroup.key);
  if(it == m_mapGeometryGroup2Residency.end())
  {
    return nullptr;
  }
  else
  {
    return &m_groups[it->second];
  }
}


// 函数：StreamingResident::addGroup。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
StreamingResident::Group* StreamingResident::addGroup(GeometryGroup geometryGroup, uint32_t clusterCount)
{
  bool     valid = false;
  uint32_t groupResidentID;
  uint32_t clusterResidentID;

  valid = m_groupAllocator.createID(groupResidentID);

  assert(valid);

  valid = m_clusterAllocator.createRangeID(clusterResidentID, clusterCount);

  assert(valid);

  StreamingResident::Group& group = m_groups[groupResidentID];

  assert(m_mapGeometryGroup2Residency.find(geometryGroup.key) == m_mapGeometryGroup2Residency.end());
  m_mapGeometryGroup2Residency.insert({geometryGroup.key, groupResidentID});

  group.activeIndex       = m_activeGroupsCount++;
  group.geometryGroup     = geometryGroup;
  group.groupResidentID   = groupResidentID;
  group.clusterResidentID = clusterResidentID;
  group.clusterCount      = clusterCount;
  group.deviceAddress     = STREAMING_INVALID_ADDRESS_START;

  m_activeGroupIndices[group.activeIndex] = groupResidentID;


  m_activeClustersCount += clusterCount;

  return &m_groups[groupResidentID];
}


// 函数：StreamingResident::removeGroup。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void StreamingResident::removeGroup(uint32_t groupResidentID)
{
  StreamingResident::Group& group = m_groups[groupResidentID];
  assert(m_mapGeometryGroup2Residency.find(group.geometryGroup.key) != m_mapGeometryGroup2Residency.end());

  m_mapGeometryGroup2Residency.erase(group.geometryGroup.key);

  {

    uint32_t activeIndex = group.activeIndex;


    if(activeIndex + 1 != m_activeGroupsCount)
    {
      uint32_t lastResidentID              = m_activeGroupIndices[m_activeGroupsCount - 1];
      m_groups[lastResidentID].activeIndex = activeIndex;
      m_activeGroupIndices[activeIndex]    = lastResidentID;


      m_groupIndicesUpdateRange.update(activeIndex);
    }
    m_activeGroupsCount--;
  }

  m_activeClustersCount -= group.clusterCount;


  m_groupAllocator.destroyID(groupResidentID);

  m_clusterAllocator.destroyRangeID(group.clusterResidentID, group.clusterCount);

  group = {};
}


void StreamingAllocator::init(Resources&                    res,
                              size_t                        totalMegaBytes,
                              uint32_t                      maxAllocationByteSize,
                              uint32_t                      granularityByteSize,
                              uint32_t                      sectorSizeShift,
                              shaderio::StreamingAllocator& shaderData)
{

  granularityByteSize = std::max(1u, granularityByteSize);


  assert(sectorSizeShift > 5 && granularityByteSize <= 0xFFFF);

  uint32_t granularityByteShift = 0;
  while((1u << granularityByteShift) < granularityByteSize && granularityByteShift <= 16)
  {
    granularityByteShift++;
  }

  assert(granularityByteShift <= 16 && granularityByteSize == (1u << granularityByteShift));

  size_t sectorSize32s = size_t(1) << sectorSizeShift;
  size_t memoryBits    = size_t(totalMegaBytes) * 1024 * 1024 / granularityByteSize;
  size_t memory32s     = memoryBits / 32;
  size_t sectorCount   = memory32s / sectorSize32s;

  m_shaderData                      = {};
  m_shaderData.freeGapsCounter      = 0;
  m_shaderData.granularityByteShift = granularityByteShift;

  m_shaderData.maxAllocationSize = (((maxAllocationByteSize + granularityByteSize - 1) / granularityByteSize) + 31) & (~31);
  m_shaderData.sectorSizeShift          = sectorSizeShift;

  m_shaderData.sectorMaxAllocationSized = uint32_t(sectorSize32s * 32 / m_shaderData.maxAllocationSize);

  m_shaderData.sectorCount              = uint32_t(sectorCount);


  m_shaderData.baseWastedSize = uint32_t(memory32s - (sectorCount * sectorSize32s));


  memory32s = sectorCount * sectorSize32s;

  BufferRanges ranges            = {};
  m_shaderData.freeGapsPos       = ranges.append(sizeof(uint32_t) * memory32s, 4);
  m_shaderData.freeGapsSize      = ranges.append(sizeof(uint16_t) * memory32s, 4);
  m_shaderData.freeGapsPosBinned = ranges.append(sizeof(uint32_t) * memory32s, 4);
  m_shaderData.freeSizeRanges    = ranges.append(sizeof(shaderio::AllocatorRange) * m_shaderData.maxAllocationSize, 8);
  m_shaderData.usedSectorBits    = ranges.append(sizeof(uint32_t) * ((sectorCount + 31) / 32), 4);
  m_shaderData.usedBits          = ranges.append(sizeof(uint32_t) * memory32s, 4);
  m_shaderData.stats             = ranges.append(sizeof(shaderio::AllocatorStats), 8);

  res.createBuffer(m_managementBuffer, ranges.getSize(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  NVVK_DBG_NAME(m_managementBuffer.buffer);

  m_shaderData.freeGapsPos += m_managementBuffer.address;
  m_shaderData.freeGapsSize += m_managementBuffer.address;
  m_shaderData.freeGapsPosBinned += m_managementBuffer.address;
  m_shaderData.freeSizeRanges += m_managementBuffer.address;
  m_shaderData.usedSectorBits += m_managementBuffer.address;
  m_shaderData.usedBits += m_managementBuffer.address;
  m_shaderData.stats += m_managementBuffer.address;

  m_shaderData.dispatchFreeGapsInsert.gridX = 1;
  m_shaderData.dispatchFreeGapsInsert.gridY = 1;
  m_shaderData.dispatchFreeGapsInsert.gridZ = 1;

  shaderData = m_shaderData;
}


// 函数：StreamingAllocator::deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
void StreamingAllocator::deinit(Resources& res)
{

  res.m_allocator.destroyBuffer(m_managementBuffer);
}


// 函数：StreamingAllocator::getOperationsSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
size_t StreamingAllocator::getOperationsSize() const
{
  return m_managementBuffer.bufferSize;
}


// 函数：StreamingAllocator::getMaxSized。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
uint32_t StreamingAllocator::getMaxSized() const
{
  return m_shaderData.sectorMaxAllocationSized * m_shaderData.sectorCount;
}


// 函数：StreamingAllocator::cmdReset。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
void StreamingAllocator::cmdReset(VkCommandBuffer cmd)
{

  vkCmdFillBuffer(cmd, m_managementBuffer.buffer, 0, m_managementBuffer.bufferSize, 0);
}


// 函数：StreamingAllocator::cmdBeginFrame。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
void StreamingAllocator::cmdBeginFrame(VkCommandBuffer cmd)
{

  vkCmdFillBuffer(cmd, m_managementBuffer.buffer, m_shaderData.freeSizeRanges - m_managementBuffer.address,
                  sizeof(shaderio::AllocatorRange) * m_shaderData.maxAllocationSize, 0);
}


// 函数：StreamingUpdates::init。初始化本模块所需状态、资源或 GPU 侧绑定。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
void StreamingUpdates::init(Resources& res, const StreamingConfig& config, uint32_t geometryCount, uint32_t groupCountAlignment, uint32_t clusterCountAlignment)
{
  m_useBlasCaching        = config.allowBlasCaching;
  m_clusterCountAlignment = clusterCountAlignment;
  m_scheduleIndex         = 0;
  m_pendingNew            = {};

  memset(m_scheduledNew, 0, sizeof(m_scheduledNew));
  memset(m_scheduledNewFrame, 0, sizeof(m_scheduledNewFrame));


  uint32_t loadRequests   = nvutils::align_up(config.maxPerFrameLoadRequests, groupCountAlignment);

  uint32_t unloadRequests = nvutils::align_up(config.maxPerFrameUnloadRequests, groupCountAlignment);

  static_assert(sizeof(shaderio::StreamingGeometryPatch) <= sizeof(shaderio::StreamingPatch));

  m_shaderData                        = {};
  m_shaderData.patchGroupsCount       = loadRequests + unloadRequests;
  m_shaderData.patchUnloadGroupsCount = unloadRequests;
  if(config.allowBlasCaching)
  {


    uint32_t blasCount = std::min(geometryCount, config.maxPerFrameLoadRequests + config.maxPerFrameUnloadRequests);


    m_shaderData.patchCachedBlasCount = nvutils::align_up(blasCount, groupCountAlignment);
  }

  m_maxCachedBlasBuilds = config.maxPerFrameLoadRequests;

  uint32_t framePatchCount = m_shaderData.patchGroupsCount + m_shaderData.patchCachedBlasCount;

  m_unloadHandles = {};

  m_unloadHandles.resize(unloadRequests * STREAMING_MAX_ACTIVE_TASKS);

  std::vector<uint32_t> sharingQueueFamilies;
  if(config.useAsyncTransfer)
  {

    sharingQueueFamilies.push_back(res.m_queueStates.primary.m_familyIndex);

    sharingQueueFamilies.push_back(res.m_queueStates.transfer.m_familyIndex);
  }

  res.createBufferTyped(m_patchesBuffer, framePatchCount * STREAMING_MAX_ACTIVE_TASKS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, 0, sharingQueueFamilies);

  NVVK_DBG_NAME(m_patchesBuffer.buffer);

  res.createBufferTyped(m_patchesHostBuffer, framePatchCount * STREAMING_MAX_ACTIVE_TASKS, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VMA_MEMORY_USAGE_CPU_ONLY, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

  NVVK_DBG_NAME(m_patchesHostBuffer.buffer);

  m_shaderData.patches = m_patchesBuffer.address;

  for(uint32_t c = 0; c < STREAMING_MAX_ACTIVE_TASKS; c++)
  {
    StreamingUpdates::TaskInfo& task = m_taskInfos[c];
    task.unloadPatches               = m_patchesHostBuffer.data() + framePatchCount * c;
    task.loadPatches                 = task.unloadPatches + unloadRequests;
    task.geometryPatches =
        config.allowBlasCaching ? reinterpret_cast<shaderio::StreamingGeometryPatch*>(task.loadPatches + loadRequests) : nullptr;
    task.unloadHandles = m_unloadHandles.data() + unloadRequests * c;
  }
}


// 函数：StreamingUpdates::getOperationsSize。根据最新状态刷新缓存数据、GPU 地址、描述符或统计信息。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：更新函数负责把“旧状态”推进到“当前状态”，因此要避免部分更新造成 CPU/GPU 视图不一致。
size_t StreamingUpdates::getOperationsSize() const
{
  return m_patchesBuffer.bufferSize;
}


// 函数：StreamingUpdates::initClas。初始化本模块所需状态、资源或 GPU 侧绑定。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
void StreamingUpdates::initClas(Resources& res, const StreamingConfig& config, const SceneConfig& sceneConfig)
{


  uint32_t maxLoadClusters =

      nvutils::align_up(config.maxPerFrameLoadRequests * sceneConfig.clusterGroupSize, m_clusterCountAlignment);

  uint32_t maxClusters = nvutils::align_up(config.maxClusters, m_clusterCountAlignment);

  BufferRanges ranges = {};

  m_shaderData.newClasBuilds      = ranges.append(sizeof(shaderio::ClasBuildInfo) * maxLoadClusters, 16);
  m_shaderData.newClasAddresses   = ranges.append(sizeof(uint64_t) * maxLoadClusters, 8);
  m_shaderData.newClasSizes       = ranges.append(sizeof(uint32_t) * maxLoadClusters, 4);
  m_shaderData.newClasResidentIDs = ranges.append(sizeof(uint32_t) * maxLoadClusters, 4);

  uint32_t maxMovedClusters = config.usePersistentClasAllocator ? maxLoadClusters : maxClusters;

  m_shaderData.moveClasDstAddresses = ranges.append(sizeof(uint64_t) * maxMovedClusters, 8);
  m_shaderData.moveClasSrcAddresses = ranges.append(sizeof(uint64_t) * maxMovedClusters, 8);

  res.createBuffer(m_clasBuffer, ranges.getSize(),
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

  NVVK_DBG_NAME(m_clasBuffer.buffer);

  m_shaderData.newClasBuilds += m_clasBuffer.address;
  m_shaderData.newClasAddresses += m_clasBuffer.address;
  m_shaderData.newClasSizes += m_clasBuffer.address;
  m_shaderData.newClasResidentIDs += m_clasBuffer.address;

  m_shaderData.moveClasDstAddresses += m_clasBuffer.address;
  m_shaderData.moveClasSrcAddresses += m_clasBuffer.address;
}


// 函数：StreamingUpdates::getClasOperationsSize。根据最新状态刷新缓存数据、GPU 地址、描述符或统计信息。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：更新函数负责把“旧状态”推进到“当前状态”，因此要避免部分更新造成 CPU/GPU 视图不一致。
size_t StreamingUpdates::getClasOperationsSize() const
{
  return m_clasBuffer.bufferSize;
}


// 函数：StreamingUpdates::getMaxCachedBlasBuilds。根据最新状态刷新缓存数据、GPU 地址、描述符或统计信息。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：更新函数负责把“旧状态”推进到“当前状态”，因此要避免部分更新造成 CPU/GPU 视图不一致。
uint32_t StreamingUpdates::getMaxCachedBlasBuilds() const
{
  return m_shaderData.patchCachedBlasCount;
}


// 函数：StreamingUpdates::deinitClas。释放或回收前面初始化的资源，保持生命周期成对管理。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
void StreamingUpdates::deinitClas(Resources& res)
{

  res.m_allocator.destroyBuffer(m_clasBuffer);

  m_shaderData.newClasBuilds      = 0;
  m_shaderData.newClasAddresses   = 0;
  m_shaderData.newClasSizes       = 0;
  m_shaderData.newClasResidentIDs = 0;

  m_shaderData.moveClasDstAddresses = 0;
  m_shaderData.moveClasSrcAddresses = 0;
}


// 函数：StreamingUpdates::deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
void StreamingUpdates::deinit(Resources& res)
{

  deinitClas(res);

  res.m_allocator.destroyBuffer(m_patchesBuffer);

  res.m_allocator.destroyBuffer(m_patchesHostBuffer);
}


// 函数：StreamingUpdates::reset。根据最新状态刷新缓存数据、GPU 地址、描述符或统计信息。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：更新函数负责把“旧状态”推进到“当前状态”，因此要避免部分更新造成 CPU/GPU 视图不一致。
void StreamingUpdates::reset()
{
  m_pendingNew = {};
  memset(m_scheduledNew, 0, sizeof(m_scheduledNew));
  memset(m_scheduledNewFrame, 0, sizeof(m_scheduledNewFrame));
  m_scheduleIndex = 0;
}


// 函数：StreamingUpdates::getNewTask。根据最新状态刷新缓存数据、GPU 地址、描述符或统计信息。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：更新函数负责把“旧状态”推进到“当前状态”，因此要避免部分更新造成 CPU/GPU 视图不一致。
lodclusters::StreamingUpdates::TaskInfo& StreamingUpdates::getNewTask(uint32_t taskIndex)
{
  TaskInfo& task                   = m_taskInfos[taskIndex];
  task.loadCount                   = 0;
  task.unloadCount                 = 0;
  task.geometryCachedCount         = 0;
  task.geometryCachedClustersCount = 0;
  task.newClusterCount             = 0;
  task.loadActiveGroupsOffset      = ~0;
  task.loadActiveClustersOffset    = ~0;

  return task;
}


// 函数：StreamingUpdates::cmdUploadTask。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
size_t StreamingUpdates::cmdUploadTask(VkCommandBuffer cmd, uint32_t taskIndex)
{
  const TaskInfo& task = m_taskInfos[taskIndex];


  assert(task.loadActiveGroupsOffset != ~0);

  assert(task.loadActiveClustersOffset != ~0);

  size_t transferSize = 0;


  VkBufferCopy regions[3];
  uint32_t     regionCount = 0;

  uint32_t framePatchCount = m_shaderData.patchGroupsCount + m_shaderData.patchCachedBlasCount;

  regions[0].srcOffset = sizeof(shaderio::StreamingPatch) * framePatchCount * taskIndex;
  regions[1].srcOffset = sizeof(shaderio::StreamingPatch) * framePatchCount * taskIndex;
  regions[2].srcOffset = sizeof(shaderio::StreamingPatch) * framePatchCount * taskIndex;

  regions[0].dstOffset = sizeof(shaderio::StreamingPatch) * framePatchCount * taskIndex;

  if(task.unloadCount)
  {

    regions[regionCount].size = sizeof(shaderio::StreamingPatch) * (task.unloadCount);

    regions[regionCount].srcOffset += 0;


    regions[regionCount + 1].dstOffset = regions[regionCount].dstOffset + regions[regionCount].size;

    transferSize += regions[0].size;

    regionCount++;
  }

  if(task.loadCount)
  {

    regions[regionCount].size = sizeof(shaderio::StreamingPatch) * (task.loadCount);

    regions[regionCount].srcOffset += sizeof(shaderio::StreamingPatch) * m_shaderData.patchUnloadGroupsCount;


    regions[regionCount + 1].dstOffset = regions[regionCount].dstOffset + regions[regionCount].size;

    transferSize += regions[regionCount].size;

    regionCount++;
  }

  if(task.geometryCachedCount)
  {
    regions[regionCount].size = sizeof(shaderio::StreamingGeometryPatch) * (task.geometryCachedCount);

    regions[regionCount].srcOffset += sizeof(shaderio::StreamingPatch) * m_shaderData.patchGroupsCount;

    transferSize += regions[regionCount].size;

    regionCount++;
  }

  if(regionCount)
  {

    vkCmdCopyBuffer(cmd, m_patchesHostBuffer.buffer, m_patchesBuffer.buffer, regionCount, regions);
  }


  m_pendingNew.clusters += task.newClusterCount;
  m_pendingNew.groups += task.loadCount;

  return transferSize;
}


// 函数：StreamingUpdates::applyTask。根据最新状态刷新缓存数据、GPU 地址、描述符或统计信息。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：更新函数负责把“旧状态”推进到“当前状态”，因此要避免部分更新造成 CPU/GPU 视图不一致。
void StreamingUpdates::applyTask(shaderio::StreamingUpdate& shaderData, uint32_t taskIndex, uint32_t frameIndex)
{
  uint32_t framePatchCount = m_shaderData.patchGroupsCount + m_shaderData.patchCachedBlasCount;

  const TaskInfo& task = m_taskInfos[taskIndex];

  shaderData = m_shaderData;

  shaderData.patchGroupsCount         = task.loadCount + task.unloadCount;
  shaderData.patchUnloadGroupsCount   = task.unloadCount;
  shaderData.patchCachedBlasCount     = task.geometryCachedCount;
  shaderData.patchCachedClustersCount = task.geometryCachedClustersCount;
  shaderData.newClasCount             = task.newClusterCount;


  shaderData.patches += sizeof(shaderio::StreamingPatch) * framePatchCount * taskIndex;
  shaderData.geometryPatches = shaderData.patches + sizeof(shaderio::StreamingPatch) * shaderData.patchGroupsCount;
  shaderData.taskIndex       = taskIndex;
  shaderData.frameIndex      = frameIndex;
  shaderData.loadActiveGroupsOffset   = task.loadActiveGroupsOffset;
  shaderData.loadActiveClustersOffset = task.loadActiveClustersOffset;


  assert(m_pendingNew.clusters >= task.newClusterCount);

  assert(m_pendingNew.groups >= task.loadCount);

  m_pendingNew.clusters -= task.newClusterCount;
  m_pendingNew.groups -= task.loadCount;
  m_scheduledNewFrame[m_scheduleIndex % STREAMING_MAX_ACTIVE_TASKS]     = frameIndex;
  m_scheduledNew[m_scheduleIndex % STREAMING_MAX_ACTIVE_TASKS].clusters = task.newClusterCount;
  m_scheduledNew[m_scheduleIndex % STREAMING_MAX_ACTIVE_TASKS].groups   = task.loadCount;
  m_scheduleIndex++;
}


// 函数：StreamingStorage::init。初始化本模块所需状态、资源或 GPU 侧绑定。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
void StreamingStorage::init(Resources& res, const StreamingConfig& config)
{
  m_maxSceneBytes    = config.maxGeometryMegaBytes * 1024 * 1024;
  m_maxTransferBytes = config.maxTransferMegaBytes * 1024 * 1024;


  if(m_maxSceneBytes > 4 * 1024 * 1024 * 1024)
    m_blockBytes = std::min(size_t(256) * 1024 * 1024, m_maxSceneBytes);
  else if(m_maxSceneBytes > 1 * 1024 * 1024 * 1024)
    m_blockBytes = std::min(size_t(128) * 1024 * 1024, m_maxSceneBytes);
  else
    m_blockBytes = std::min(size_t(64) * 1024 * 1024, m_maxSceneBytes);

  m_dataQueueFamilies = {};
  if(config.useAsyncTransfer)
  {

    m_dataQueueFamilies.push_back(res.m_queueStates.primary.m_familyIndex);

    m_dataQueueFamilies.push_back(res.m_queueStates.transfer.m_familyIndex);

    m_taskCommandPool.init(res.m_device, res.m_queueStates.transfer.m_familyIndex, nvvk::ManagedCommandPools::Mode::EXPLICIT_INDEX,
                           VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, STREAMING_MAX_ACTIVE_TASKS);
  }

  res.createBuffer(m_transferHostBuffer, m_maxTransferBytes * STREAMING_MAX_ACTIVE_TASKS, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VMA_MEMORY_USAGE_CPU_ONLY, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

  NVVK_DBG_NAME(m_transferHostBuffer.buffer);

  m_dataInfo.blockSize         = m_blockBytes;
  m_dataInfo.memoryUsage       = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  m_dataInfo.maxAllocatedSize  = m_maxSceneBytes;
  m_dataInfo.queueFamilies     = m_dataQueueFamilies;
  m_dataInfo.resourceAllocator = &res.m_allocator;
  m_dataInfo.usageFlags =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;


  m_dataInfo.minAlignment = 16;


  m_dataAllocator.init(m_dataInfo);

  m_copyRegions = {};
  m_copyInfos   = {};


  m_copyRegions.reserve(std::min(config.maxGroups, 2048u));
  m_copyInfos.reserve(std::min(config.maxGroups, 2048u));
}


// 函数：StreamingStorage::deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
void StreamingStorage::deinit(Resources& res)
{

  res.m_allocator.destroyBuffer(m_transferHostBuffer);

  m_dataAllocator.deinit();

  m_taskCommandPool.deinit();

  m_copyInfos   = {};
  m_copyRegions = {};
}


// 函数：StreamingStorage::getOperationsSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
size_t StreamingStorage::getOperationsSize() const
{

  return 0;
}


// 函数：StreamingStorage::getMaxDataSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
size_t StreamingStorage::getMaxDataSize() const
{
  return (m_maxSceneBytes / m_blockBytes) * m_blockBytes;
}


// 函数：StreamingStorage::getNewTask。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
lodclusters::StreamingStorage::TaskInfo& StreamingStorage::getNewTask(uint32_t taskIndex)
{
  TaskInfo& task  = m_taskOperations[taskIndex];
  task.baseOffset = m_maxTransferBytes * taskIndex;
  task.usedMemory = 0;


  m_copyInfos.clear();

  m_copyRegions.clear();

  return task;
}


// 函数：StreamingStorage::canTransfer。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
bool StreamingStorage::canTransfer(const TaskInfo& task, size_t size) const
{
  return task.usedMemory + size <= m_maxTransferBytes;
}


// 函数：StreamingStorage::appendTransfer。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void* StreamingStorage::appendTransfer(TaskInfo& task, const nvvk::BufferSubAllocation& dstHandle)
{

  nvvk::BufferRange dstBinding = m_dataAllocator.subRange(dstHandle);


  assert(task.usedMemory + dstBinding.range <= m_maxTransferBytes);

  size_t transferOffset  = task.baseOffset;
  void*  transferPointer = reinterpret_cast<uint8_t*>(m_transferHostBuffer.mapping) + task.baseOffset;

  task.usedMemory += dstBinding.range;
  task.baseOffset += dstBinding.range;

  if(!m_copyInfos.empty() && m_copyInfos.back().targetBuffer == dstBinding.buffer)
  {

    VkBufferCopy& lastRegion = m_copyRegions.back();


    if(lastRegion.dstOffset + lastRegion.size == dstBinding.offset)
    {
      lastRegion.size += dstBinding.range;
      return transferPointer;
    }


  }
  else
  {

    CopyInfo task;
    task.targetBuffer = dstBinding.buffer;

    task.regionOffset = m_copyRegions.size();
    task.regionCount  = 0;

    m_copyInfos.push_back(task);
  }

  {

    VkBufferCopy region;
    region.srcOffset = transferOffset;
    region.dstOffset = dstBinding.offset;
    region.size      = dstBinding.range;

    m_copyInfos.back().regionCount++;

    m_copyRegions.push_back(region);
  }

  return transferPointer;
}


// 函数：StreamingStorage::cmdUploadTask。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
uint32_t StreamingStorage::cmdUploadTask(VkCommandBuffer cmd)
{
  for(auto it : m_copyInfos)
  {
    vkCmdCopyBuffer(cmd, m_transferHostBuffer.buffer, it.targetBuffer, uint32_t(it.regionCount), &m_copyRegions[it.regionOffset]);
  }

  return uint32_t(m_copyRegions.size());
}


// 函数：StreamingStorage::reset。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void StreamingStorage::reset()
{

  m_dataAllocator.deinit();
  NVVK_CHECK(m_dataAllocator.init(m_dataInfo));
}


// 函数：StreamingStorage::allocate。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
bool StreamingStorage::allocate(nvvk::BufferSubAllocation& handle, GeometryGroup group, size_t sz, uint64_t& deviceAddress)
{
  if(m_dataAllocator.subAllocate(handle, sz) == VK_SUCCESS)
  {
    deviceAddress = m_dataAllocator.subRange(handle).address;
    return true;
  }
  else
  {
    return false;
  }
}


// 函数：StreamingStorage::free。释放或回收前面初始化的资源，保持生命周期成对管理。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
void StreamingStorage::free(nvvk::BufferSubAllocation& handle)
{

  assert(handle);

  m_dataAllocator.subFree(handle);
}


// 函数：StreamingStorage::getStats。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void StreamingStorage::getStats(StreamingStats& stats) const
{

  nvvk::BufferSubAllocator::Report report = m_dataAllocator.getReport();
  stats.reservedDataBytes                 = report.freeSize + report.reservedSize;
  stats.usedDataBytes                     = report.requestedSize;
}

}
