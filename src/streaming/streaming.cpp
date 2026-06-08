//==============================================================================
// 文件：src/streaming/streaming.cpp
// 模块定位：SceneStreaming 主流程实现，处理请求完成、数据上传、地址更新、age filter、同步和计算 管线。
// 数据流：输入是 GPU 上一帧产生的请求和 CPU 已完成任务；输出是新驻留数据、卸载修补、更新后的 Geometry 地址表。
// 方法说明：该流程形成闭环：遍历产生需求，CPU 满足需求，GPU 地址表被修补，下一帧遍历基于新驻留集合继续决策。
// 正确性约束：异步 transfer 与 graphics 队列必须通过 时间线 semaphore 关联；失败或重复请求必须安全释放任务索引。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <volk.h>
#include <fmt/format.h>
#include "streaming.hpp"


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAMING_DEBUG_FORCE_REQUESTS 0


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace lodclusters {

template <class T>


// 结构：OffsetOrPointer。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct OffsetOrPointer
{
  union
  {
    uint64_t offset;
    T*       pointer;
  };
};


// 函数：SceneStreaming::init。初始化本模块所需状态、资源或 GPU 侧绑定。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
bool SceneStreaming::init(Resources* resources, const Scene* scene, const StreamingConfig& config)
{

  assert(!m_resources && "no init before deinit");

  assert(resources && scene);
  Resources& res = *resources;

  m_resources = resources;
  m_scene     = scene;
  m_config    = config;
  m_shaderData = {};
  m_shaders    = {};
  m_pipelines  = {};
  m_lastUpdateIndex         = 0;
  m_frameIndex              = 1;
  m_operationsSize          = 0;
  m_persistentGeometrySize  = 0;
  m_stats                   = {};
  m_config.maxGroups = std::max(m_config.maxGroups, uint32_t(scene->getActiveGeometryCount()));
  if(m_config.maxClusters == 0)
  {
    m_config.maxClusters = config.maxGroups * scene->m_config.clusterGroupSize;
  }
  m_config.maxClusters =
      std::max(m_config.maxClusters, uint32_t(scene->getActiveGeometryCount()) * scene->m_config.clusterGroupSize);

  m_stats.maxLoadCount     = m_config.maxPerFrameLoadRequests;
  m_stats.maxUnloadCount   = m_config.maxPerFrameUnloadRequests;
  m_stats.maxGroups        = m_config.maxGroups;
  m_stats.maxClusters      = m_config.maxClusters;
  m_stats.maxTransferBytes = m_config.maxTransferMegaBytes * 1024 * 1024;


  {
    nvvk::DescriptorBindings bindings;

    bindings.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

    bindings.addBinding(BINDINGS_READBACK_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

    bindings.addBinding(BINDINGS_GEOMETRIES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

    bindings.addBinding(BINDINGS_SCENEBUILDING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

    bindings.addBinding(BINDINGS_SCENEBUILDING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

    bindings.addBinding(BINDINGS_STREAMING_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

    bindings.addBinding(BINDINGS_STREAMING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

    m_dsetPack.init(bindings, res.m_device);

    nvvk::createPipelineLayout(res.m_device, &m_pipelineLayout, {m_dsetPack.getLayout()},
                               {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)}});
  }

  if(!initShadersAndPipelines())
  {

    m_dsetPack.deinit();
    return false;
  }


  uint32_t groupCountAlignment = std::max(std::max(STREAM_AGEFILTER_GROUPS_WORKGROUP, STREAM_UPDATE_SCENE_WORKGROUP),
                                          STREAM_COMPACTION_OLD_CLAS_WORKGROUP);

  uint32_t clusterCountAlignment = STREAM_COMPACTION_NEW_CLAS_WORKGROUP;


  m_requestsTaskQueue = {};
  m_updatesTaskQueue  = {};
  m_storageTaskQueue  = {};


  m_requests.init(res, m_config, groupCountAlignment, clusterCountAlignment);

  m_resident.init(res, m_config, groupCountAlignment, clusterCountAlignment);
  m_updates.init(res, m_config, uint32_t(m_scene->getActiveGeometryCount()), groupCountAlignment, clusterCountAlignment);

  m_storage.init(res, m_config);


  m_stats.maxDataBytes = m_storage.getMaxDataSize();

  m_operationsSize += logMemoryUsage(m_requests.getOperationsSize(), "operations", "stream requests");
  m_operationsSize += logMemoryUsage(m_resident.getOperationsSize(), "operations", "stream resident");
  m_operationsSize += logMemoryUsage(m_updates.getOperationsSize(), "operations", "stream updates");
  m_operationsSize += logMemoryUsage(m_storage.getOperationsSize(), "operations", "stream storage");

  res.createBuffer(m_shaderBuffer, sizeof(shaderio::SceneStreaming),
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                       | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

  NVVK_DBG_NAME(m_shaderBuffer.buffer);


  m_operationsSize += logMemoryUsage(m_shaderBuffer.bufferSize, "operations", "stream shaderio");


  initGeometries(res, scene);

  return true;
}


// 函数：SceneStreaming::updateBindings。根据最新状态刷新缓存数据、GPU 地址、描述符或统计信息。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：更新函数负责把“旧状态”推进到“当前状态”，因此要避免部分更新造成 CPU/GPU 视图不一致。
void SceneStreaming::updateBindings(const nvvk::Buffer& sceneBuildingBuffer)
{
  nvvk::WriteSetContainer writeSets;
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_FRAME_UBO), m_resources->m_commonBuffers.frameConstants);
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_READBACK_SSBO), m_resources->m_commonBuffers.readBack);
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_GEOMETRIES_SSBO), m_shaderGeometriesBuffer);
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_SCENEBUILDING_SSBO), sceneBuildingBuffer);
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_SCENEBUILDING_UBO), sceneBuildingBuffer);
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_STREAMING_SSBO), m_shaderBuffer);
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_STREAMING_UBO), m_shaderBuffer);
  vkUpdateDescriptorSets(m_resources->m_device, writeSets.size(), writeSets.data(), 0, nullptr);
}


// 函数：SceneStreaming::resetGeometryGroupAddresses。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void SceneStreaming::resetGeometryGroupAddresses(Resources::BatchedUploader& uploader)
{


  for(size_t geometryIndex = 0; geometryIndex < m_scene->getActiveGeometryCount(); geometryIndex++)
  {
    SceneStreaming::PersistentGeometry& persistentGeometry = m_persistentGeometries[geometryIndex];
    shaderio::Geometry&                 shaderGeometry     = m_shaderGeometries[geometryIndex];

    const Scene::GeometryView&          sceneGeometry      = m_scene->getActiveGeometry(geometryIndex);


    shaderio::LodLevel lastLodLevel = sceneGeometry.lodLevels.back();

    uint64_t* groupAddresses = uploader.uploadBuffer(persistentGeometry.groupAddresses, (uint64_t*)nullptr);
    for(uint32_t groupIndex = 0; groupIndex < lastLodLevel.groupOffset; groupIndex++)
    {
      groupAddresses[groupIndex] = STREAMING_INVALID_ADDRESS_START;
    }

    groupAddresses[lastLodLevel.groupOffset] = persistentGeometry.lowDetailGroupsData.address;


    uint32_t maxLodLevel = persistentGeometry.lodLevelsCount - 1;
    for(uint32_t i = 0; i < maxLodLevel; i++)
    {
      persistentGeometry.lodLoadedGroupsCount[i] = 0;
    }
    persistentGeometry.lodLoadedGroupsCount[maxLodLevel] = 1;
  }
}


// 函数：SceneStreaming::initGeometries。初始化本模块所需状态、资源或 GPU 侧绑定。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
void SceneStreaming::initGeometries(Resources& res, const Scene* scene)
{


  // 函数：uploader。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
  Resources::BatchedUploader uploader(res);

  m_shaderGeometries.resize(scene->getActiveGeometryCount());
  m_persistentGeometries.resize(scene->getActiveGeometryCount());

  uint32_t instancesOffset = 0;
  for(size_t geometryIndex = 0; geometryIndex < scene->getActiveGeometryCount(); geometryIndex++)
  {
    shaderio::Geometry&                 shaderGeometry     = m_shaderGeometries[geometryIndex];
    SceneStreaming::PersistentGeometry& persistentGeometry = m_persistentGeometries[geometryIndex];

    const Scene::GeometryView&          sceneGeometry      = m_scene->getActiveGeometry(geometryIndex);


    size_t numGroups = sceneGeometry.groupInfos.size();

    res.createBufferTyped(persistentGeometry.groupAddresses, numGroups, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    NVVK_DBG_NAME(persistentGeometry.groupAddresses.buffer);


    size_t numNodes = sceneGeometry.lodNodes.size();

    res.createBufferTyped(persistentGeometry.nodes, numNodes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    res.createBufferTyped(persistentGeometry.nodeBboxes, numNodes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    NVVK_DBG_NAME(persistentGeometry.nodes.buffer);

    NVVK_DBG_NAME(persistentGeometry.nodeBboxes.buffer);

    uint32_t numLodLevels = sceneGeometry.lodLevelsCount;

    res.createBufferTyped(persistentGeometry.lodLevels, numLodLevels, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    NVVK_DBG_NAME(persistentGeometry.lodLevels.buffer);

    m_persistentGeometrySize += persistentGeometry.groupAddresses.bufferSize;
    m_persistentGeometrySize += persistentGeometry.nodes.bufferSize;
    m_persistentGeometrySize += persistentGeometry.nodeBboxes.bufferSize;


    shaderGeometry                         = {};
    shaderGeometry.bbox                    = sceneGeometry.bbox;
    shaderGeometry.nodes                   = persistentGeometry.nodes.address;
    shaderGeometry.nodeBboxes              = persistentGeometry.nodeBboxes.address;
    shaderGeometry.streamingGroupAddresses = persistentGeometry.groupAddresses.address;
    shaderGeometry.lodLevelsCount          = numLodLevels;
    shaderGeometry.lodLevels               = persistentGeometry.lodLevels.address;

    shaderGeometry.instancesCount          = sceneGeometry.instanceReferenceCount * scene->getGeometryInstanceFactor();
    shaderGeometry.instancesOffset         = instancesOffset;

    instancesOffset += shaderGeometry.instancesCount;

    persistentGeometry.lodLevelsCount = numLodLevels;
    for(uint32_t i = 0; i < numLodLevels; i++)
    {
      persistentGeometry.lodGroupsCount[i] = sceneGeometry.lodLevels[i].groupCount;
    }


    uploader.uploadBuffer(persistentGeometry.nodes, sceneGeometry.lodNodes.data());
    uploader.uploadBuffer(persistentGeometry.nodeBboxes, sceneGeometry.lodNodeBboxes.data());
    uploader.uploadBuffer(persistentGeometry.lodLevels, sceneGeometry.lodLevels.data());


    shaderio::LodLevel     lastLodLevel = sceneGeometry.lodLevels.back();
    const Scene::GroupInfo groupInfo    = sceneGeometry.groupInfos[lastLodLevel.groupOffset];


    // 函数：groupView。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    Scene::GroupView       groupView(sceneGeometry.groupData, groupInfo);

    assert(groupInfo.clusterCount == 1);

    GeometryGroup geometryGroup     = {uint32_t(geometryIndex), lastLodLevel.groupOffset};
    uint32_t      lastClustersCount = groupInfo.clusterCount;

    uint64_t      lastGroupSize     = groupInfo.getDeviceSize();


    res.createBuffer(persistentGeometry.lowDetailGroupsData, lastGroupSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    NVVK_DBG_NAME(persistentGeometry.lowDetailGroupsData.buffer);
    m_persistentGeometrySize += persistentGeometry.lowDetailGroupsData.bufferSize;


    assert(lastClustersCount <= 0xFFFFFFFF);
    assert(m_resident.canAllocateGroup(uint32_t(lastClustersCount)));


    StreamingResident::Group* rgroup = m_resident.addGroup(geometryGroup, lastClustersCount);
    rgroup->deviceAddress            = persistentGeometry.lowDetailGroupsData.address;
    rgroup->lodLevel                 = groupInfo.lodLevel;

    persistentGeometry.lodLoadedGroupsCount[groupInfo.lodLevel] = 1;


    void* loGroupData = uploader.uploadBuffer(persistentGeometry.lowDetailGroupsData, (void*)nullptr);

    Scene::fillGroupRuntimeData(groupInfo, groupView, geometryGroup.groupID, rgroup->groupResidentID,
                                rgroup->clusterResidentID, loGroupData, persistentGeometry.lowDetailGroupsData.bufferSize);

    shaderGeometry.lowDetailClusterID = rgroup->clusterResidentID;
    shaderGeometry.lowDetailTriangles = groupInfo.triangleCount;
  }


  resetGeometryGroupAddresses(uploader);

  res.createBufferTyped(m_shaderGeometriesBuffer, scene->getActiveGeometryCount(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  NVVK_DBG_NAME(m_shaderGeometriesBuffer.buffer);

  m_operationsSize += logMemoryUsage(m_shaderGeometriesBuffer.bufferSize, "operations", "stream geo buffer");

  uploader.uploadBuffer(m_shaderGeometriesBuffer, m_shaderGeometries.data());


  m_resident.uploadInitialState(uploader, m_shaderData.resident);


  uploader.flush();
}

void SceneStreaming::cmdBeginFrame(VkCommandBuffer         cmd,
                                   QueueState&             cmdQueueState,
                                   QueueState&             asyncQueueState,
                                   const FrameSettings&    settings,
                                   nvvk::ProfilerGpuTimer& profiler)
{


  auto     timerSection = profiler.cmdFrameSection(cmd, "Stream Begin");
  VkDevice device       = m_resources->m_device;


  const bool ensureAcquisition = true;


  uint32_t updateTaskCount = 0;
  while(m_updatesTaskQueue.canPop(device, ensureAcquisition))
  {


    uint32_t popUpdateIndex = m_updatesTaskQueue.pop();


    const StreamingUpdates::TaskInfo& update = m_updates.getCompletedTask(popUpdateIndex);
    for(uint32_t g = 0; g < update.unloadCount; g++)
    {

      m_storage.free(update.unloadHandles[g]);
    }


    m_updatesTaskQueue.releaseTaskIndex(popUpdateIndex);


    if(++updateTaskCount >= 16)
    {
      updateTaskCount = 0;

      break;
    }
  }


  uint32_t pushUpdateIndex = INVALID_TASK_INDEX;


  if(m_storageTaskQueue.canPop(device, ensureAcquisition))
  {


    uint32_t dependentIndex  = INVALID_TASK_INDEX;

    uint32_t popStorageIndex = m_storageTaskQueue.popWithDependent(dependentIndex);

    m_storageTaskQueue.releaseTaskIndex(popStorageIndex);


    if(dependentIndex != INVALID_TASK_INDEX)
    {
      pushUpdateIndex = dependentIndex;
    }
  }

  bool isImmediateUpdate = false;


  if(m_requestsTaskQueue.canPop(device, ensureAcquisition))
  {

    uint32_t popRequestIndex = m_requestsTaskQueue.pop();

#if 1


    while(m_requestsTaskQueue.canPop(device, false))
    {


      m_requestsTaskQueue.releaseTaskIndex(popRequestIndex);


      popRequestIndex = m_requestsTaskQueue.pop();
    }
#endif


    uint32_t dependentIndex = handleCompletedRequest(cmd, cmdQueueState, asyncQueueState, settings, popRequestIndex);

    if(dependentIndex != INVALID_TASK_INDEX)
    {


      assert(pushUpdateIndex == INVALID_TASK_INDEX);

      pushUpdateIndex   = dependentIndex;
      isImmediateUpdate = true;
    }
  }


  if(pushUpdateIndex != INVALID_TASK_INDEX)
  {


    m_resident.applyTask(m_shaderData.resident, pushUpdateIndex, m_frameIndex);

    m_updates.applyTask(m_shaderData.update, pushUpdateIndex, m_frameIndex);


    m_updatesTaskQueue.push(pushUpdateIndex, cmdQueueState.getCurrentState());

    m_lastUpdateIndex = pushUpdateIndex;
  }
  else
  {

    m_shaderData.update.patchGroupsCount         = 0;
    m_shaderData.update.patchUnloadGroupsCount   = 0;
    m_shaderData.update.patchCachedBlasCount     = 0;
    m_shaderData.update.patchCachedClustersCount = 0;
    m_shaderData.update.loadActiveGroupsOffset   = 0;
    m_shaderData.update.loadActiveClustersOffset = 0;
    m_shaderData.update.newClasCount             = 0;
    m_shaderData.update.taskIndex                = INVALID_TASK_INDEX;
    m_shaderData.update.frameIndex               = m_frameIndex;
  }


  {


    uint32_t pushRequestIndex = m_requestsTaskQueue.acquireTaskIndex();


    assert(pushRequestIndex != INVALID_TASK_INDEX);


    m_requests.applyTask(m_shaderData.request, pushRequestIndex, m_frameIndex);
  }

  m_shaderData.frameIndex               = m_frameIndex;
  m_shaderData.ageThreshold             = settings.ageThreshold;


  vkCmdUpdateBuffer(cmd, m_shaderBuffer.buffer, 0, sizeof(m_shaderData), &m_shaderData);
}

uint32_t SceneStreaming::handleCompletedRequest(VkCommandBuffer      cmd,
                                                QueueState&          cmdQueueState,
                                                QueueState&          asyncQueueState,
                                                const FrameSettings& settings,
                                                uint32_t             popRequestIndex)
{


  const StreamingRequests::TaskInfo& request = m_requests.getCompletedTask(popRequestIndex);


  uint32_t loadCount   = std::min(request.shaderData->maxLoads, request.shaderData->loadCounter);

  uint32_t unloadCount = std::min(request.shaderData->maxUnloads, request.shaderData->unloadCounter);


#if !STREAMING_DEBUG_FORCE_REQUESTS
  if((!loadCount && !unloadCount) || !m_debugFrameLimit)
  {


    m_requestsTaskQueue.releaseTaskIndex(popRequestIndex);
    return INVALID_TASK_INDEX;
  }
#endif


  if(m_debugFrameLimit > 0)
    m_debugFrameLimit--;


  uint32_t pushStorageIndex = m_storageTaskQueue.acquireTaskIndex();

  uint32_t pushUpdateIndex  = m_updatesTaskQueue.acquireTaskIndex();


  if(pushStorageIndex == INVALID_TASK_INDEX || pushUpdateIndex == INVALID_TASK_INDEX)
  {

    if(pushStorageIndex != INVALID_TASK_INDEX)
    {

      m_storageTaskQueue.releaseTaskIndex(pushStorageIndex);
    }
    if(pushUpdateIndex != INVALID_TASK_INDEX)
    {

      m_updatesTaskQueue.releaseTaskIndex(pushUpdateIndex);
    }

    m_requestsTaskQueue.releaseTaskIndex(popRequestIndex);
    return INVALID_TASK_INDEX;
  }


  StreamingStorage::TaskInfo& storageTask = m_storage.getNewTask(pushStorageIndex);

  StreamingUpdates::TaskInfo& updateTask  = m_updates.getNewTask(pushUpdateIndex);


  for(uint32_t g = 0; g < unloadCount; g++)
  {
    GeometryGroup geometryGroup = request.unloadGeometryGroups[g];

    assert(geometryGroup.geometryID < m_scene->getActiveGeometryCount());
    assert(geometryGroup.groupID < m_scene->getActiveGeometry(geometryGroup.geometryID).totalClustersCount);


    const StreamingResident::Group* group = m_resident.findGroup(geometryGroup);
    if(!group)
    {


      continue;
    }


    uint32_t                  unloadIndex = updateTask.unloadCount++;
    shaderio::StreamingPatch& patch       = updateTask.unloadPatches[unloadIndex];
    patch.geometryID                      = geometryGroup.geometryID;
    patch.groupID                         = geometryGroup.groupID;
    patch.groupAddress                    = STREAMING_INVALID_ADDRESS_START;


    assert(group->storageHandle);
    updateTask.unloadHandles[unloadIndex] = group->storageHandle;


    assert(m_persistentGeometries[geometryGroup.geometryID].lodLoadedGroupsCount[group->lodLevel] > 0);
    m_persistentGeometries[geometryGroup.geometryID].lodLoadedGroupsCount[group->lodLevel]--;


    m_resident.removeGroup(group->groupResidentID);


  }


  updateTask.loadActiveGroupsOffset   = m_resident.getLoadActiveGroupsOffset();

  updateTask.loadActiveClustersOffset = m_resident.getLoadActiveClustersOffset();

  uint64_t transferBytes = 0;

  m_stats.couldNotAllocateClas  = 0;
  m_stats.couldNotTransfer      = 0;
  m_stats.couldNotAllocateGroup = 0;
  m_stats.couldNotStore         = 0;
  m_stats.uncompletedLoadCount  = 0;


  uint32_t processedLoads = 0;
  for(uint32_t g = 0; g < loadCount; g++)
  {
    GeometryGroup geometryGroup = request.loadGeometryGroups[g];

    assert(geometryGroup.geometryID < m_scene->getActiveGeometryCount());
    assert(geometryGroup.groupID < m_scene->getActiveGeometry(geometryGroup.geometryID).totalClustersCount);

    if(m_resident.findGroup(geometryGroup))
    {


      continue;
    }


    const Scene::GeometryView& sceneGeometry = m_scene->getActiveGeometry(geometryGroup.geometryID);


    const Scene::GroupInfo groupInfo       = sceneGeometry.groupInfos[geometryGroup.groupID];
    uint32_t               clusterCount    = groupInfo.clusterCount;

    uint64_t               groupDeviceSize = groupInfo.getDeviceSize();

    uint64_t                  deviceAddress;
    nvvk::BufferSubAllocation storageHandle;


    if(!m_storage.canTransfer(storageTask, groupDeviceSize))
    {
      m_stats.couldNotTransfer++;
      m_stats.uncompletedLoadCount++;
      continue;
    }


    bool canStore         = m_storage.allocate(storageHandle, geometryGroup, groupDeviceSize, deviceAddress);

    bool canAllocateGroup = m_resident.canAllocateGroup(clusterCount);


    if(!canStore || !canAllocateGroup)
    {
      m_stats.couldNotAllocateGroup += (!canAllocateGroup);
      m_stats.couldNotStore += (!canStore);

      if(canStore)
      {


        m_storage.free(storageHandle);
      }

      if(clusterCount < 8)
      {
        m_stats.uncompletedLoadCount += loadCount - g;
        break;
      }
      else
      {
        m_stats.uncompletedLoadCount++;
        continue;
      }
    }

    processedLoads++;


    StreamingResident::Group* residentGroup = m_resident.addGroup(geometryGroup, clusterCount);
    residentGroup->storageHandle            = storageHandle;
    residentGroup->deviceAddress            = deviceAddress;
    residentGroup->lodLevel                 = groupInfo.lodLevel;

    void* groupData                         = m_storage.appendTransfer(storageTask, residentGroup->storageHandle);


    assert(deviceAddress % 16 == 0);

    {


      // 函数：groupView。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
      // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
      // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
      Scene::GroupView groupView(sceneGeometry.groupData, groupInfo);
      if(groupInfo.uncompressedSizeBytes)
      {

        Scene::decompressGroup(groupInfo, groupView, groupData, groupDeviceSize);
      }
      else
      {


        memcpy(groupData, groupView.raw, groupView.rawSize);
      }
    }

    m_persistentGeometries[geometryGroup.geometryID].lodLoadedGroupsCount[groupInfo.lodLevel]++;


    shaderio::StreamingPatch& patch = updateTask.loadPatches[updateTask.loadCount++];
    patch.geometryID                = geometryGroup.geometryID;
    patch.groupID                   = geometryGroup.groupID;
    patch.groupAddress              = deviceAddress;
    patch.groupResidentID           = residentGroup->groupResidentID;
    patch.clusterResidentID         = residentGroup->clusterResidentID;
    patch.clusterCount              = groupInfo.clusterCount;
    patch.lodLevel                  = groupInfo.lodLevel;


    transferBytes += groupInfo.sizeBytes;
  }

#if !STREAMING_DEBUG_FORCE_REQUESTS
  if(updateTask.loadCount == 0 && updateTask.unloadCount == 0)
  {


    m_requestsTaskQueue.releaseTaskIndex(popRequestIndex);

    m_updatesTaskQueue.releaseTaskIndex(pushUpdateIndex);

    m_storageTaskQueue.releaseTaskIndex(pushStorageIndex);
    return INVALID_TASK_INDEX;
  }
#endif

  if(m_config.useAsyncTransfer)
  {


    NVVK_CHECK(m_storage.m_taskCommandPool.acquireCommandBuffer(pushStorageIndex, cmd));
    VkCommandBufferBeginInfo cmdInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
    };


    vkBeginCommandBuffer(cmd, &cmdInfo);
  }

  uint32_t transferCount = 0;


  transferBytes += m_updates.cmdUploadTask(cmd, pushUpdateIndex);

  transferBytes += m_resident.cmdUploadTask(cmd, pushUpdateIndex);

  transferCount += m_storage.cmdUploadTask(cmd);
  transferCount += 2;


  if(processedLoads > 0) {

    m_stats.transferBytes = transferBytes;
    m_stats.transferCount = transferCount;
    m_stats.loadCount = updateTask.loadCount;
  }
  if(updateTask.unloadCount > 0) {
    m_stats.unloadCount = updateTask.unloadCount;
  }


  bool useDecoupledUpdate = m_config.useAsyncTransfer && m_config.useDecoupledAsyncTransfer;

  nvvk::SemaphoreState storageSemaphoreState =

      m_config.useAsyncTransfer ? asyncQueueState.getCurrentState() : cmdQueueState.getCurrentState();

  if(m_config.useAsyncTransfer)
  {

    vkEndCommandBuffer(cmd);

    if(!m_config.useDecoupledAsyncTransfer)
    {


      VkSemaphoreSubmitInfo semWaitInfo = asyncQueueState.getWaitSubmit(VK_PIPELINE_STAGE_2_TRANSFER_BIT);


      cmdQueueState.m_pendingWaits.push_back(semWaitInfo);
    }


    VkSemaphoreSubmitInfo     semSubmitInfo = asyncQueueState.advanceSignalSubmit(VK_PIPELINE_STAGE_2_TRANSFER_BIT);
    VkCommandBufferSubmitInfo cmdBufInfo    = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmdBufInfo.commandBuffer                = cmd;


    VkSubmitInfo2 submits            = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR};
    submits.pCommandBufferInfos      = &cmdBufInfo;
    submits.commandBufferInfoCount   = 1;
    submits.pSignalSemaphoreInfos    = &semSubmitInfo;
    submits.signalSemaphoreInfoCount = 1;


    vkQueueSubmit2(asyncQueueState.m_queue, 1, &submits, nullptr);
  }


  m_requestsTaskQueue.releaseTaskIndex(popRequestIndex);


  m_storageTaskQueue.push(pushStorageIndex, storageSemaphoreState, useDecoupledUpdate ? pushUpdateIndex : INVALID_TASK_INDEX);


  return useDecoupledUpdate ? INVALID_TASK_INDEX : pushUpdateIndex;
}


// 函数：getWorkGroupCount。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
static uint32_t getWorkGroupCount(uint32_t numThreads, uint32_t workGroupSize)
{

  return (numThreads + workGroupSize - 1) / workGroupSize;
}


// 函数：SceneStreaming::cmdPreTraversal。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
void SceneStreaming::cmdPreTraversal(VkCommandBuffer cmd, VkDeviceAddress clasScratchBuffer, nvvk::ProfilerGpuTimer& profiler)
{
  Resources& res = *m_resources;


  auto timerSection = profiler.cmdFrameSection(cmd, "Stream Pre Traversal");


  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);


  if(m_shaderData.update.patchGroupsCount)
  {

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeUpdateSceneRaster);

    res.cmdLinearDispatch(cmd, getWorkGroupCount(m_shaderData.update.patchGroupsCount, STREAM_UPDATE_SCENE_WORKGROUP));


    m_resident.cmdRunTask(cmd, m_shaderData.update.taskIndex);
  }


}


// 函数：SceneStreaming::cmdPostTraversal。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
void SceneStreaming::cmdPostTraversal(VkCommandBuffer cmd, VkDeviceAddress clasScratchBuffer, bool runAgeFilter, nvvk::ProfilerGpuTimer& profiler)
{
  Resources& res = *m_resources;


  auto timerSection = profiler.cmdFrameSection(cmd, "Stream Post Traversal");

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);

  if(m_shaderData.resident.activeGroupsCount && runAgeFilter)
  {


    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeAgeFilterGroups);
    res.cmdLinearDispatch(cmd, getWorkGroupCount(m_shaderData.resident.activeGroupsCount, STREAM_AGEFILTER_GROUPS_WORKGROUP));
  }


}


// 函数：SceneStreaming::cmdEndFrame。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
void SceneStreaming::cmdEndFrame(VkCommandBuffer cmd, QueueState& cmdQueueState, nvvk::ProfilerGpuTimer& profiler)
{


  auto timerSection = profiler.cmdFrameSection(cmd, "Stream End");

  m_requests.cmdRunTask(cmd, m_shaderData.request, m_shaderBuffer.buffer, offsetof(shaderio::SceneStreaming, request));

  m_requestsTaskQueue.push(m_shaderData.request.taskIndex, cmdQueueState.getCurrentState());

  m_frameIndex++;


  size_t geoSize = getGeometrySize(false);
  if(geoSize > m_peakGeometrySize)
  {
    m_peakGeometrySize = geoSize;
    m_peakFrameIndex   = m_frameIndex;
  }
  else if(m_frameIndex == m_peakFrameIndex + 2)
  {

    LOGI("streaming: geometry peak frame %d\n", m_peakFrameIndex);
  }
}


// 函数：SceneStreaming::getStats。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void SceneStreaming::getStats(StreamingStats& stats) const
{
  stats = m_stats;


  m_storage.getStats(stats);

  m_resident.getStats(stats);
  stats.persistentDataBytes = m_persistentGeometrySize;
}


// 函数：SceneStreaming::getGeometrySize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
size_t SceneStreaming::getGeometrySize(bool reserved) const
{
  StreamingStats stats;

  getStats(stats);

  if(reserved)
  {
    return m_persistentGeometrySize + stats.reservedDataBytes;
  }
  else
  {
    return m_persistentGeometrySize + stats.usedDataBytes;
  }
}


// 函数：SceneStreaming::deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
void SceneStreaming::deinit()
{
  if(!m_resources)
    return;

  Resources& res = *m_resources;


  deinitShadersAndPipelines();

  m_dsetPack.deinit();

  vkDestroyPipelineLayout(res.m_device, m_pipelineLayout, nullptr);


  m_resident.deinit(res);

  m_storage.deinit(res);

  m_updates.deinit(res);

  m_requests.deinit(res);

  for(auto& it : m_persistentGeometries)
  {

    res.m_allocator.destroyBuffer(it.groupAddresses);

    res.m_allocator.destroyBuffer(it.nodeBboxes);

    res.m_allocator.destroyBuffer(it.nodes);

    res.m_allocator.destroyBuffer(it.lodLevels);

    res.m_allocator.destroyBuffer(it.lowDetailGroupsData);
  }

  m_persistentGeometries.clear();


  res.m_allocator.destroyBuffer(m_shaderGeometriesBuffer);

  res.m_allocator.destroyBuffer(m_shaderBuffer);

  m_resources = nullptr;
  m_scene     = nullptr;
}


// 函数：SceneStreaming::reset。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void SceneStreaming::reset()
{
  Resources& res = *m_resources;


  vkDeviceWaitIdle(res.m_device);

  m_debugFrameLimit  = s_defaultDebugFrameLimit;
  m_peakFrameIndex   = ~0;
  m_peakGeometrySize = 0;

  m_requestsTaskQueue = {};
  m_storageTaskQueue  = {};
  m_updatesTaskQueue  = {};


  m_resident.reset(m_shaderData.resident);

  m_updates.reset();


  m_storage.reset();


  m_frameIndex = 1;


  // 函数：uploader。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
  Resources::BatchedUploader uploader(res);

  resetGeometryGroupAddresses(uploader);

  uploader.flush();
}


// 函数：SceneStreaming::initShadersAndPipelines。初始化本模块所需状态、资源或 GPU 侧绑定。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
bool SceneStreaming::initShadersAndPipelines()
{
  Resources& res = *m_resources;


  shaderc::CompileOptions options = res.makeCompilerOptions();
  options.AddMacroDefinition("SUBGROUP_SIZE", fmt::format("{}", res.m_physicalDeviceInfo.properties11.subgroupSize));
  options.AddMacroDefinition("USE_16BIT_DISPATCH", fmt::format("{}", res.m_use16bitDispatch ? 1 : 0));

  shaderc::CompileOptions optionsRaster = options;

  optionsRaster.AddMacroDefinition("TARGETS_RASTERIZATION", "1");


  res.compileShader(m_shaders.computeAgeFilterGroups, VK_SHADER_STAGE_COMPUTE_BIT, "streaming/stream_agefilter_groups.comp.glsl", &options);

  res.compileShader(m_shaders.computeSetup, VK_SHADER_STAGE_COMPUTE_BIT, "streaming/stream_setup.comp.glsl", &options);

  res.compileShader(m_shaders.computeUpdateSceneRaster, VK_SHADER_STAGE_COMPUTE_BIT, "streaming/stream_update_scene.comp.glsl", &optionsRaster);

  if(!res.verifyShaders(m_shaders))
  {
    return false;
  }

  {
    VkComputePipelineCreateInfo compInfo   = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkShaderModuleCreateInfo    shaderInfo = {};
    compInfo.stage                         = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    compInfo.stage.stage                   = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.pName                   = "main";
    compInfo.stage.pNext                   = &shaderInfo;
    compInfo.layout                        = m_pipelineLayout;


    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeAgeFilterGroups);

    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeAgeFilterGroups);


    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeSetup);

    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeSetup);


    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeUpdateSceneRaster);

    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeUpdateSceneRaster);
  }

  return true;
}


// 函数：SceneStreaming::deinitShadersAndPipelines。释放或回收前面初始化的资源，保持生命周期成对管理。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
void SceneStreaming::deinitShadersAndPipelines()
{
  Resources& res = *m_resources;


  res.destroyPipelines(m_pipelines);
}


}
