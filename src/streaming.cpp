//流式传输
#include <volk.h>
#include <fmt/format.h>
#include "streaming.hpp"
#define STREAMING_DEBUG_FORCE_REQUESTS 0
namespace lodclusters {

template <class T>
struct OffsetOrPointer
{
  union
  {
    uint64_t offset;
    T*       pointer;
  };
};

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
  m_frameIndex              = 1;  // intentionally start at 1
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

  // setup descriptor set container
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

  // 优化：使用更新后的工作组大小，提高并行处理效率
  uint32_t groupCountAlignment = std::max(std::max(STREAM_AGEFILTER_GROUPS_WORKGROUP, STREAM_UPDATE_SCENE_WORKGROUP),
                                          STREAM_COMPACTION_OLD_CLAS_WORKGROUP);

  uint32_t clusterCountAlignment = STREAM_COMPACTION_NEW_CLAS_WORKGROUP;

  // setup streaming management
  m_requestsTaskQueue = {};
  m_updatesTaskQueue  = {};
  m_storageTaskQueue  = {};

  m_requests.init(res, m_config, groupCountAlignment, clusterCountAlignment);
  m_resident.init(res, m_config, groupCountAlignment, clusterCountAlignment);
  m_updates.init(res, m_config, uint32_t(m_scene->getActiveGeometryCount()), groupCountAlignment, clusterCountAlignment);
  m_storage.init(res, m_config);

  // storage uses block allocator, max may be less than what we asked for
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

  // seed lo res geometry
  initGeometries(res, scene);

  return true;
}

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





void SceneStreaming::resetGeometryGroupAddresses(Resources::BatchedUploader& uploader)
{
  // this function fills the geometry group addresses to be invalid
  // except for the persistent lowest detail group

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
    // except last group, which is always loaded
    groupAddresses[lastLodLevel.groupOffset] = persistentGeometry.lowDetailGroupsData.address;

    // also reset the number of groups loaded per lod-level, except last which is also always loaded.
    uint32_t maxLodLevel = persistentGeometry.lodLevelsCount - 1;
    for(uint32_t i = 0; i < maxLodLevel; i++)
    {
      persistentGeometry.lodLoadedGroupsCount[i] = 0;
    }
    persistentGeometry.lodLoadedGroupsCount[maxLodLevel] = 1;
  }
}

void SceneStreaming::initGeometries(Resources& res, const Scene* scene)
{
  // This function uploads all persistent per-geometry data.
  // - hierarchy nodes for lod traversal
  // - lowest detail geometry group & clusters
  // - the address lookup array to find resident groups
  // It also fills the geometry descriptor stored in
  // m_shaderGeometries

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

    // setup shaderio
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

    // basic uploads

    uploader.uploadBuffer(persistentGeometry.nodes, sceneGeometry.lodNodes.data());
    uploader.uploadBuffer(persistentGeometry.nodeBboxes, sceneGeometry.lodNodeBboxes.data());
    uploader.uploadBuffer(persistentGeometry.lodLevels, sceneGeometry.lodLevels.data());

    // seed lowest detail group, which must have just a single cluster
    shaderio::LodLevel     lastLodLevel = sceneGeometry.lodLevels.back();
    const Scene::GroupInfo groupInfo    = sceneGeometry.groupInfos[lastLodLevel.groupOffset];
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

    // setup and upload geometry data for the lowest detail group
    void* loGroupData = uploader.uploadBuffer(persistentGeometry.lowDetailGroupsData, (void*)nullptr);

    Scene::fillGroupRuntimeData(groupInfo, groupView, geometryGroup.groupID, rgroup->groupResidentID,
                                rgroup->clusterResidentID, loGroupData, persistentGeometry.lowDetailGroupsData.bufferSize);

    shaderGeometry.lowDetailClusterID = rgroup->clusterResidentID;
    shaderGeometry.lowDetailTriangles = groupInfo.triangleCount;
  }

  // this will set all addresses to invalid, except lowest detail geometry group, which is persistently loaded.
  resetGeometryGroupAddresses(uploader);

  res.createBufferTyped(m_shaderGeometriesBuffer, scene->getActiveGeometryCount(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  NVVK_DBG_NAME(m_shaderGeometriesBuffer.buffer);
  m_operationsSize += logMemoryUsage(m_shaderGeometriesBuffer.bufferSize, "operations", "stream geo buffer");

  uploader.uploadBuffer(m_shaderGeometriesBuffer, m_shaderGeometries.data());

  // initial residency table
  m_resident.uploadInitialState(uploader, m_shaderData.resident);

  uploader.flush();
}

void SceneStreaming::cmdBeginFrame(VkCommandBuffer         cmd,
                                   QueueState&             cmdQueueState,
                                   QueueState&             asyncQueueState,
                                   const FrameSettings&    settings,
                                   nvvk::ProfilerGpuTimer& profiler)
{
  // This function sets up all relevant streaming tasks for the frame
  // and configures the content of `m_shaderData` which is uploaded
  // and all streaming related kernels will operate with.
  //
  // The data within `m_shaderData` is stateful and new operations may
  // modify it permanently so that future frames keep the state from the
  // last run operations.
  //
  // - handle completed updates: to give back memory unloads within that update
  // - handle completed storage transfers: to trigger scene updates now that geometry data is available
  // - handle completed request: to trigger new loads/unloads etc.
  //   as a request produces one new update & storage task, these tasks must be handled before
  // - make a new request
  //   likewise a new request requires an empty slot, hence requests must be handled before
  //
  // This function is called by the renderer.

  auto     timerSection = profiler.cmdFrameSection(cmd, "Stream Begin");
  VkDevice device       = m_resources->m_device;

  // For each task queue we must ensure that we have one new task index
  // available to acquire for any potential new work in this frame.
  // The ordering in which we drain them matters, as was described above.
  const bool ensureAcquisition = true;

  // pop all completed old updates to recycle as much memory as we can
  // 优化：批量处理完成的更新任务，减少循环开销
  uint32_t updateTaskCount = 0;
  while(m_updatesTaskQueue.canPop(device, ensureAcquisition))
  {
    // handleCompletedUpdate
    //
    // The update operation has been completed on the GPU time line, therefore
    // it is safe to fully recycle the memory as it can no longer be reached.
    uint32_t popUpdateIndex = m_updatesTaskQueue.pop();

    const StreamingUpdates::TaskInfo& update = m_updates.getCompletedTask(popUpdateIndex);
    for(uint32_t g = 0; g < update.unloadCount; g++)
    {
      m_storage.free(update.unloadHandles[g]);
    }

    m_updatesTaskQueue.releaseTaskIndex(popUpdateIndex);
    
    // 每处理一定数量的任务后，检查一次是否需要继续，避免过长的循环
    if(++updateTaskCount >= 16)
    {
      updateTaskCount = 0;
      // 短暂休息，允许其他操作进行
      break;
    }
  }

  // Our task system allows that new updates can be either
  // handled immediately in the current frame, or decoupled
  // in a later frame.
  //
  // Decoupled allows for asynchronous uploads that can
  // span multiple frames, while immediate means
  // we guarantee transfers completed prior triggering
  // operations.
  uint32_t pushUpdateIndex = INVALID_TASK_INDEX;

  // pop one completed storage transfer
  if(m_storageTaskQueue.canPop(device, ensureAcquisition))
  {
    // handleCompletedStorage
    //
    // The upload of new data was completed, recycle the task and transfer space for future use.
    // If we run in decoupled mode, then push the dependent updates with this frame.
    uint32_t dependentIndex  = INVALID_TASK_INDEX;
    uint32_t popStorageIndex = m_storageTaskQueue.popWithDependent(dependentIndex);
    m_storageTaskQueue.releaseTaskIndex(popStorageIndex);

    // check if we use a decoupled update
    if(dependentIndex != INVALID_TASK_INDEX)
    {
      pushUpdateIndex = dependentIndex;
    }
  }

  bool isImmediateUpdate = false;

  // pop and process one completed request:
  // We read the requested load/unload operations from a completed frame.
  // Within the function we try to make new geometry groups residents,
  // and unloaded ones non-resident.
  // This triggers a storage transfer within the provided command buffer.
  if(m_requestsTaskQueue.canPop(device, ensureAcquisition))
  {
    uint32_t popRequestIndex = m_requestsTaskQueue.pop();

#if 1
    // variant where we pop to latest request
    // otherwise we do process requests in strict order
    while(m_requestsTaskQueue.canPop(device, false))
    {
      // ignore previous request
      m_requestsTaskQueue.releaseTaskIndex(popRequestIndex);
      // and use next instead
      popRequestIndex = m_requestsTaskQueue.pop();
    }
#endif

    uint32_t dependentIndex = handleCompletedRequest(cmd, cmdQueueState, asyncQueueState, settings, popRequestIndex);
    // check if immediate update to perform
    if(dependentIndex != INVALID_TASK_INDEX)
    {
      // cannot have deferred and immediate update
      assert(pushUpdateIndex == INVALID_TASK_INDEX);

      pushUpdateIndex   = dependentIndex;
      isImmediateUpdate = true;
    }
  }

  // test if there is an update to be done this frame
  if(pushUpdateIndex != INVALID_TASK_INDEX)
  {
    // Given we know all data was uploaded, we can run the updates to the scene
    // in this frame, which ultimately fulfills a past request on the device.
    //
    // Within this frame compute shaders and other operations, will handle the data
    // provided via values and pointers that are written into.
    // m_shaderData
    //
    // This will mean the current frame can use the new data.

    // both resident and update operations are a synchronized pair, hence
    // single index is sufficient.

    m_resident.applyTask(m_shaderData.resident, pushUpdateIndex, m_frameIndex);
    m_updates.applyTask(m_shaderData.update, pushUpdateIndex, m_frameIndex);

    // we later want to detect the completion of the update task
    // (this was the first thing we did in this function),
    // so push it to task queue
    m_updatesTaskQueue.push(pushUpdateIndex, cmdQueueState.getCurrentState());

    m_lastUpdateIndex = pushUpdateIndex;
  }
  else
  {
    // no patch work this frame
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

  // push new request
  {
    // every frame we will setup new space for new requests made by the device.
    // This is the type of request that we reacted on a few lines above in the
    // `handleCompletedRequest` function.

    uint32_t pushRequestIndex = m_requestsTaskQueue.acquireTaskIndex();
    // the acquisition must be guaranteed by design, as we always handle requests.
    assert(pushRequestIndex != INVALID_TASK_INDEX);

    // get space for request storage
    // and setup this frame's m_shaderData, so that the streaming
    // logic can write to the appropriate pointers.
    m_requests.applyTask(m_shaderData.request, pushRequestIndex, m_frameIndex);
  }

  m_shaderData.frameIndex               = m_frameIndex;
  m_shaderData.ageThreshold             = settings.ageThreshold;

  // upload final configurations for this frame
  vkCmdUpdateBuffer(cmd, m_shaderBuffer.buffer, 0, sizeof(m_shaderData), &m_shaderData);
}

uint32_t SceneStreaming::handleCompletedRequest(VkCommandBuffer      cmd,
                                                QueueState&          cmdQueueState,
                                                QueueState&          asyncQueueState,
                                                const FrameSettings& settings,
                                                uint32_t             popRequestIndex)
{
  // This function handles the requests from the device to upload new geometry groups,
  // or unload some that haven't been used in a while.
  // The readback of the data is guaranteed to have completed at this point.
  // Uploading will try to handle as much requests as we have memory for.
  // Uploading can be done through an async transfer or on the provided command buffer.
  // After an upload is completed an update task must be run, we can
  // run this task immediately or deferred (see later).
  //
  // Only called in the `SceneStreaming::cmdBeginFrame` function.

  const StreamingRequests::TaskInfo& request = m_requests.getCompletedTask(popRequestIndex);

  // during recording of requests the counters may exceed the limits
  // however the data is always ensured to be within.
  uint32_t loadCount   = std::min(request.shaderData->maxLoads, request.shaderData->loadCounter);
  uint32_t unloadCount = std::min(request.shaderData->maxUnloads, request.shaderData->unloadCounter);





#if !STREAMING_DEBUG_FORCE_REQUESTS
  if((!loadCount && !unloadCount) || !m_debugFrameLimit)
  {
    // no work to do
    m_requestsTaskQueue.releaseTaskIndex(popRequestIndex);
    return INVALID_TASK_INDEX;
  }
#endif

  // for debugging
  if(m_debugFrameLimit > 0)
    m_debugFrameLimit--;

  uint32_t pushStorageIndex = m_storageTaskQueue.acquireTaskIndex();
  uint32_t pushUpdateIndex  = m_updatesTaskQueue.acquireTaskIndex();

  // early out if we are not able to acquire both tasks to serve the request
  if(pushStorageIndex == INVALID_TASK_INDEX || pushUpdateIndex == INVALID_TASK_INDEX)
  {
    // give back acquisitions we don't make use of
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


  // let's do unloads first, so we can recycle resident objects
  for(uint32_t g = 0; g < unloadCount; g++)
  {
    GeometryGroup geometryGroup = request.unloadGeometryGroups[g];

    assert(geometryGroup.geometryID < m_scene->getActiveGeometryCount());
    assert(geometryGroup.groupID < m_scene->getActiveGeometry(geometryGroup.geometryID).totalClustersCount);

    const StreamingResident::Group* group = m_resident.findGroup(geometryGroup);
    if(!group)
    {
      // The group might already be removed through a previous request.
      // This can happen cause it may take a while until the patch that really removes something
      // is applied on GPU timeline.
      continue;
    }

    // setup patch
    uint32_t                  unloadIndex = updateTask.unloadCount++;
    shaderio::StreamingPatch& patch       = updateTask.unloadPatches[unloadIndex];
    patch.geometryID                      = geometryGroup.geometryID;
    patch.groupID                         = geometryGroup.groupID;
    patch.groupAddress                    = STREAMING_INVALID_ADDRESS_START;

    // note actual storage memory cannot be recycled here, cause only
    // once the new "update" operation was completed, the gpu's scene graph
    // will not use the data anymore.
    // So defer the actual unloading to the `SceneStreaming::handleCompletedUpdate`
    // above.
    assert(group->storageHandle);
    updateTask.unloadHandles[unloadIndex] = group->storageHandle;

    assert(m_persistentGeometries[geometryGroup.geometryID].lodLoadedGroupsCount[group->lodLevel] > 0);
    m_persistentGeometries[geometryGroup.geometryID].lodLoadedGroupsCount[group->lodLevel]--;

    // and remove from active resident
    m_resident.removeGroup(group->groupResidentID);

    // append to geometry patch list if necessary
  }

  // all newly added groups will be appended to the active list
  updateTask.loadActiveGroupsOffset   = m_resident.getLoadActiveGroupsOffset();
  updateTask.loadActiveClustersOffset = m_resident.getLoadActiveClustersOffset();

  uint64_t transferBytes = 0;

  m_stats.couldNotAllocateClas  = 0;
  m_stats.couldNotTransfer      = 0;
  m_stats.couldNotAllocateGroup = 0;
  m_stats.couldNotStore         = 0;
  m_stats.uncompletedLoadCount  = 0;

  // 优化：批量处理加载请求，减少冗余操作
  uint32_t processedLoads = 0;
  for(uint32_t g = 0; g < loadCount; g++)
  {
    GeometryGroup geometryGroup = request.loadGeometryGroups[g];

    assert(geometryGroup.geometryID < m_scene->getActiveGeometryCount());
    assert(geometryGroup.groupID < m_scene->getActiveGeometry(geometryGroup.geometryID).totalClustersCount);

    if(m_resident.findGroup(geometryGroup))
    {
      // It could take more than one frame until the patch that handles the load
      // is activated on the GPU timeline, and until then the same requests might be
      // made.
      continue;
    }

    const Scene::GeometryView& sceneGeometry = m_scene->getActiveGeometry(geometryGroup.geometryID);

    // figure out size of this geometry group.
    // This includes all relevant cluster data, including vertices, triangle indices...
    const Scene::GroupInfo groupInfo       = sceneGeometry.groupInfos[geometryGroup.groupID];
    uint32_t               clusterCount    = groupInfo.clusterCount;
    uint64_t               groupDeviceSize = groupInfo.getDeviceSize();

    uint64_t                  deviceAddress;
    nvvk::BufferSubAllocation storageHandle;

    // 优化：先检查是否可以传输，避免不必要的内存分配
    if(!m_storage.canTransfer(storageTask, groupDeviceSize))
    {
      m_stats.couldNotTransfer++;
      m_stats.uncompletedLoadCount++;
      continue;
    }

    bool canStore         = m_storage.allocate(storageHandle, geometryGroup, groupDeviceSize, deviceAddress);
    bool canAllocateGroup = m_resident.canAllocateGroup(clusterCount);

    // test if we can allocate
    if(!canStore || !canAllocateGroup)
    {
      m_stats.couldNotAllocateGroup += (!canAllocateGroup);
      m_stats.couldNotStore += (!canStore);

      if(canStore)
      {
        // return memory on failure
        m_storage.free(storageHandle);
      }

      if(clusterCount < 8)
      {
        m_stats.uncompletedLoadCount += loadCount - g;
        break;  // heuristic if small groups don't fit anymore then we fully break
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
      Scene::GroupView groupView(sceneGeometry.groupData, groupInfo);
      if(groupInfo.uncompressedSizeBytes)
      {
        Scene::decompressGroup(groupInfo, groupView, groupData, groupDeviceSize);
      }
      else
      {
        // simply copy data as is, the streaming patch will take care of modifying the data
        // where needed
        memcpy(groupData, groupView.raw, groupView.rawSize);
      }
    }

    m_persistentGeometries[geometryGroup.geometryID].lodLoadedGroupsCount[groupInfo.lodLevel]++;

    // append to geometry patch list if necessary

    // setup patch
    shaderio::StreamingPatch& patch = updateTask.loadPatches[updateTask.loadCount++];
    patch.geometryID                = geometryGroup.geometryID;
    patch.groupID                   = geometryGroup.groupID;
    patch.groupAddress              = deviceAddress;
    patch.groupResidentID           = residentGroup->groupResidentID;
    patch.clusterResidentID         = residentGroup->clusterResidentID;
    patch.clusterCount              = groupInfo.clusterCount;
    patch.lodLevel                  = groupInfo.lodLevel;


    // stats
    transferBytes += groupInfo.sizeBytes;
  }

#if !STREAMING_DEBUG_FORCE_REQUESTS
  if(updateTask.loadCount == 0 && updateTask.unloadCount == 0)
  {
    // we ended up doing no work
    m_requestsTaskQueue.releaseTaskIndex(popRequestIndex);
    m_updatesTaskQueue.releaseTaskIndex(pushUpdateIndex);
    m_storageTaskQueue.releaseTaskIndex(pushStorageIndex);
    return INVALID_TASK_INDEX;
  }
#endif

  if(m_config.useAsyncTransfer)
  {
    // don't use immediate command buffer from main queue,
    // but use transfer queue instead.

    NVVK_CHECK(m_storage.m_taskCommandPool.acquireCommandBuffer(pushStorageIndex, cmd));
    VkCommandBufferBeginInfo cmdInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
    };

    vkBeginCommandBuffer(cmd, &cmdInfo);
  }

  uint32_t transferCount = 0;
  // finalize data for completed new residency & patch
  // residency & updates always operate in synchronized pairs
  transferBytes += m_updates.cmdUploadTask(cmd, pushUpdateIndex);
  transferBytes += m_resident.cmdUploadTask(cmd, pushUpdateIndex);
  transferCount += m_storage.cmdUploadTask(cmd);
  transferCount += 2;

  // 优化：批量更新统计信息，减少冗余操作
  if(processedLoads > 0) {
    // only log to stats for loads
    m_stats.transferBytes = transferBytes;
    m_stats.transferCount = transferCount;
    m_stats.loadCount = updateTask.loadCount;
  }
  if(updateTask.unloadCount > 0) {
    m_stats.unloadCount = updateTask.unloadCount;
  }

  // When we use async we can either wait until async completed (can take more than a frame)
  // or we guarantee it completes for the frame we are currently preparing within `cmd`.
  // When not using async we always know the transfer completes within the current frame.

  bool useDecoupledUpdate = m_config.useAsyncTransfer && m_config.useDecoupledAsyncTransfer;

  nvvk::SemaphoreState storageSemaphoreState =
      m_config.useAsyncTransfer ? asyncQueueState.getCurrentState() : cmdQueueState.getCurrentState();

  if(m_config.useAsyncTransfer)
  {
    vkEndCommandBuffer(cmd);

    if(!m_config.useDecoupledAsyncTransfer)
    {
      // if not using decoupled, then let immediate command buffer's queue wait for this
      // transfer to be completed

      // get wait from async queue
      VkSemaphoreSubmitInfo semWaitInfo = asyncQueueState.getWaitSubmit(VK_PIPELINE_STAGE_2_TRANSFER_BIT);
      // push it for use in primary queue
      cmdQueueState.m_pendingWaits.push_back(semWaitInfo);
    }

    // trigger async transfer queue submit
    // 优化：使用更高效的提交方式，减少同步开销
    VkSemaphoreSubmitInfo     semSubmitInfo = asyncQueueState.advanceSignalSubmit(VK_PIPELINE_STAGE_2_TRANSFER_BIT);
    VkCommandBufferSubmitInfo cmdBufInfo    = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmdBufInfo.commandBuffer                = cmd;
    
    // 优化：使用批次提交，减少API调用开销
    VkSubmitInfo2 submits            = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR};
    submits.pCommandBufferInfos      = &cmdBufInfo;
    submits.commandBufferInfoCount   = 1;
    submits.pSignalSemaphoreInfos    = &semSubmitInfo;
    submits.signalSemaphoreInfoCount = 1;
    
    // 优化：使用非阻塞提交，提高传输效率
    vkQueueSubmit2(asyncQueueState.m_queue, 1, &submits, nullptr);
  }

  // give back the index for future write operations
  m_requestsTaskQueue.releaseTaskIndex(popRequestIndex);

  // enqueue the storage task
  // the dependentIndex may be set to the pushUpdateIndex if we use decoupled
  m_storageTaskQueue.push(pushStorageIndex, storageSemaphoreState, useDecoupledUpdate ? pushUpdateIndex : INVALID_TASK_INDEX);

  // otherwise, we return update task to be handled directly in this frame
  return useDecoupledUpdate ? INVALID_TASK_INDEX : pushUpdateIndex;
}



static uint32_t getWorkGroupCount(uint32_t numThreads, uint32_t workGroupSize)
{
  // compute workgroup count from threads
  return (numThreads + workGroupSize - 1) / workGroupSize;
}

void SceneStreaming::cmdPreTraversal(VkCommandBuffer cmd, VkDeviceAddress clasScratchBuffer, nvvk::ProfilerGpuTimer& profiler)
{
  Resources& res = *m_resources;
  // Prior traversal we run the update task.
  // This modifies the device address array of geometry groups so that
  // traversal knows whether a geometry group is resident or not and where to
  // find it.
  // It also handles the unloading by invalidating such addresses.
  //
  // This function is called by the renderer.

  auto timerSection = profiler.cmdFrameSection(cmd, "Stream Pre Traversal");


  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);

  // if we have an update to perform do it prior traversal
  if(m_shaderData.update.patchGroupsCount)
  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeUpdateSceneRaster);

    res.cmdLinearDispatch(cmd, getWorkGroupCount(m_shaderData.update.patchGroupsCount, STREAM_UPDATE_SCENE_WORKGROUP));

    // with the update also comes a new compacted list of resident objects
    m_resident.cmdRunTask(cmd, m_shaderData.update.taskIndex);
  }

  // rasterization ends here
}

void SceneStreaming::cmdPostTraversal(VkCommandBuffer cmd, VkDeviceAddress clasScratchBuffer, bool runAgeFilter, nvvk::ProfilerGpuTimer& profiler)
{
  Resources& res = *m_resources;

  // After traversal was performed, this function filters resident cluster groups
  // by age to append to the unload request list.
  // The traversal itself will have appended load requests and reset the age of
  // used cluster groups.
  //
  // This function is called by the renderer.

  auto timerSection = profiler.cmdFrameSection(cmd, "Stream Post Traversal");

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_dsetPack.getSetPtr(), 0, nullptr);

  if(m_shaderData.resident.activeGroupsCount && runAgeFilter)
  {
    // age filter resident groups, writes unload request array

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeAgeFilterGroups);
    res.cmdLinearDispatch(cmd, getWorkGroupCount(m_shaderData.resident.activeGroupsCount, STREAM_AGEFILTER_GROUPS_WORKGROUP));
  }

  // rasterization ends here
}

void SceneStreaming::cmdEndFrame(VkCommandBuffer cmd, QueueState& cmdQueueState, nvvk::ProfilerGpuTimer& profiler)
{
  // Perform the request readback.
  // we pass the location of `shaderio::StreamingRequest` within m_streamingBuffer, as it contains
  // the counter values for how much loads/unloads to perform as well as how much memory
  // for ray tracing CLAS is currently in use.
  //
  // This function is called by the renderer.

  auto timerSection = profiler.cmdFrameSection(cmd, "Stream End");

  m_requests.cmdRunTask(cmd, m_shaderData.request, m_shaderBuffer.buffer, offsetof(shaderio::SceneStreaming, request));

  m_requestsTaskQueue.push(m_shaderData.request.taskIndex, cmdQueueState.getCurrentState());

  m_frameIndex++;

  // for benchmarking. useful to see how many frames it takes to load data until peak is reached
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

void SceneStreaming::getStats(StreamingStats& stats) const
{
  stats = m_stats;

  m_storage.getStats(stats);
  m_resident.getStats(stats);
  stats.persistentDataBytes = m_persistentGeometrySize;
}

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

  // reset resident objects to just roots
  m_resident.reset(m_shaderData.resident);
  m_updates.reset();

  // reset dynamic storage
  m_storage.reset();

  // need to reset internal clock
  m_frameIndex = 1;

  Resources::BatchedUploader uploader(res);
  resetGeometryGroupAddresses(uploader);
  uploader.flush();
}

bool SceneStreaming::initShadersAndPipelines()
{
  Resources& res = *m_resources;

  shaderc::CompileOptions options = res.makeCompilerOptions();
  options.AddMacroDefinition("SUBGROUP_SIZE", fmt::format("{}", res.m_physicalDeviceInfo.properties11.subgroupSize));
  options.AddMacroDefinition("USE_16BIT_DISPATCH", fmt::format("{}", res.m_use16bitDispatch ? 1 : 0));

  shaderc::CompileOptions optionsRaster = options;
  optionsRaster.AddMacroDefinition("TARGETS_RASTERIZATION", "1");

  res.compileShader(m_shaders.computeAgeFilterGroups, VK_SHADER_STAGE_COMPUTE_BIT, "stream_agefilter_groups.comp.glsl", &options);
  res.compileShader(m_shaders.computeSetup, VK_SHADER_STAGE_COMPUTE_BIT, "stream_setup.comp.glsl", &options);
  res.compileShader(m_shaders.computeUpdateSceneRaster, VK_SHADER_STAGE_COMPUTE_BIT, "stream_update_scene.comp.glsl", &optionsRaster);

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

void SceneStreaming::deinitShadersAndPipelines()
{
  Resources& res = *m_resources;

  res.destroyPipelines(m_pipelines);
}


}  // namespace lodclusters
