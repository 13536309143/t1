//==============================================================================
// 文件：src/renderer/preloaded.cpp
// 模块定位：预加载 GPU 场景实现，判断显存容量、创建 组/簇 数据 缓冲 并填充 着色器 地址。
// 数据流：输入是 Scene 几何视图；输出是全量驻留的 GPU 数据和 Geometry 地址表。
// 方法说明：该实现把 CPU 的分散 span 打包成 GPU 连续存储，减少运行时地址修补和缺页处理成本。
// 正确性约束：容量估算要保守；每个 Geometry 的 low detail、LOD level、node 和 组 地址必须对应正确 缓冲 偏移。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <volk.h>
#include "preloaded.hpp"


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace lodclusters {


// 函数：ScenePreloaded::canPreload。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
bool ScenePreloaded::canPreload(VkDeviceSize deviceLocalHeapSize, const Scene* scene)
{
  VkDeviceSize sizeLimit = (deviceLocalHeapSize * 600) / 1000;
  VkDeviceSize testSize  = 0;

  for(size_t geometryIndex = 0; geometryIndex < scene->getActiveGeometryCount(); geometryIndex++)
  {

    const Scene::GeometryView& sceneGeometry = scene->getActiveGeometry(geometryIndex);
    ScenePreloaded::Geometry   preloadGeometry;

    size_t numNodes = sceneGeometry.lodNodes.size();
    testSize += preloadGeometry.lodNodes.value_size * numNodes;
    testSize += preloadGeometry.lodNodeBboxes.value_size * numNodes;

    uint32_t numLodLevels = sceneGeometry.lodLevelsCount;
    testSize += preloadGeometry.lodLevels.value_size * numLodLevels;
  }

  if(testSize > sizeLimit)
  {

    LOGI("Likely exceeding device memory limit for preloaded scene\n");
    return false;
  }

  return true;
}


// 函数：ScenePreloaded::init。初始化本模块所需状态、资源或 GPU 侧绑定。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
bool ScenePreloaded::init(Resources* res, const Scene* scene, const Config& config)
{

  assert(m_resources == nullptr && "init called without prior deinit");

  m_resources = res;
  m_scene     = scene;
  m_config    = config;

  if(!canPreload(res->getDeviceLocalHeapSize(), scene))
  {

    LOGW("Likely exceeding device memory limit for preloaded scene\n");
    return false;
  }


  m_shaderGeometries.resize(scene->getActiveGeometryCount());
  m_geometries.resize(scene->getActiveGeometryCount());


  // 函数：uploader。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
  Resources::BatchedUploader uploader(*res);

  uint32_t instancesOffset = 0;
  for(size_t geometryIndex = 0; geometryIndex < scene->getActiveGeometryCount(); geometryIndex++)
  {
    shaderio::Geometry&        shaderGeometry  = m_shaderGeometries[geometryIndex];
    ScenePreloaded::Geometry&  preloadGeometry = m_geometries[geometryIndex];

    const Scene::GeometryView& sceneGeometry   = scene->getActiveGeometry(geometryIndex);


    size_t groupDataSize = sceneGeometry.groupData.size_bytes();

    if(scene->m_config.useCompressedData)
    {
      groupDataSize = 0;
      for(size_t g = 0; g < sceneGeometry.groupInfos.size(); g++)
      {
        const Scene::GroupInfo groupInfo = sceneGeometry.groupInfos[g];

        groupDataSize += groupInfo.getDeviceSize();
      }
    }

    res->createBuffer(preloadGeometry.groupData, groupDataSize,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

    NVVK_DBG_NAME(preloadGeometry.groupData.buffer);

    res->createBufferTyped(preloadGeometry.groupAddresses, sceneGeometry.groupInfos.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    res->createBufferTyped(preloadGeometry.clusterAddresses, sceneGeometry.totalClustersCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    NVVK_DBG_NAME(preloadGeometry.groupAddresses.buffer);

    NVVK_DBG_NAME(preloadGeometry.clusterAddresses.buffer);


    size_t numNodes = sceneGeometry.lodNodes.size();

    res->createBufferTyped(preloadGeometry.lodNodes, numNodes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    res->createBufferTyped(preloadGeometry.lodNodeBboxes, numNodes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    NVVK_DBG_NAME(preloadGeometry.lodNodes.buffer);

    NVVK_DBG_NAME(preloadGeometry.lodNodeBboxes.buffer);

    uint32_t numLodLevels = sceneGeometry.lodLevelsCount;

    res->createBufferTyped(preloadGeometry.lodLevels, numLodLevels, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    NVVK_DBG_NAME(preloadGeometry.lodLevels.buffer);

    m_geometrySize += preloadGeometry.groupData.bufferSize;
    m_geometrySize += preloadGeometry.groupAddresses.bufferSize;
    m_geometrySize += preloadGeometry.clusterAddresses.bufferSize;
    m_geometrySize += preloadGeometry.lodLevels.bufferSize;
    m_geometrySize += preloadGeometry.lodNodes.bufferSize;
    m_geometrySize += preloadGeometry.lodNodeBboxes.bufferSize;


    shaderGeometry                    = {};
    shaderGeometry.bbox               = sceneGeometry.bbox;
    shaderGeometry.nodes              = preloadGeometry.lodNodes.address;
    shaderGeometry.nodeBboxes         = preloadGeometry.lodNodeBboxes.address;
    shaderGeometry.preloadedGroups    = preloadGeometry.groupAddresses.address;
    shaderGeometry.preloadedClusters  = preloadGeometry.clusterAddresses.address;

    shaderGeometry.lodLevelsCount     = uint32_t(numLodLevels);
    shaderGeometry.lodLevels          = preloadGeometry.lodLevels.address;

    shaderGeometry.instancesCount     = sceneGeometry.instanceReferenceCount * scene->getGeometryInstanceFactor();
    shaderGeometry.instancesOffset    = instancesOffset;

    instancesOffset += shaderGeometry.instancesCount;


    shaderio::LodLevel lastLodLevel = sceneGeometry.lodLevels.back();

    assert(lastLodLevel.groupCount == 1 && lastLodLevel.clusterCount == 1);

    shaderGeometry.lowDetailClusterID = lastLodLevel.clusterOffset;
    shaderGeometry.lowDetailTriangles = sceneGeometry.groupInfos[lastLodLevel.groupOffset].triangleCount;


    uploader.uploadBuffer(preloadGeometry.lodNodes, sceneGeometry.lodNodes.data());
    uploader.uploadBuffer(preloadGeometry.lodNodeBboxes, sceneGeometry.lodNodeBboxes.data());
    uploader.uploadBuffer(preloadGeometry.lodLevels, sceneGeometry.lodLevels.data());


    uint64_t* clusterAddresses = uploader.uploadBuffer(preloadGeometry.clusterAddresses, (uint64_t*)nullptr);
    uint64_t* groupAddresses =
        uploader.uploadBuffer(preloadGeometry.groupAddresses, (uint64_t*)nullptr, Resources::FlushState::DONT_FLUSH);
    uint8_t* groupData = uploader.uploadBuffer(preloadGeometry.groupData, (uint8_t*)nullptr, Resources::FlushState::DONT_FLUSH);

    uint32_t clusterOffset   = 0;
    size_t   groupDataOffset = 0;
    for(size_t g = 0; g < sceneGeometry.groupInfos.size(); g++)
    {
      const Scene::GroupInfo groupInfo = sceneGeometry.groupInfos[g];


      // 函数：groupView。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
      // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
      // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
      const Scene::GroupView groupView(sceneGeometry.groupData, groupInfo);
      uint64_t               groupVA = preloadGeometry.groupData.address + groupDataOffset;

      groupAddresses[g] = groupVA;

      Scene::fillGroupRuntimeData(groupInfo, groupView, uint32_t(g), uint32_t(g), clusterOffset,
                                  groupData + groupDataOffset, groupInfo.getDeviceSize());


      groupDataOffset += groupInfo.getDeviceSize();

      for(uint32_t c = 0; c < groupInfo.clusterCount; c++)
      {
        clusterAddresses[c + clusterOffset] = groupVA + sizeof(shaderio::Group) + sizeof(shaderio::Cluster) * c;
      }

      clusterOffset += groupInfo.clusterCount;
    }
  }

  res->createBufferTyped(m_shaderGeometriesBuffer, scene->getActiveGeometryCount(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  NVVK_DBG_NAME(m_shaderGeometriesBuffer.buffer);

  m_operationsSize += logMemoryUsage(m_shaderGeometriesBuffer.bufferSize, "operations", "preloaded geo buffer");

  uploader.uploadBuffer(m_shaderGeometriesBuffer, m_shaderGeometries.data());

  uploader.flush();

  return true;
}


// 函数：ScenePreloaded::deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
void ScenePreloaded::deinit()
{
  if(!m_resources)
    return;

  for(auto& it : m_geometries)
  {

    m_resources->m_allocator.destroyBuffer(it.clusterAddresses);

    m_resources->m_allocator.destroyBuffer(it.groupData);

    m_resources->m_allocator.destroyBuffer(it.groupAddresses);

    m_resources->m_allocator.destroyBuffer(it.lodNodes);

    m_resources->m_allocator.destroyBuffer(it.lodNodeBboxes);

    m_resources->m_allocator.destroyBuffer(it.lodLevels);
  }


  m_resources->m_allocator.destroyBuffer(m_shaderGeometriesBuffer);
  m_resources    = nullptr;
  m_scene        = nullptr;
  m_geometrySize = 0;
}
}
