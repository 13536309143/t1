//==============================================================================
// 文件：src/scene/scene.cpp
// 模块定位：Scene 主流程实现，组织缓存检查、glTF 读取、几何处理、LOD 构建、压缩、实例网格复制和统计生成。
// 数据流：输入是场景文件、加载配置和构建配置；输出是完整 Scene、histogram、bbox、active geometry 和可保存缓存。
// 方法说明：该流程把“语义导入”和“渲染友好重排”分离，先建立几何语义，再生成适合 GPU 随机访问和批量遍历的运行时布局。
// 正确性约束：多线程处理时进度和日志需线程安全；缓存命中不得重复构建；所有统计必须在 active geometry/grid 更新后重新归约。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <cinttypes>
#include <cstring>
#include <algorithm>
#include <random>
#include <meshoptimizer.h>
#include <nvutils/logger.hpp>
#include <nvutils/parallel_work.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/hash_operations.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext/scalar_constants.hpp>
#include "scene.hpp"


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace lodclusters {

namespace {

shaderio::BBox makeEmptyBBox()
{
  return {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0.0f, 0.0f};
}

bool isValidBBox(const shaderio::BBox& bbox)
{
  return bbox.lo.x <= bbox.hi.x && bbox.lo.y <= bbox.hi.y && bbox.lo.z <= bbox.hi.z;
}

void updateBBoxEdges(shaderio::BBox& bbox)
{
  if(!isValidBBox(bbox))
  {
    bbox.shortestEdge = 0.0f;
    bbox.longestEdge  = 0.0f;
    return;
  }

  const glm::vec3 extent = bbox.hi - bbox.lo;
  bbox.shortestEdge      = std::min(extent.x, std::min(extent.y, extent.z));
  bbox.longestEdge       = std::max(extent.x, std::max(extent.y, extent.z));
}

void mergeBBox(shaderio::BBox& dst, const shaderio::BBox& src)
{
  if(!isValidBBox(src))
  {
    return;
  }

  dst.lo = glm::min(dst.lo, src.lo);
  dst.hi = glm::max(dst.hi, src.hi);
  updateBBoxEdges(dst);
}

}


// 函数：Scene::ProcessingInfo::init。初始化本模块所需状态、资源或 GPU 侧绑定。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
void Scene::ProcessingInfo::init(float processingThreadsPct)
{

  numPoolThreadsOriginal = nvutils::get_thread_pool().get_thread_count();
  numPoolThreads = numPoolThreadsOriginal;
  if(processingThreadsPct > 0.0f && processingThreadsPct < 1.0f)
  {
    numPoolThreads = std::min(numPoolThreads, std::max(1u, uint32_t(ceilf(float(numPoolThreads) * processingThreadsPct))));

    if(numPoolThreads != numPoolThreadsOriginal)

      nvutils::get_thread_pool().reset(numPoolThreads);
  }
}


// 函数：Scene::ProcessingInfo::setupParallelism。初始化本模块所需状态、资源或 GPU 侧绑定。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
void Scene::ProcessingInfo::setupParallelism(size_t geometryCount_, size_t geometryCompletedCount, int parallelismMode)
{
  geometryCount = geometryCount_;
  bool preferInnerParallelism = (geometryCount - geometryCompletedCount) < numPoolThreads;

  if(parallelismMode < 0)
  {
    preferInnerParallelism = true;
  }
  if(parallelismMode > 0)
  {
    preferInnerParallelism = false;
  }

  numOuterThreads = preferInnerParallelism ? 1 : numPoolThreads;
  numInnerThreads = preferInnerParallelism ? numPoolThreads : 1;
}


// 函数：Scene::ProcessingInfo::setupCompressedGltf。初始化本模块所需状态、资源或 GPU 侧绑定。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
void Scene::ProcessingInfo::setupCompressedGltf(size_t bufferViewCount)
{
  bufferViewUsers.resize(bufferViewCount, {0});
  bufferViewLocks.resize(bufferViewCount, {0});
}


// 函数：Scene::ProcessingInfo::logBegin。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void Scene::ProcessingInfo::logBegin(uint64_t totalTriangleCount)
{
  LOGI("... geometry load & processing: geometries %" PRIu64 ", threads outer %d inner %d\n", geometryCount,
       numOuterThreads, numInnerThreads);


  startTime = clock.getMicroseconds();

  triangleCount               = totalTriangleCount;
  progressTrianglesCompleted  = 0;
  progressGeometriesCompleted = 0;
  progressLastPercentage      = 0;
}


// 函数：Scene::ProcessingInfo::logCompletedGeometry。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
uint32_t Scene::ProcessingInfo::logCompletedGeometry(uint64_t geometryTriangleCount)
{


  // 函数：lock。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  std::lock_guard lock(progressMutex);

  progressGeometriesCompleted++;
  progressTrianglesCompleted += geometryTriangleCount;

  uint32_t percentage;
  if(!triangleCount)
  {
    percentage = uint32_t(double(progressGeometriesCompleted * 100) / double(geometryCount));
  }
  else
  {
    percentage = uint32_t((double(progressTrianglesCompleted) * 100) / double(triangleCount));
  }


  const uint32_t precentageGranularity = 5;
  uint32_t       percentageSnapped     = (percentage / precentageGranularity) * precentageGranularity;

  if(percentageSnapped > progressLastPercentage)
  {
    progressLastPercentage = percentageSnapped;

    LOGI("... geometry load & processing: %3d%%\n", percentageSnapped);
  }

  return percentage;
}


// 函数：Scene::ProcessingInfo::logEnd。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void Scene::ProcessingInfo::logEnd()
{

  double endTime = clock.getMicroseconds();

  LOGI("... geometry load & processing: %f milliseconds\n", (endTime - startTime) / 1000.0f);


  if(stats.groups)
  {

    LOGI("Group Data Stats\n");
    LOGI("Groups:               %12" PRIu64 "\n", (uint64_t)stats.groups);
    LOGI("Clusters:             %12" PRIu64 "\n", (uint64_t)stats.clusters);
    LOGI("Vertices:             %12" PRIu64 "\n", (uint64_t)stats.vertices);
    LOGI("Group Unique Verts:   %12" PRIu64 "\n", (uint64_t)stats.groupUniqueVertices);
    LOGI("Group Header Bytes:   %12" PRIu64 "\n", (uint64_t)stats.groupHeaderBytes);
    LOGI("Cluster Header Bytes: %12" PRIu64 "\n", (uint64_t)stats.clusterHeaderBytes);
    LOGI("Cluster BBox Bytes:   %12" PRIu64 "\n", (uint64_t)stats.clusterBboxBytes);
    LOGI("Cluster GGrp Bytes:   %12" PRIu64 "\n", (uint64_t)stats.clusterGenBytes);
    LOGI("Triangle Index Bytes: %12" PRIu64 "\n", (uint64_t)stats.triangleIndexBytes);
    LOGI("Vertex All Bytes:     %12" PRIu64 "\n", (uint64_t)(stats.vertexPosBytes + stats.vertexNrmBytes + stats.vertexTexCoordBytes));
    LOGI("Vertex Pos Bytes:     %12" PRIu64 "\n", (uint64_t)stats.vertexPosBytes);
    LOGI("Vertex TexCrd Bytes:  %12" PRIu64 "\n", (uint64_t)stats.vertexTexCoordBytes);
    LOGI("Vertex N&T Bytes:     %12" PRIu64 "\n", (uint64_t)stats.vertexNrmBytes);
    LOGI("Vertex Comp Bytes:    %12" PRIu64 "\n", (uint64_t)stats.vertexCompressedBytes);

    LOGI("\n");
  }

  if(stats.featureInputVertices)
  {
    const uint64_t featureVertices = uint64_t(stats.featureInputVertices);
    const auto pct = [featureVertices](uint64_t value) {
      return featureVertices ? (100.0 * double(value) / double(featureVertices)) : 0.0;
    };

    LOGI("Feature Retention Stats\n");
    LOGI("Input Feature Vertices: %12" PRIu64 "\n", featureVertices);
    LOGI("Input Feature Tris:     %12" PRIu64 "\n", (uint64_t)stats.featureInputTriangles);
    LOGI("Boundary Vertices:      %12" PRIu64 " (%6.2f%%)\n", (uint64_t)stats.featureBoundaryVertices,
         pct((uint64_t)stats.featureBoundaryVertices));
    LOGI("Non-Manifold Vertices:  %12" PRIu64 " (%6.2f%%)\n", (uint64_t)stats.featureNonManifoldVertices,
         pct((uint64_t)stats.featureNonManifoldVertices));
    LOGI("Sharp Edge Vertices:    %12" PRIu64 " (%6.2f%%)\n", (uint64_t)stats.featureSharpVertices,
         pct((uint64_t)stats.featureSharpVertices));
    LOGI("Boundary Components:    %12" PRIu64 "\n", (uint64_t)stats.featureBoundaryLoopComponents);
    LOGI("Sharp Ring Components:  %12" PRIu64 "\n", (uint64_t)stats.featureSharpRingComponents);
    LOGI("Circular Hole Loops:    %12" PRIu64 "\n", (uint64_t)stats.featureCircularHoleLoops);
    LOGI("Circular Hole Vertices: %12" PRIu64 " (%6.2f%%)\n", (uint64_t)stats.featureCircularHoleVertices,
         pct((uint64_t)stats.featureCircularHoleVertices));
    LOGI("Functional Boundaries:  %12" PRIu64 " (%6.2f%%)\n", (uint64_t)stats.featureFunctionalBoundaryVertices,
         pct((uint64_t)stats.featureFunctionalBoundaryVertices));
    LOGI("Cylindrical Vertices:   %12" PRIu64 " (%6.2f%%)\n", (uint64_t)stats.featureCylindricalPatchVertices,
         pct((uint64_t)stats.featureCylindricalPatchVertices));
    LOGI("Thin-Wall Vertices:     %12" PRIu64 " (%6.2f%%)\n", (uint64_t)stats.featureThinWallVertices,
         pct((uint64_t)stats.featureThinWallVertices));
    LOGI("Protected Vertices:     %12" PRIu64 " (%6.2f%%)\n", (uint64_t)stats.featureProtectedVertices,
         pct((uint64_t)stats.featureProtectedVertices));
    LOGI("Critical Vertices:      %12" PRIu64 " (%6.2f%%)\n", (uint64_t)stats.featureCriticalVertices,
         pct((uint64_t)stats.featureCriticalVertices));
    LOGI("Avg Feature Importance: %12.4f\n",
         double((uint64_t)stats.featureImportanceSumPpm) / double(featureVertices) / 1000000.0);
    LOGI("Max Feature Importance: %12.4f\n", double((uint64_t)stats.featureImportanceMaxPpm) / 1000000.0);
    LOGI("\n");
  }
}


// 函数：Scene::ProcessingInfo::deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
void Scene::ProcessingInfo::deinit()
{
  if(numPoolThreads != numPoolThreadsOriginal)

    nvutils::get_thread_pool().reset(numPoolThreadsOriginal);
}

void Scene::fillGroupRuntimeData(const GroupInfo& srcGroupInfo,
                                 const GroupView& srcGroupView,
                                 uint32_t         groupID,
                                 uint32_t         groupResidentID,
                                 uint32_t         clusterResidentID,
                                 void*            dst,
                                 size_t           dstSize)
{
  GroupInfo dstGroupInfo = srcGroupInfo;
  if(srcGroupInfo.uncompressedSizeBytes)
  {

    decompressGroup(srcGroupInfo, srcGroupView, dst, dstSize);

    dstGroupInfo.sizeBytes       = dstGroupInfo.uncompressedSizeBytes;
    dstGroupInfo.vertexDataCount = dstGroupInfo.uncompressedVertexDataCount;
  }
  else
  {

    assert(srcGroupView.rawSize <= dstSize);

    memcpy(dst, srcGroupView.raw, srcGroupView.rawSize);
  }


  {


    // 函数：groupStorage。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    GroupStorage groupStorage(dst, dstGroupInfo);
    groupStorage.group->residentID        = groupResidentID;
    groupStorage.group->clusterResidentID = clusterResidentID;
  }
}

Scene::Result Scene::init(const std::filesystem::path& filePath,
                          const SceneConfig&           config,
                          const SceneLoaderConfig&     loaderConfig,
                          const std::string&           cacheSuffix,
                          bool                         skipCache)
{
  *this = {};

  m_filePath             = filePath;
  m_config               = config;
  m_loaderConfig         = loaderConfig;
  m_loadedFromCache      = false;
  m_cacheFilePath        = filePath;
  m_cachePartialFilePath = filePath;
  m_cacheFileSize        = 0;
  m_cacheSuffix          = cacheSuffix;


  std::string oldExtension = filePath.extension().string();

  m_cacheFilePath.replace_extension(oldExtension + cacheSuffix);

  m_cachePartialFilePath.replace_extension(oldExtension + cacheSuffix + "_partial");

  if(!skipCache && !m_loaderConfig.processingOnly && m_loaderConfig.autoLoadCache)
  {

    openCache();
  }

  ProcessingInfo processingInfo;

  processingInfo.init(m_loaderConfig.processingThreadsPct);


  Result loadResult = loadGLTF(processingInfo, filePath);
  if(loadResult == SCENE_RESULT_NEEDS_PREPROCESS || loadResult == SCENE_RESULT_CACHE_INVALID)
  {

    LOGI("Scene::init large scene or invalid cache detected\n  using dedicated preprocess pass\n");

    closeCache();

    m_loaderConfig.processingOnly = true;

    loadResult                    = loadGLTF(processingInfo, filePath);
    m_loaderConfig.processingOnly = false;
    if(loadResult == SCENE_RESULT_PREPROCESS_COMPLETED)
    {

      openCache();

      loadResult = loadGLTF(processingInfo, filePath);
    }
  }


  processingInfo.deinit();

  m_processingStats.groups                = uint64_t(processingInfo.stats.groups);
  m_processingStats.clusters              = uint64_t(processingInfo.stats.clusters);
  m_processingStats.vertices              = uint64_t(processingInfo.stats.vertices);
  m_processingStats.groupUniqueVertices   = uint64_t(processingInfo.stats.groupUniqueVertices);
  m_processingStats.groupHeaderBytes      = uint64_t(processingInfo.stats.groupHeaderBytes);
  m_processingStats.triangleIndexBytes    = uint64_t(processingInfo.stats.triangleIndexBytes);
  m_processingStats.vertexPosBytes        = uint64_t(processingInfo.stats.vertexPosBytes);
  m_processingStats.vertexTexCoordBytes   = uint64_t(processingInfo.stats.vertexTexCoordBytes);
  m_processingStats.vertexNrmBytes        = uint64_t(processingInfo.stats.vertexNrmBytes);
  m_processingStats.vertexCompressedBytes = uint64_t(processingInfo.stats.vertexCompressedBytes);
  m_processingStats.clusterBboxBytes      = uint64_t(processingInfo.stats.clusterBboxBytes);
  m_processingStats.clusterHeaderBytes    = uint64_t(processingInfo.stats.clusterHeaderBytes);
  m_processingStats.clusterGenBytes       = uint64_t(processingInfo.stats.clusterGenBytes);

  m_processingStats.featureInputVertices              = uint64_t(processingInfo.stats.featureInputVertices);
  m_processingStats.featureInputTriangles             = uint64_t(processingInfo.stats.featureInputTriangles);
  m_processingStats.featureBoundaryVertices           = uint64_t(processingInfo.stats.featureBoundaryVertices);
  m_processingStats.featureNonManifoldVertices        = uint64_t(processingInfo.stats.featureNonManifoldVertices);
  m_processingStats.featureSharpVertices              = uint64_t(processingInfo.stats.featureSharpVertices);
  m_processingStats.featureBoundaryLoopComponents     = uint64_t(processingInfo.stats.featureBoundaryLoopComponents);
  m_processingStats.featureSharpRingComponents        = uint64_t(processingInfo.stats.featureSharpRingComponents);
  m_processingStats.featureCircularHoleLoops          = uint64_t(processingInfo.stats.featureCircularHoleLoops);
  m_processingStats.featureCircularHoleVertices       = uint64_t(processingInfo.stats.featureCircularHoleVertices);
  m_processingStats.featureFunctionalBoundaryVertices = uint64_t(processingInfo.stats.featureFunctionalBoundaryVertices);
  m_processingStats.featureCylindricalPatchVertices   = uint64_t(processingInfo.stats.featureCylindricalPatchVertices);
  m_processingStats.featureThinWallVertices           = uint64_t(processingInfo.stats.featureThinWallVertices);
  m_processingStats.featureProtectedVertices          = uint64_t(processingInfo.stats.featureProtectedVertices);
  m_processingStats.featureCriticalVertices           = uint64_t(processingInfo.stats.featureCriticalVertices);
  m_processingStats.featureImportanceSumPpm           = uint64_t(processingInfo.stats.featureImportanceSumPpm);
  m_processingStats.featureImportanceMaxPpm           = uint64_t(processingInfo.stats.featureImportanceMaxPpm);

  if(loadResult != SCENE_RESULT_SUCCESS)
  {

    closeCache();

    return loadResult;
  }

  if(m_loadedFromCache)
  {

    const uint32_t assemblyCullingMinInstances = m_config.assemblyCullingMinInstances;
    const float    assemblyLodPixelThreshold   = m_config.assemblyLodPixelThreshold;
    m_cacheFileView.getSceneConfig(m_config);
    m_config.assemblyCullingMinInstances = assemblyCullingMinInstances;
    m_config.assemblyLodPixelThreshold   = assemblyLodPixelThreshold;

    m_cacheFileView.getHistograms(m_histograms);
    m_cacheFileView.getProcessingStats(m_processingStats);
  }


  m_originalInstanceCount = m_instances.size();

  m_originalGeometryCount = m_geometryViews.size();
  m_activeGeometryCount   = m_originalGeometryCount;


  computeInstanceBBoxes();
  m_gridBbox = m_bbox;

  glm::vec3 modelExtent = m_bbox.hi - m_bbox.lo;
  m_isBig = modelExtent.y < 0.15f * std::max(modelExtent.x, modelExtent.z) && m_originalInstanceCount > 1024;

  for(auto& geometry : m_geometryViews)
  {

    m_hiPerGeometryTriangles  = std::max(m_hiPerGeometryTriangles, geometry.hiTriangleCount);

    m_hiPerGeometryVertices   = std::max(m_hiPerGeometryVertices, geometry.hiVerticesCount);

    m_hiPerGeometryClusters   = std::max(m_hiPerGeometryClusters, geometry.hiClustersCount);

    m_maxPerGeometryTriangles = std::max(m_maxPerGeometryTriangles, geometry.totalTriangleCount);

    m_maxPerGeometryVertices  = std::max(m_maxPerGeometryVertices, geometry.totalVerticesCount);

    m_maxPerGeometryClusters  = std::max(m_maxPerGeometryClusters, geometry.totalClustersCount);

    m_maxClusterVertices      = std::max(m_maxClusterVertices, geometry.clusterMaxVerticesCount);

    m_maxClusterTriangles     = std::max(m_maxClusterTriangles, geometry.clusterMaxTrianglesCount);

    m_maxLodLevelsCount       = std::max(m_maxLodLevelsCount, geometry.lodLevelsCount);

    m_hiTrianglesCount += geometry.hiTriangleCount;
    m_hiClustersCount += geometry.hiClustersCount;
    m_hiVerticesCount += geometry.hiVerticesCount;
    m_totalClustersCount += geometry.totalClustersCount;
    m_totalTrianglesCount += geometry.totalTriangleCount;
    m_totalVerticesCount += geometry.totalVerticesCount;
  }
  for(size_t i = 0; i < m_instances.size(); i++)
  {
    const GeometryView& geometry = m_geometryViews[m_instances[i].geometryID];
    m_hiTrianglesCountInstanced += geometry.hiTriangleCount;
    m_hiClustersCountInstanced += geometry.hiClustersCount;
  }


  LOGI("clusters:  %" PRIu64 "\n", m_totalClustersCount);

  LOGI("triangles: %" PRIu64 "\n", m_totalTrianglesCount);
  LOGI("triangles/cluster: %.2f\n", double(m_totalTrianglesCount) / double(m_totalClustersCount));

  LOGI("vertices: %" PRIu64 "\n", m_totalVerticesCount);
  LOGI("vertices/cluster: %.2f\n", double(m_totalVerticesCount) / double(m_totalClustersCount));

  LOGI("hi clusters:  %" PRIu64 "\n", m_hiClustersCount);

  LOGI("hi triangles: %" PRIu64 "\n", m_hiTrianglesCount);

  LOGI("hi vertices: %" PRIu64 "\n", m_hiVerticesCount);
  LOGI("hi triangles/cluster: %.2f\n", double(m_hiTrianglesCount) / double(m_hiClustersCount));

  LOGI("assembly nodes: %zu, templates: %zu, min instances: %u, lod pixels: %.2f\n",
       m_assemblyNodes.size(), m_assemblyTemplates.size(), m_config.assemblyCullingMinInstances,
       double(m_config.assemblyLodPixelThreshold));

  if(!m_loadedFromCache && m_loaderConfig.autoSaveCache)
  {

    saveCache();
  }

  if(m_loadedFromCache && !m_loaderConfig.memoryMappedCache)
  {


    closeCache();
  }

  return loadResult;
}


// 函数：Scene::deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
void Scene::deinit()
{
  *this = {};
}


// 函数：Scene::updateSceneGrid。根据最新状态刷新缓存数据、GPU 地址、描述符或统计信息。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：更新函数负责把“旧状态”推进到“当前状态”，因此要避免部分更新造成 CPU/GPU 视图不一致。
void Scene::updateSceneGrid(const SceneGridConfig& gridConfig)
{
  m_gridConfig = gridConfig;


  size_t copiesCount = std::max(1u, gridConfig.numCopies);

  size_t numOldCopies = m_instances.size() / m_originalInstanceCount;


  m_instances.resize(m_originalInstanceCount * copiesCount);
  m_activeGeometryCount = gridConfig.uniqueGeometriesForCopies ? m_originalGeometryCount * copiesCount : m_originalGeometryCount;


  // 函数：rng。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  std::default_random_engine            rng(2342);


  // 函数：randomUnorm。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  std::uniform_real_distribution<float> randomUnorm(0.0f, 1.0f);

  uint32_t axis    = gridConfig.gridBits;
  size_t   sq      = 1;
  int      numAxis = 0;
  if(!axis)
    axis = 3;

  for(int i = 0; i < 3; i++)
  {
    numAxis += (axis & (1 << i)) ? 1 : 0;
  }

  switch(numAxis)
  {
    case 1:
      sq = copiesCount;
      break;
    case 2:
      while(sq * sq < copiesCount)
      {
        sq++;
      }
      break;
    case 3:
      while(sq * sq * sq < copiesCount)
      {
        sq++;
      }
      break;
  }


  size_t lastCopyIndex = 0;

  glm::vec3 modelExtent = (m_bbox.hi - m_bbox.lo);
  glm::vec3 modelCenter = (m_bbox.hi + m_bbox.lo) * 0.5f;

  float     modelSize   = glm::length(modelExtent);
  glm::vec3 gridShift;
  glm::mat4 gridRotMatrix;

  for(size_t copyIndex = 1; copyIndex < copiesCount; copyIndex++)
  {
    gridShift = gridConfig.refShift * modelExtent;
    size_t c  = copyIndex;

    float u = 0;
    float v = 0;
    float w = 0;

    switch(numAxis)
    {
      case 1:

        u = float(c);
        break;
      case 2:

        u = float(c % sq);

        v = float(c / sq);
        break;
      case 3:

        u = float(c % sq);
        v = float((c / sq) % sq);
        w = float(c / (sq * sq));
        break;
    }

    float use = u;

    if(axis & (1 << 0))
    {
      gridShift.x *= -use;
      if(numAxis > 1)
        use = v;
    }
    else
    {
      gridShift.x = 0;
    }

    if(axis & (1 << 1))
    {
      gridShift.y *= use;
      if(numAxis > 2)
        use = w;
      else if(numAxis > 1)
        use = v;
    }
    else
    {
      gridShift.y = 0;
    }

    if(axis & (1 << 2))
    {
      gridShift.z *= -use;
    }
    else
    {
      gridShift.z = 0;
    }


    glm::mat4 scaleMatrix = glm::mat4(1.0f);

    if(gridConfig.minScale != 1.0f || gridConfig.maxScale != 1.0f)
    {
      float scale = glm::mix(gridConfig.minScale, gridConfig.maxScale, randomUnorm(rng));
      scaleMatrix = glm::scale(scaleMatrix, glm::vec3(scale));

      if(scale < 1.0f)
      {
        gridShift.y += modelSize * (1.0f - scale);
      }
      else
      {
        gridShift.y -= modelSize * (scale - 1.0f);
      }
    }

    if(axis & (8 | 16 | 32))
    {
      glm::vec3 mask    = {axis & 8 ? 1.0f : 0.0f, axis & 16 ? 1.0f : 0.0f, axis & 32 ? 1.0f : 0.0f};
      glm::vec3 gridDir = glm::vec3(randomUnorm(rng), randomUnorm(rng), randomUnorm(rng));

      gridDir           = glm::max(gridDir * mask, mask * 0.00001f);
      float gridAngle   = randomUnorm(rng) * 360.0f;

      gridDir           = glm::normalize(gridDir);


      if(gridConfig.snapAngle > 0.0)
      {

        float remainder = std::fmod(gridAngle, gridConfig.snapAngle);
        gridAngle       = gridAngle - remainder;
      }


      gridAngle = gridAngle * glm::pi<float>() / 180.0f;

      gridRotMatrix = glm::rotate(glm::mat4(1), gridAngle, gridDir);
    }

    for(size_t i = 0; i < m_originalInstanceCount; i++)
    {
      Instance& instance = m_instances[i + copyIndex * m_originalInstanceCount];

      instance = m_instances[i];

      if(gridConfig.uniqueGeometriesForCopies)
      {


        instance.geometryID += uint32_t(c * m_originalGeometryCount);
      }


      glm::mat4 worldMatrix = m_instances[i].matrix;
      glm::vec3 translation = worldMatrix[3];

      worldMatrix[3]        = glm::vec4(translation - modelCenter, 1.f);

      worldMatrix = scaleMatrix * worldMatrix;

      if(axis & (8 | 16 | 32))
      {
        worldMatrix = gridRotMatrix * worldMatrix;
      }
      translation    = worldMatrix[3];

      worldMatrix[3] = glm::vec4(translation + modelCenter + gridShift, 1.f);

      instance.matrix = worldMatrix;
    }
  }

  m_gridBbox = m_bbox;

  if(copiesCount > 1 && !m_assemblyNodes.empty())
  {
    m_assemblyNodes.clear();
    m_assemblyTemplates.clear();
    m_assemblyTemplateMap.clear();
    for(auto& instance : m_instances)
    {
      instance.assemblyID = SHADERIO_INVALID_ASSEMBLY;
    }
  }

  computeInstanceBBoxes();

  std::swap(m_gridBbox, m_bbox);
}


// 函数：Scene::computeInstanceBBoxes。计算派生值，供后续剔除、LOD、统计或资源规划使用。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：计算结果通常参与阈值比较或内存规划，数值稳定性和边界条件需要特别注意。
void Scene::computeInstanceBBoxes()
{
  m_bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0, 0};

  for(auto& instance : m_instances)
  {

    const GeometryView& geometry = getActiveGeometry(instance.geometryID);

    instance.bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0, 0};

    for(uint32_t v = 0; v < 8; v++)
    {
      bool x = (v & 1) != 0;
      bool y = (v & 2) != 0;
      bool z = (v & 4) != 0;


      // 函数：weight。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
      // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
      // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
      glm::bvec3 weight(x, y, z);

      glm::vec3  corner = glm::mix(geometry.bbox.lo, geometry.bbox.hi, weight);

      corner            = instance.matrix * glm::vec4(corner, 1.0f);

      instance.bbox.lo  = glm::min(instance.bbox.lo, corner);

      instance.bbox.hi  = glm::max(instance.bbox.hi, corner);
    }


    m_bbox.lo = glm::min(m_bbox.lo, instance.bbox.lo);

    m_bbox.hi = glm::max(m_bbox.hi, instance.bbox.hi);
  }

  for(auto& assembly : m_assemblyNodes)
  {
    assembly.bbox             = makeEmptyBBox();
    const size_t first        = std::min<size_t>(assembly.firstInstance, m_instances.size());
    const size_t instanceEnd  = std::min<size_t>(m_instances.size(), size_t(assembly.firstInstance) + size_t(assembly.instanceCount));
    assembly.firstInstance    = uint32_t(first);
    assembly.instanceCount    = uint32_t(instanceEnd - first);

    for(size_t instanceIndex = first; instanceIndex < instanceEnd; instanceIndex++)
    {
      mergeBBox(assembly.bbox, m_instances[instanceIndex].bbox);
    }
  }
}


// 函数：Scene::processGeometry。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void Scene::processGeometry(ProcessingInfo& processingInfo, size_t geometryIndex, bool isCached)
{
  GeometryStorage& geometryStorage = m_geometryStorages[geometryIndex];
  GeometryView&    geometryView    = m_geometryViews[geometryIndex];

  bool viewFromStorage = true;

  if(isCached)
  {
    if(m_loaderConfig.memoryMappedCache)
    {

      m_cacheFileView.getGeometryView(geometryView, geometryIndex);

      viewFromStorage = false;
    }
    else
    {

      loadCachedGeometry(geometryStorage, geometryIndex);
    }
  }
  else
  {
    if(geometryStorage.triangles.empty())
    {
      geometryStorage = {};
    }
    else
    {


      geometryStorage.lodInfo.inputTriangleCount       = geometryStorage.triangles.size();

      geometryStorage.lodInfo.inputVertexCount         = geometryStorage.vertexPositions.size();
      geometryStorage.lodInfo.inputTriangleIndicesHash = 0;
      geometryStorage.lodInfo.inputVerticesHash        = 0;


      size_t originalVertexCount = geometryStorage.vertexPositions.size();


      if(geometryStorage.vertexPositions.size() >= (geometryStorage.triangles.size() + geometryStorage.triangles.size() / 2))
      {

        buildGeometryDedupVertices(processingInfo, geometryStorage);
      }


      buildGeometryLod(processingInfo, geometryStorage);
    }
  }

  if(viewFromStorage)
  {
    (GeometryBase&)geometryView = geometryStorage;

    geometryView.groupData        = geometryStorage.groupData;
    geometryView.groupInfos       = geometryStorage.groupInfos;
    geometryView.lodLevels        = geometryStorage.lodLevels;
    geometryView.lodNodes         = geometryStorage.lodNodes;
    geometryView.lodNodeBboxes    = geometryStorage.lodNodeBboxes;
    geometryView.localMaterialIDs = geometryStorage.localMaterialIDs;
  }


  geometryView.instanceReferenceCount = 0;

  if(m_processingOnlyFile)
  {

    saveProcessingOnly(processingInfo, geometryIndex);
  }
}


// 函数：Scene::computeLodBboxes_recursive。计算派生值，供后续剔除、LOD、统计或资源规划使用。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：计算结果通常参与阈值比较或内存规划，数值稳定性和边界条件需要特别注意。
void Scene::computeLodBboxes_recursive(GeometryStorage& geometry, size_t i)
{
  const shaderio::Node& node = geometry.lodNodes[i];
  shaderio::BBox&       bbox = geometry.lodNodeBboxes[i];

  bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0.0f, 0.0f};

  if(node.groupRange.isGroup)
  {
    GroupInfo groupInfo = geometry.groupInfos[node.groupRange.groupIndex];


    // 函数：groupView。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    GroupView groupView(geometry.groupData, groupInfo);

    for(uint32_t c = 0; c < groupInfo.clusterCount; c++)
    {

      bbox.lo = glm::min(bbox.lo, groupView.clusterBboxes[c].lo);

      bbox.hi = glm::max(bbox.hi, groupView.clusterBboxes[c].hi);
    }
  }
  else
  {
    ((std::atomic_uint32_t&)m_histograms.nodeChildren[node.nodeRange.childCountMinusOne + 1])++;

    for(uint32_t n = 0; n <= node.nodeRange.childCountMinusOne; n++)
    {

      computeLodBboxes_recursive(geometry, node.nodeRange.childOffset + n);
    }

    for(uint32_t n = 0; n <= node.nodeRange.childCountMinusOne; n++)
    {

      bbox.lo = glm::min(bbox.lo, geometry.lodNodeBboxes[node.nodeRange.childOffset + n].lo);

      bbox.hi = glm::max(bbox.hi, geometry.lodNodeBboxes[node.nodeRange.childOffset + n].hi);
    }
  }
}


// 结构：HashVertexRange。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct HashVertexRange
{
  uint32_t offset = 0;
  uint32_t count  = 0;
};

static_assert(std::atomic_uint32_t::is_always_lock_free && sizeof(std::atomic_uint32_t) == sizeof(uint32_t));


// 函数：Scene::buildGeometryDedupVertices。构建派生数据结构，通常用于 LOD、层次结构、间接命令或加速访问。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：构建结果会被后续阶段高频读取，必须保证布局紧凑、索引合法并与共享结构定义一致。
void Scene::buildGeometryDedupVertices(ProcessingInfo& processingInfo, GeometryStorage& geometry)
{


  // 函数：remap。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  std::vector<uint32_t> remap(geometry.vertexPositions.size());

  size_t uniqueVertices = 0;


  size_t attributeStride = geometry.vertexAttributes.size() / geometry.vertexPositions.size();

  if(geometry.attributeBits)
  {
    uint32_t       texOffset = 0;
    meshopt_Stream streams[4];
    uint32_t       streamCount = 1;


    streams[0].data   = geometry.vertexPositions.data();
    streams[0].size   = sizeof(float) * 3;
    streams[0].stride = sizeof(glm::vec3);
    if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL)
    {

      streams[1].data   = geometry.vertexAttributes.data();
      streams[1].size   = sizeof(float) * 3;
      streams[1].stride = sizeof(float) * attributeStride;
      streamCount++;
      texOffset = 3;
    }
    if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0)
    {
      streams[streamCount].data   = geometry.vertexAttributes.data() + texOffset;
      streams[streamCount].size   = sizeof(float) * 2;
      streams[streamCount].stride = sizeof(float) * attributeStride;
      streamCount++;
    }
    if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT)
    {
      streams[streamCount].data   = geometry.vertexAttributes.data() + texOffset + 2;
      streams[streamCount].size   = sizeof(float) * 4;
      streams[streamCount].stride = sizeof(float) * attributeStride;
      streamCount++;
    }

    uniqueVertices =
        meshopt_generateVertexRemapMulti(remap.data(), reinterpret_cast<const uint32_t*>(geometry.triangles.data()),
                                         geometry.triangles.size() * 3, geometry.vertexPositions.size(), streams, streamCount);
  }
  else
  {
    uniqueVertices = meshopt_generateVertexRemap(remap.data(), reinterpret_cast<const uint32_t*>(geometry.triangles.data()),
                                                 geometry.triangles.size() * 3, geometry.vertexPositions.data(),
                                                 geometry.vertexPositions.size(), sizeof(glm::vec3));
  }

  {


    // 函数：newPositions。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    std::vector<glm::vec3> newPositions(uniqueVertices);
    meshopt_remapVertexBuffer(newPositions.data(), geometry.vertexPositions.data(), geometry.vertexPositions.size(),
                              sizeof(glm::vec3), remap.data());

    geometry.vertexPositions = std::move(newPositions);
  }

  if(geometry.attributeBits)
  {


    // 函数：newAttributes。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    std::vector<float> newAttributes(uniqueVertices * attributeStride);
    meshopt_remapVertexBuffer(newAttributes.data(), geometry.vertexAttributes.data(), geometry.vertexPositions.size(),
                              sizeof(float) * attributeStride, remap.data());

    geometry.vertexAttributes = std::move(newAttributes);
  }

  meshopt_remapIndexBuffer(reinterpret_cast<uint32_t*>(geometry.triangles.data()),
                           reinterpret_cast<uint32_t*>(geometry.triangles.data()), geometry.triangles.size() * 3, remap.data());
}


// 函数：Scene::computeHistogramMaxs。计算派生值，供后续剔除、LOD、统计或资源规划使用。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：计算结果通常参与阈值比较或内存规划，数值稳定性和边界条件需要特别注意。
void Scene::computeHistogramMaxs()
{
  m_histograms.clusterTrianglesMax = 0u;
  m_histograms.clusterVerticesMax  = 0u;
  m_histograms.groupClustersMax    = 0u;
  m_histograms.nodeChildrenMax     = 0u;
  m_histograms.lodLevelsMax        = 0u;

  for(size_t i = 0; i < m_histograms.clusterTriangles.size(); i++)
  {

    m_histograms.clusterTrianglesMax = std::max(m_histograms.clusterTrianglesMax, m_histograms.clusterTriangles[i]);
  }
  for(size_t i = 0; i < m_histograms.clusterVertices.size(); i++)
  {

    m_histograms.clusterVerticesMax = std::max(m_histograms.clusterVerticesMax, m_histograms.clusterVertices[i]);
  }

  for(size_t i = 0; i < m_histograms.groupClusters.size(); i++)
  {

    m_histograms.groupClustersMax = std::max(m_histograms.groupClustersMax, m_histograms.groupClusters[i]);
  }

  for(size_t i = 0; i < m_histograms.nodeChildren.size(); i++)
  {

    m_histograms.nodeChildrenMax = std::max(m_histograms.nodeChildrenMax, m_histograms.nodeChildren[i]);
  }

  for(size_t i = 0; i < m_histograms.lodLevels.size(); i++)
  {

    m_histograms.lodLevelsMax = std::max(m_histograms.lodLevelsMax, m_histograms.lodLevels[i]);
  }
}
}
