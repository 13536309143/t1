//==============================================================================
// 文件：src/scene/scene.hpp
// 模块定位：CPU 侧场景核心数据结构声明，定义配置、几何体、簇组、实例、相机、统计和缓存处理接口。
// 数据流：输入来自 glTF primitive 和构建参数；输出是可被预加载或流式上传模块消费的 GeometryView 与 GroupView。
// 方法说明：Scene 将语义层 mesh 拆分为渲染层 簇/组/LOD hierarchy，形成 CPU 构建与 GPU 遍历之间的契约。
// 正确性约束：GeometryView 中所有 span 必须指向稳定存储；GroupInfo 的偏移和大小必须与 着色器io 布局一致；实例 bbox 需覆盖变换后的几何体。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#pragma once


// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <vector>
#include <array>
#include <string>
#include <atomic>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <glm/glm.hpp>
#include <nvutils/file_mapping.hpp>
#include <nvutils/timers.hpp>
#include <nvutils/alignment.hpp>
#include "serialization.hpp"
#include "meshlod.h"
#include "shaderio_scene.h"


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace lodclusters {


// 结构：SceneConfig。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct SceneConfig
{
  static const uint32_t version = 8;


  uint32_t clusterVertices    = 128;
  uint32_t clusterTriangles   = 128;
  uint32_t clusterGroupSize   = 32;
  uint32_t preferredNodeWidth = 8;


  bool useCompressedData = false;


  uint32_t enabledAttributes = shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL;


  float meshoptFillWeight  = 0.5f;
  float meshoptSplitFactor = 2.0f;


  float lodLevelDecimationFactor = 0.5f;


  float lodErrorMergePrevious = 1.5;
  float lodErrorMergeAdditive = 0.0f;


  float simplifyNormalWeight      = 0.5f;
  float simplifyTangentWeight     = 0.01f;
  float simplifyTangentSignWeight = 0.5f;
  float simplifyTexCoordWeight    = 0;


  uint32_t compressionPosDropBits = 7;
  uint32_t compressionTexDropBits = 7;


  float lodErrorEdgeLimit = 0;


  uint32_t assemblyCullingMinInstances = 8;
  float    assemblyLodPixelThreshold   = 24.0f;//

  bool  featureConstraints        = true;
  float featureImportanceWeight   = 4.0f;
  float featureProtectThreshold   = 0.78f;
  float featureCriticalThreshold  = 0.93f;

  uint32_t reservedData[12] = {};


};


// 结构：SceneLoaderConfig。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct SceneLoaderConfig
{


  float processingThreadsPct = 0.5;


  bool processingOnly = false;

  bool processingAllowPartial = false;

  int processingMode = 0;


  bool autoSaveCache = true;

  bool autoLoadCache = true;


  bool memoryMappedCache = false;


  size_t forcePreprocessMiB = size_t(2) * 1024;


  std::atomic_uint32_t* progressPct = nullptr;
};


// 结构：SceneGridConfig。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct SceneGridConfig
{
  static const uint32_t minCopies = 1;
  static const uint32_t maxCopies = 32;


  bool      uniqueGeometriesForCopies = false;
  uint32_t  numCopies                 = 1;
  uint32_t  gridBits                  = 13;
  glm::vec3 refShift                  = {1.0f, 1.0f, 1.0f};
  float     snapAngle                 = 0;
  float     minScale                  = 1.0f;
  float     maxScale                  = 1.0f;
};


// 类型：Scene。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class Scene
{
public:


  // 枚举：Result。集中定义本模块可选模式或状态值，避免调用点使用裸整数。
  // 设计意图：把实验开关、渲染模式或阶段编号显式命名，使配置文件、UI 和代码路径可以互相对应。
  // 使用约束：新增枚举值时需要同步 UI 文本、参数解析和相关 switch 分支。
  enum Result
  {
    SCENE_RESULT_SUCCESS,
    SCENE_RESULT_CACHE_INVALID,
    SCENE_RESULT_NEEDS_PREPROCESS,
    SCENE_RESULT_PREPROCESS_COMPLETED,
    SCENE_RESULT_ERROR,
  };

  Result init(const std::filesystem::path& filePath,
              const SceneConfig&           config,
              const SceneLoaderConfig&     loaderConfig,
              const std::string&           cacheSuffix,
              bool                         skipCache);


  // 函数：saveCache。把当前状态写入缓存、缓冲、文件或着色器可消费的数据布局。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：写入路径应明确字节对齐、所有权和可见性，避免后续读取端解释错误。
  bool   saveCache() const;


  // 函数：deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void   deinit();


  // 函数：updateSceneGrid。根据最新状态刷新缓存数据、GPU 地址、描述符或统计信息。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：更新函数负责把“旧状态”推进到“当前状态”，因此要避免部分更新造成 CPU/GPU 视图不一致。
  void updateSceneGrid(const SceneGridConfig& gridConfig);

  bool isMemoryMappedCache() const { return m_loadedFromCache && m_cacheFileMapping.valid(); }

  const std::filesystem::path& getFilePath() const { return m_filePath; }
  const std::filesystem::path& getCacheFilePath() const { return m_cacheFilePath; }


  // 结构：Range。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct Range
  {
    uint32_t offset;
    uint32_t count;
  };


  // 结构：GroupInfo。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct GroupInfo
  {
    uint64_t offsetBytes : 42;
    uint64_t sizeBytes : 22;
    uint16_t vertexCount;
    uint16_t triangleCount;
    uint8_t  lodLevel;
    uint8_t  clusterCount;
    uint8_t  attributeBits;
    uint8_t  reserved1 = 0;
    uint64_t vertexDataCount : 21;


    uint64_t uncompressedVertexDataCount : 21;
    uint64_t uncompressedSizeBytes : 22;


    uint32_t getDeviceSize() const { return uint32_t(uncompressedSizeBytes ? uncompressedSizeBytes : sizeBytes); }


    // 函数：estimateVertexDataCount。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    uint32_t estimateVertexDataCount() const
    {
      uint32_t dataCount = vertexCount * 3;
      if(attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL)
      {
        dataCount += vertexCount * 1;
      }
      if(attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0)
      {
        dataCount += vertexCount * 2;
        dataCount += clusterCount;
      }
      if(attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1)
      {
        dataCount += vertexCount * 2;
        dataCount += clusterCount;
      }
      return dataCount;
    }


    // 函数：computeSize。计算派生值，供后续剔除、LOD、统计或资源规划使用。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：计算结果通常参与阈值比较或内存规划，数值稳定性和边界条件需要特别注意。
    size_t computeSize() const
    {
      size_t threadGroupSize = sizeof(shaderio::Group);
      threadGroupSize        = nvutils::align_up(threadGroupSize, 16) + sizeof(shaderio::Cluster) * clusterCount;
      threadGroupSize        = nvutils::align_up(threadGroupSize, 4) + sizeof(uint32_t) * clusterCount;
      threadGroupSize        = nvutils::align_up(threadGroupSize, 16) + sizeof(shaderio::BBox) * clusterCount;
      threadGroupSize        = threadGroupSize + sizeof(uint8_t) * triangleCount * 3;
      threadGroupSize        = nvutils::align_up(threadGroupSize, 8) + sizeof(float) * vertexDataCount;
      return nvutils::align_up(threadGroupSize, 16);
    }


    // 函数：computeUncompressedSectionSize。执行压缩或解压流程，在体积和运行时访问格式之间做转换。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：压缩必须保留可验证的重建语义；当压缩收益不足或超出约束时应回退到未压缩表示。
    size_t computeUncompressedSectionSize() const
    {
      size_t threadGroupSize = sizeof(shaderio::Group);
      threadGroupSize        = nvutils::align_up(threadGroupSize, 16) + sizeof(shaderio::Cluster) * clusterCount;
      threadGroupSize        = nvutils::align_up(threadGroupSize, 4) + sizeof(uint32_t) * clusterCount;
      threadGroupSize        = nvutils::align_up(threadGroupSize, 16) + sizeof(shaderio::BBox) * clusterCount;
      threadGroupSize        = threadGroupSize + sizeof(uint8_t) * triangleCount * 3;

      threadGroupSize        = nvutils::align_up(threadGroupSize, 8);
      return threadGroupSize;
    }
  };


  // 结构：GroupView。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct GroupView
  {
    const uint8_t*                     raw     = nullptr;
    const size_t                       rawSize = 0;
    const shaderio::Group*             group   = nullptr;
    std::span<const shaderio::Cluster> clusters;
    std::span<const uint32_t>          clusterGeneratingGroups;
    std::span<const shaderio::BBox>    clusterBboxes;
    std::span<const uint8_t>           indices;
    std::span<const float>             vertices;

    GroupView() {};


    GroupView(std::span<const uint8_t> groupDatas, const GroupInfo& info)

        : rawSize(info.sizeBytes)
    {
      assert(info.offsetBytes + info.sizeBytes <= groupDatas.size());
      raw = &groupDatas[info.offsetBytes];


      size_t startAddress = size_t(raw);

      group = (const shaderio::Group*)raw;
      clusters = std::span((const shaderio::Cluster*)nvutils::align_up(startAddress + sizeof(shaderio::Group), 16), info.clusterCount);
      clusterGeneratingGroups =
          std::span((const uint32_t*)nvutils::align_up(size_t(clusters.data() + info.clusterCount), 4), info.clusterCount);
      clusterBboxes =
          std::span((const shaderio::BBox*)nvutils::align_up(size_t(clusterGeneratingGroups.data() + info.clusterCount), 16),
                    info.clusterCount);

      indices = std::span((const uint8_t*)size_t(clusterBboxes.data() + info.clusterCount), info.triangleCount * 3);

      vertices = std::span((const float*)nvutils::align_up(size_t(indices.data() + info.triangleCount * 3), 8), info.vertexDataCount);
      assert((size_t(vertices.data() + info.vertexDataCount) - startAddress) <= size_t(info.sizeBytes));
    }


    // 函数：getClusterIndices。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    const uint8_t* getClusterIndices(size_t clusterIndex) const
    {

      return (const uint8_t*)(size_t(&clusters[clusterIndex]) + clusters[clusterIndex].indices);
    }


    // 函数：getClusterVertices。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    const glm::vec3* getClusterVertices(size_t clusterIndex) const
    {

      return (const glm::vec3*)(size_t(&clusters[clusterIndex]) + clusters[clusterIndex].vertices);
    }
  };


  // 结构：GroupStorage。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct GroupStorage
  {
    uint8_t*                     raw;
    const size_t                 rawSize = 0;
    shaderio::Group*             group;
    std::span<shaderio::Cluster> clusters;
    std::span<uint32_t>          clusterGeneratingGroups;
    std::span<shaderio::BBox>    clusterBboxes;
    std::span<uint8_t>           indices;
    std::span<float>             vertices;

    GroupStorage() {};


    GroupStorage(void* groupData, const GroupInfo& info)

        : rawSize(info.sizeBytes)
    {
      size_t startAddress = (size_t)groupData;

      raw   = (uint8_t*)groupData;
      group = (shaderio::Group*)startAddress;
      clusters = std::span((shaderio::Cluster*)nvutils::align_up(startAddress + sizeof(shaderio::Group), 16), info.clusterCount);
      clusterGeneratingGroups =
          std::span((uint32_t*)nvutils::align_up(size_t(clusters.data() + info.clusterCount), 4), info.clusterCount);
      clusterBboxes =
          std::span((shaderio::BBox*)nvutils::align_up(size_t(clusterGeneratingGroups.data() + info.clusterCount), 16),
                    info.clusterCount);
      indices = std::span((uint8_t*)size_t(clusterBboxes.data() + info.clusterCount), info.triangleCount * 3);
      vertices = std::span((float*)nvutils::align_up(size_t(indices.data() + info.triangleCount * 3), 8), info.vertexDataCount);
      assert((size_t(vertices.data() + info.vertexDataCount) - startAddress) <= size_t(info.sizeBytes));
    }


    uint32_t getClusterLocalOffset(uint32_t clusterIndex, const void* input, size_t overrideSize = 0) const
    {
      assert(size_t(input) >= size_t(&clusters[clusterIndex]));
      assert(size_t(input) < size_t(raw + (overrideSize ? overrideSize : rawSize)));

      return uint32_t(size_t(input) - size_t(&clusters[clusterIndex]));
    }


    // 函数：getClusterLocalData。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    uint32_t* getClusterLocalData(uint32_t clusterIndex, uint32_t localOffset)
    {
      return (uint32_t*)(size_t(&clusters[clusterIndex]) + localOffset);
    }
  };


  static void fillGroupRuntimeData(const GroupInfo& srcGroupInfo,
                                   const GroupView& srcGroupView,
                                   uint32_t         groupID,
                                   uint32_t         groupResidentID,
                                   uint32_t         clusterResidentID,
                                   void*            dst,
                                   size_t           dstSize);


  // 函数：decompressGroup。执行压缩或解压流程，在体积和运行时访问格式之间做转换。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：压缩必须保留可验证的重建语义；当压缩收益不足或超出约束时应回退到未压缩表示。
  static void decompressGroup(const GroupInfo& info, const GroupView& groupView, void* dstWriteOnly, size_t dstSize);


  // 结构：GeometryLodInput。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct GeometryLodInput
  {
    uint64_t inputTriangleCount       = 0;
    uint64_t inputVertexCount         = 0;
    uint64_t inputTriangleIndicesHash = 0;
    uint64_t inputVerticesHash        = 0;
  };


  // 结构：GeometryBase。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct GeometryBase
  {
    uint32_t attributeBits = 0;

    uint32_t clusterMaxVerticesCount{};
    uint32_t clusterMaxTrianglesCount{};

    uint32_t lodLevelsCount{};


    uint32_t hiTriangleCount{};
    uint32_t hiVerticesCount{};
    uint32_t hiClustersCount{};


    uint32_t totalTriangleCount{};
    uint32_t totalVerticesCount{};
    uint32_t totalClustersCount{};

    shaderio::BBox bbox{};

    GeometryLodInput lodInfo;

    uint32_t instanceReferenceCount{};
  };


  // 结构：GeometryView。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct GeometryView : GeometryBase
  {

    std::span<const uint8_t> groupData;


    std::span<const GroupInfo> groupInfos;

    std::span<const shaderio::LodLevel> lodLevels;
    std::span<const shaderio::Node>     lodNodes;
    std::span<const shaderio::BBox>     lodNodeBboxes;


    std::span<const uint32_t> localMaterialIDs;


    // 函数：getCachedSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    inline uint64_t getCachedSize() const
    {
      uint64_t cachedSize = 0;

      cachedSize += (sizeof(GeometryBase) + serialization::ALIGN_MASK) & ~serialization::ALIGN_MASK;

      cachedSize += serialization::getCachedSize(groupData);

      cachedSize += serialization::getCachedSize(groupInfos);

      cachedSize += serialization::getCachedSize(lodLevels);

      cachedSize += serialization::getCachedSize(lodNodes);

      cachedSize += serialization::getCachedSize(lodNodeBboxes);

      cachedSize += serialization::getCachedSize(localMaterialIDs);

      return cachedSize;
    }
  };


  const GeometryView& getActiveGeometry(size_t idx) const { return m_geometryViews[idx % m_originalGeometryCount]; }
  size_t              getActiveGeometryCount() const { return m_activeGeometryCount; }


  // 函数：getGeometryInstanceFactor。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  uint32_t getGeometryInstanceFactor() const
  {
    return m_gridConfig.uniqueGeometriesForCopies ? 1u : uint32_t(m_instances.size() / m_originalInstanceCount);
  }


  // 结构：Instance。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct Instance
  {
    glm::mat4      matrix;
    shaderio::BBox bbox;
    uint32_t       geometryID = ~0U;
    uint32_t       materialID = ~0U;
    uint32_t       assemblyID = SHADERIO_INVALID_ASSEMBLY;
    bool           twoSided   = false;
    glm::vec4      color{0.8, 0.8, 0.8, 1.0f};
  };

  struct GltfNodeImportResult
  {
    uint32_t       firstInstance = 0;
    uint32_t       instanceCount = 0;
    shaderio::BBox bbox          = {};
  };

  struct AssemblyTemplate
  {
    uint64_t fingerprint        = 0;
    uint32_t firstAssembly      = SHADERIO_INVALID_ASSEMBLY;
    uint32_t assemblyCount      = 0;
    uint32_t instanceCount      = 0;
  };


  // 结构：Camera。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct Camera
  {
    glm::mat4 worldMatrix{1};
    glm::vec3 eye{0, 0, 0};
    glm::vec3 center{0, 0, 0};
    glm::vec3 up{0, 1, 0};
    float     fovy;
  };


  // 结构：Histograms。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct Histograms
  {
    static const uint32_t version = 1;

    std::array<uint32_t, 256 + 1>                         clusterTriangles = {};
    std::array<uint32_t, 256 + 1>                         clusterVertices  = {};
    std::array<uint32_t, SHADERIO_MAX_GROUP_CLUSTERS + 1> groupClusters    = {};
    std::array<uint32_t, SHADERIO_MAX_NODE_CHILDREN + 1>  nodeChildren     = {};
    std::array<uint32_t, SHADERIO_MAX_LOD_LEVELS + 1>     lodLevels        = {};

    uint32_t clusterTrianglesMax = {};
    uint32_t clusterVerticesMax  = {};
    uint32_t groupClustersMax    = {};
    uint32_t nodeChildrenMax     = {};
    uint32_t lodLevelsMax        = {};
  };

  struct ProcessingStatsSnapshot
  {
    static const uint32_t version = 2;

    uint64_t groups                = 0;
    uint64_t clusters              = 0;
    uint64_t vertices              = 0;
    uint64_t groupUniqueVertices   = 0;
    uint64_t groupHeaderBytes      = 0;
    uint64_t triangleIndexBytes    = 0;
    uint64_t vertexPosBytes        = 0;
    uint64_t vertexTexCoordBytes   = 0;
    uint64_t vertexNrmBytes        = 0;
    uint64_t vertexCompressedBytes = 0;
    uint64_t clusterBboxBytes      = 0;
    uint64_t clusterHeaderBytes    = 0;
    uint64_t clusterGenBytes       = 0;
    uint64_t inputFeatureVertices  = 0;
    uint64_t inputFeatureTris      = 0;
    uint64_t boundaryVertices      = 0;
    uint64_t nonManifoldVertices   = 0;
    uint64_t sharpEdgeVertices     = 0;
    uint64_t boundaryComponents    = 0;
    uint64_t sharpRingComponents   = 0;
    uint64_t circularHoleLoops     = 0;
    uint64_t circularHoleVertices  = 0;
    uint64_t functionalBoundaryVertices = 0;
    uint64_t cylindricalVertices   = 0;
    uint64_t thinWallVertices      = 0;
    uint64_t protectedVertices     = 0;
    uint64_t criticalVertices      = 0;
    uint64_t featureImportanceSumPpm = 0;
    uint64_t featureImportanceMaxPpm = 0;

  };


  SceneConfig       m_config;
  SceneLoaderConfig m_loaderConfig;
  SceneGridConfig   m_gridConfig;
  std::string       m_cacheSuffix;

  shaderio::BBox m_bbox;
  shaderio::BBox m_gridBbox;

  std::vector<Instance> m_instances;
  std::vector<shaderio::AssemblyNode> m_assemblyNodes;
  std::vector<AssemblyTemplate>       m_assemblyTemplates;
  std::vector<Camera>   m_cameras;

  bool m_isBig       = false;
  bool m_hasTwoSided = false;

  uint32_t m_maxClusterTriangles       = 0;
  uint32_t m_maxClusterVertices        = 0;
  uint32_t m_maxPerGeometryClusters    = 0;
  uint32_t m_maxPerGeometryTriangles   = 0;
  uint32_t m_maxPerGeometryVertices    = 0;
  uint32_t m_maxLodLevelsCount         = 0;
  uint32_t m_hiPerGeometryClusters     = 0;
  uint32_t m_hiPerGeometryTriangles    = 0;
  uint32_t m_hiPerGeometryVertices     = 0;
  uint64_t m_hiClustersCount           = 0;
  uint64_t m_hiVerticesCount           = 0;
  uint64_t m_hiTrianglesCount          = 0;
  uint64_t m_hiClustersCountInstanced  = 0;
  uint64_t m_hiTrianglesCountInstanced = 0;
  uint64_t m_totalClustersCount        = 0;
  uint64_t m_totalTrianglesCount       = 0;
  uint64_t m_totalVerticesCount        = 0;

  Histograms m_histograms;
  ProcessingStatsSnapshot m_processingStats;

  bool m_loadedFromCache    = false;
  bool m_hasVertexNormals   = false;
  bool m_hasVertexTexCoord0 = false;
  bool m_hasVertexTexCoord1 = false;
  bool m_hasVertexTangents  = false;

  size_t m_originalInstanceCount = 0;
  size_t m_originalGeometryCount = 0;

  size_t m_cacheFileSize = 0;

private:


  // 结构：GeometryStorage。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct GeometryStorage : GeometryBase
  {

    std::vector<glm::vec3>  vertexPositions;
    std::vector<float>      vertexAttributes;
    std::vector<glm::uvec3> triangles;

    uint32_t attributesWithWeights  = 0u;
    uint32_t attributeNormalOffset  = ~0u;
    uint32_t attributeTex0offset    = ~0u;
    uint32_t attributeTex1offset    = ~0u;
    uint32_t attributeTangentOffset = ~0u;


    std::vector<uint8_t>   groupData;
    std::vector<GroupInfo> groupInfos;

    std::vector<shaderio::LodLevel> lodLevels;
    std::vector<shaderio::BBox>     lodNodeBboxes;
    std::vector<shaderio::Node>     lodNodes;

    std::vector<uint32_t> localMaterialIDs;
  };

  size_t m_activeGeometryCount = 0;

  std::vector<GeometryStorage> m_geometryStorages;
  std::vector<GeometryView>    m_geometryViews;
  std::unordered_map<uint64_t, uint32_t> m_assemblyTemplateMap;


  // 函数：loadCached。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
  static bool     loadCached(GeometryView& view, uint64_t dataSize, const void* data);


  // 函数：storeCached。把当前状态写入缓存、缓冲、文件或着色器可消费的数据布局。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：写入路径应明确字节对齐、所有权和可见性，避免后续读取端解释错误。
  static bool     storeCached(const GeometryView& view, uint64_t dataSize, void* data);


  // 函数：storeCached。把当前状态写入缓存、缓冲、文件或着色器可消费的数据布局。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：写入路径应明确字节对齐、所有权和可见性，避免后续读取端解释错误。
  static uint64_t storeCached(const GeometryView& view, FILE* outFile);


  // 函数：openCache。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void openCache();


  // 函数：closeCache。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void closeCache();


  // 函数：checkCache。返回条件判断结果，用于调用方选择后续分支或验证输入状态。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：谓词函数应保持无副作用或低副作用，使调用方可以安全地把它用于断言、过滤和早退。
  bool checkCache(const GeometryLodInput& info, size_t geometryIndex);


  // 函数：loadCachedGeometry。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
  void loadCachedGeometry(GeometryStorage& geometry, size_t geometryIndex);


  // 类型：CacheFileHeader。封装本模块的长期状态、资源所有权和对外操作接口。
  // 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
  // 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
  class CacheFileHeader
  {
  public:

    CacheFileHeader()
    {
      memset(this, 0, sizeof(CacheFileHeader));
      header = {};
      config = {};
    }


    // 函数：isValid。返回条件判断结果，用于调用方选择后续分支或验证输入状态。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：谓词函数应保持无副作用或低副作用，使调用方可以安全地把它用于断言、过滤和早退。
    bool isValid() const
    {
      Header reference = {};

      return header.magic == reference.magic && header.geoVersion == reference.geoVersion
             && header.geoStructSize == reference.geoStructSize && header.configStructSize == reference.configStructSize
             && header.configVersion == reference.configVersion && header.histogramsVersion == reference.histogramsVersion
             && header.histogramStructSize == reference.histogramStructSize
             && header.processingStatsVersion == reference.processingStatsVersion
             && header.processingStatsStructSize == reference.processingStatsStructSize && header.alignment == reference.alignment;
    }

  private:


    // 结构：Header。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
    // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
    // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
    struct Header
    {
      uint64_t magic               = 0x006f65676e73766eULL;
      uint32_t geoVersion          = 11;
      uint32_t geoStructSize       = uint32_t(sizeof(GeometryView));
      uint32_t configVersion       = SceneConfig::version;
      uint32_t configStructSize    = uint32_t(sizeof(SceneConfig));
      uint32_t histogramsVersion   = Histograms::version;
      uint32_t histogramStructSize = uint32_t(sizeof(Histograms));
      uint32_t processingStatsVersion = ProcessingStatsSnapshot::version;
      uint32_t processingStatsStructSize = uint32_t(sizeof(ProcessingStatsSnapshot));
      uint64_t alignment           = serialization::ALIGNMENT;


    };

    Header header;

  public:
    SceneConfig             config;
    Histograms              histograms;
    ProcessingStatsSnapshot processingStats;
    uint32_t                pad[7];
  };

  static_assert(sizeof(CacheFileHeader) % serialization::ALIGNMENT == 0, "CacheFileHeader size unaligned");


  // 类型：CacheFileView。封装本模块的长期状态、资源所有权和对外操作接口。
  // 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
  // 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
  class CacheFileView
  {


#if 0


    // 结构：CacheFile。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
    // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
    // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
    struct CacheFile
    {

      CacheHeader header;

      uint8_t geometryViewData[];


      uint64_t geometryOffsets[geometryCount * 2];
      uint64_t geometryCount;
    };
#endif

  public:
    bool isValid() const { return m_dataSize != 0; }


    // 函数：init。初始化本模块所需状态、资源或 GPU 侧绑定。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
    bool init(uint64_t dataSize, const void* data);

    void deinit() { *(this) = {}; }

    uint64_t getGeometryCount() const { return m_geometryCount; }


    // 函数：getSceneConfig。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    void getSceneConfig(SceneConfig& settings) const;


    // 函数：getHistograms。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    void getHistograms(Histograms& histograms) const;

    void getProcessingStats(ProcessingStatsSnapshot& stats) const;


    // 函数：getGeometryView。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    bool getGeometryView(GeometryView& view, uint64_t geometryIndex) const;

  private:
    template <class T>

    const T* getPointer(uint64_t offset, uint64_t count = 1) const
    {
      assert(offset + sizeof(T) * count <= m_dataSize);
      return reinterpret_cast<const T*>(m_dataBytes + offset);
    }

    uint64_t       m_dataSize      = 0;
    uint64_t       m_tableStart    = 0;
    const uint8_t* m_dataBytes     = nullptr;
    uint64_t       m_geometryCount = 0;
  };


  // 结构：CachePartialEntry。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct CachePartialEntry
  {
    uint64_t geometryIndex = 0;
    uint64_t offset        = 0;
    uint64_t dataSize      = 0;
  };

  std::filesystem::path m_filePath;
  std::filesystem::path m_cacheFilePath;
  std::filesystem::path m_cachePartialFilePath;


  nvutils::FileReadMapping m_cacheFileMapping;
  CacheFileView            m_cacheFileView;


  FILE*                 m_processingOnlyFile             = nullptr;
  FILE*                 m_processingOnlyPartialFile      = nullptr;
  size_t                m_processingOnlyPartialCompleted = 0;
  uint64_t              m_processingOnlyFileOffset       = 0;
  std::vector<uint64_t> m_processingOnlyGeometryOffsets;


  // 结构：ProcessingInfo。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct ProcessingInfo
  {


    uint32_t numPoolThreadsOriginal = 1;
    uint32_t numPoolThreads         = 1;

    uint32_t numOuterThreads = 1;
    uint32_t numInnerThreads = 1;


    size_t   geometryCount = 0;
    uint64_t triangleCount = 0;

    std::mutex processOnlySaveMutex;


    std::vector<uint32_t> bufferViewUsers;
    std::vector<uint32_t> bufferViewLocks;


    // 结构：Stats。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
    // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
    // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
    struct Stats
    {
      std::atomic_uint64_t groups                = 0;
      std::atomic_uint64_t clusters              = 0;
      std::atomic_uint64_t vertices              = 0;
      std::atomic_uint64_t groupUniqueVertices   = 0;
      std::atomic_uint64_t groupHeaderBytes      = 0;
      std::atomic_uint64_t triangleIndexBytes    = 0;
      std::atomic_uint64_t vertexPosBytes        = 0;
      std::atomic_uint64_t vertexTexCoordBytes   = 0;
      std::atomic_uint64_t vertexNrmBytes        = 0;
      std::atomic_uint64_t vertexCompressedBytes = 0;
      std::atomic_uint64_t clusterBboxBytes      = 0;
      std::atomic_uint64_t clusterHeaderBytes    = 0;
      std::atomic_uint64_t clusterGenBytes       = 0;
      std::atomic_uint64_t inputFeatureVertices  = 0;
      std::atomic_uint64_t inputFeatureTris      = 0;
      std::atomic_uint64_t boundaryVertices      = 0;
      std::atomic_uint64_t nonManifoldVertices   = 0;
      std::atomic_uint64_t sharpEdgeVertices     = 0;
      std::atomic_uint64_t boundaryComponents    = 0;
      std::atomic_uint64_t sharpRingComponents   = 0;
      std::atomic_uint64_t circularHoleLoops     = 0;
      std::atomic_uint64_t circularHoleVertices  = 0;
      std::atomic_uint64_t functionalBoundaryVertices = 0;
      std::atomic_uint64_t cylindricalVertices   = 0;
      std::atomic_uint64_t thinWallVertices      = 0;
      std::atomic_uint64_t protectedVertices     = 0;
      std::atomic_uint64_t criticalVertices      = 0;
      std::atomic_uint64_t featureImportanceSumPpm = 0;
      std::atomic_uint64_t featureImportanceMaxPpm = 0;
    } stats;


    uint32_t   progressLastPercentage      = 0;
    uint32_t   progressGeometriesCompleted = 0;
    uint64_t   progressTrianglesCompleted  = 0;
    std::mutex progressMutex;

    nvutils::PerformanceTimer clock;
    double                    startTime = 0;


    // 函数：init。初始化本模块所需状态、资源或 GPU 侧绑定。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
    void init(float pct);


    // 函数：setupParallelism。初始化本模块所需状态、资源或 GPU 侧绑定。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
    void setupParallelism(size_t geometryCount_, size_t geometryCompletedCount, int parallelismMode);


    // 函数：setupCompressedGltf。初始化本模块所需状态、资源或 GPU 侧绑定。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
    void setupCompressedGltf(size_t bufferViewCount);


    // 函数：deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
    void deinit();


    // 函数：logBegin。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    void     logBegin(uint64_t totalTriangleCount);

    uint32_t logCompletedGeometry(uint64_t triangleCount = 0);


    // 函数：logEnd。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    void     logEnd();
  };


  // 函数：loadGLTF。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
  Result loadGLTF(ProcessingInfo& processingInfo, const std::filesystem::path& filePath);

private:


  // 函数：loadGeometryGLTF。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
  void loadGeometryGLTF(ProcessingInfo& processingInfo, uint64_t geometryIndex, size_t meshIndex, const struct cgltf_data* gltf);
  GltfNodeImportResult addInstancesFromNodeGLTF(const std::vector<size_t>& meshToGeometry,
                                                const struct cgltf_data*   data,
                                                const struct cgltf_node*   node,
                                                const glm::mat4 parentObjToWorldTransform = glm::mat4(1),
                                                uint32_t        depth                     = 0);

  void assignAssemblyToRange(uint32_t assemblyID, uint32_t firstInstance, uint32_t instanceCount);


  bool loadCompressedViewsGLTF(ProcessingInfo&                                processingInfo,
                               std::unordered_set<struct cgltf_buffer_view*>& bufferViews,
                               const struct cgltf_data*                       gltf);
  void unloadCompressedViewsGLTF(ProcessingInfo&                                processingInfo,
                                 std::unordered_set<struct cgltf_buffer_view*>& bufferViews,
                                 const struct cgltf_data*                       gltf);


  // 函数：processGeometry。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void processGeometry(ProcessingInfo& processingInfo, size_t geometryIndex, bool isCached);


  // 函数：buildGeometryLod。构建派生数据结构，通常用于 LOD、层次结构、间接命令或加速访问。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：构建结果会被后续阶段高频读取，必须保证布局紧凑、索引合法并与共享结构定义一致。
  void buildGeometryLod(ProcessingInfo& processingInfo, GeometryStorage& geometry);


  // 函数：buildHierarchy。构建派生数据结构，通常用于 LOD、层次结构、间接命令或加速访问。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：构建结果会被后续阶段高频读取，必须保证布局紧凑、索引合法并与共享结构定义一致。
  void buildHierarchy(ProcessingInfo& processingInfo, GeometryStorage& geometry);


  // 函数：computeLodBboxes_recursive。计算派生值，供后续剔除、LOD、统计或资源规划使用。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：计算结果通常参与阈值比较或内存规划，数值稳定性和边界条件需要特别注意。
  void computeLodBboxes_recursive(GeometryStorage& geometry, size_t nodeIdx);


  // 函数：buildGeometryDedupVertices。构建派生数据结构，通常用于 LOD、层次结构、间接命令或加速访问。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：构建结果会被后续阶段高频读取，必须保证布局紧凑、索引合法并与共享结构定义一致。
  void buildGeometryDedupVertices(ProcessingInfo& processingInfo, GeometryStorage& geometry);


  // 函数：computeHistogramMaxs。计算派生值，供后续剔除、LOD、统计或资源规划使用。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：计算结果通常参与阈值比较或内存规划，数值稳定性和边界条件需要特别注意。
  void computeHistogramMaxs();


  // 函数：computeInstanceBBoxes。计算派生值，供后续剔除、LOD、统计或资源规划使用。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：计算结果通常参与阈值比较或内存规划，数值稳定性和边界条件需要特别注意。
  void computeInstanceBBoxes();


  // 函数：beginProcessingOnly。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void beginProcessingOnly(size_t geometryCount);


  // 函数：saveProcessingOnly。把当前状态写入缓存、缓冲、文件或着色器可消费的数据布局。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：写入路径应明确字节对齐、所有权和可见性，避免后续读取端解释错误。
  void saveProcessingOnly(ProcessingInfo& processingInfo, size_t geometryIndex);


  // 函数：endProcessingOnly。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  bool endProcessingOnly(ProcessingInfo& processingInfo, bool hadError);


  // 结构：TempContext。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct TempContext
  {
    ProcessingInfo&  processingInfo;
    GeometryStorage& geometry;
    Scene&           scene;

    bool      innerThreadingActive   = false;
    bool      levelGroupOffsetValid  = false;
    GroupInfo threadGroupInfo        = {};
    uint32_t  threadGroupSize        = 0;
    uint32_t  threadGroupStorageSize = 0;
    uint32_t  lodLevel               = ~0u;
    size_t    levelGroupOffset       = 0;


    std::mutex           groupMutex;
    std::atomic_uint32_t groupIndexOrdered = 0;
    std::atomic_size_t   groupDataOrdered  = 0;
    std::vector<uint8_t> threadGroupDatas;
  };


  // 结构：TempGroup。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct TempGroup
  {
    uint32_t                  lodLevel;
    uint32_t                  clusterCount;
    shaderio::TraversalMetric traversalMetric;
  };


  // 结构：TempCluster。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct TempCluster
  {
    const uint32_t* indices         = nullptr;
    uint32_t        indexCount      = 0;
    uint32_t        generatingGroup = 0;
  };

  uint32_t storeGroup(TempContext*       context,
                      uint32_t           threadIndex,
                      uint32_t           groupIndex,
                      const clodGroup&   group,
                      uint32_t           clusterCount,
                      const clodCluster* clusters);


  // 函数：compressGroup。执行压缩或解压流程，在体积和运行时访问格式之间做转换。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：压缩必须保留可验证的重建语义；当压缩收益不足或超出约束时应回退到未压缩表示。
  void compressGroup(TempContext* context, GroupStorage& groupTempStorage, GroupInfo& groupInfo, uint32_t* vertexCacheLocal);


  // 函数：clodIterationMeshoptimizer。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  static void clodIterationMeshoptimizer(void* iteration_context, void* output_context, int depth, size_t task_count);
  static int  clodGroupMeshoptimizer(void*              output_context,
                                     clodGroup          group,
                                     const clodCluster* clusters,
                                     size_t             cluster_count,
                                     size_t             task_index,
                                     uint32_t           thread_index);
};

}
