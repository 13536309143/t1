//==============================================================================
// 文件：src/core/cache.cpp
// 模块定位：Scene 缓存文件读写实现，负责几何运行时视图的序列化、校验、复用和离线处理保存。
// 数据流：输入是已处理的 GeometryView 或缓存文件；输出是可直接接入 Scene 的数组视图和 offset table。
// 方法说明：缓存机制把昂贵的 glTF 解析、LOD 构建和压缩结果持久化，使交互式渲染实验可以复现同一几何数据集。
// 正确性约束：缓存配置必须与当前 SceneConfig 匹配；内存映射模式下不能复制后释放原映射；partial processing 要保证 offset table 一致。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <nvutils/file_operations.hpp>
#include <nvutils/parallel_work.hpp>
#include <nvutils/file_mapping.hpp>
#include <nvutils/logger.hpp>
#include "scene.hpp"


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace lodclusters {


// 函数：Scene::storeCached。把当前状态写入缓存、缓冲、文件或着色器可消费的数据布局。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：写入路径应明确字节对齐、所有权和可见性，避免后续读取端解释错误。
bool Scene::storeCached(const GeometryView& view, uint64_t dataSize, void* data)
{
  uint64_t dataAddress = reinterpret_cast<uint64_t>(data);
  uint64_t dataEnd     = dataAddress + dataSize;
  bool isValid = (dataAddress % serialization::ALIGNMENT) == 0 && (dataAddress + sizeof(GeometryBase)) <= dataEnd;
  if(isValid)
  {
    memcpy(reinterpret_cast<void*>(dataAddress), (const GeometryBase*)&view, sizeof(GeometryBase));
    dataAddress += (sizeof(GeometryBase) + serialization::ALIGN_MASK) & ~serialization::ALIGN_MASK;

    serialization::storeAndAdvance(isValid, dataAddress, dataEnd, view.groupData);

    serialization::storeAndAdvance(isValid, dataAddress, dataEnd, view.groupInfos);

    serialization::storeAndAdvance(isValid, dataAddress, dataEnd, view.lodLevels);

    serialization::storeAndAdvance(isValid, dataAddress, dataEnd, view.lodNodes);

    serialization::storeAndAdvance(isValid, dataAddress, dataEnd, view.lodNodeBboxes);

    serialization::storeAndAdvance(isValid, dataAddress, dataEnd, view.localMaterialIDs);
  }
  return isValid;
}


// 函数：fileWriteAligned。把当前状态写入缓存、缓冲、文件或着色器可消费的数据布局。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：写入路径应明确字节对齐、所有权和可见性，避免后续读取端解释错误。
static bool fileWriteAligned(uint64_t& outAccumulatedSize, FILE* outFile, size_t dataSize, const void* data)
{

  assert(outAccumulatedSize % serialization::ALIGNMENT == 0);

  static const uint8_t padBytes[serialization::ALIGNMENT] = {};

  if(fwrite(data, dataSize, 1, outFile) != 1)
    return false;

  uint64_t newDataSize = (dataSize + serialization::ALIGN_MASK) & ~serialization::ALIGN_MASK;

  uint64_t padSize = newDataSize - dataSize;
  if(padSize)
  {
    if(fwrite(padBytes, padSize, 1, outFile) != 1)
    {
      return false;
    }
  }

  outAccumulatedSize += newDataSize;
  return true;
}

template <typename T>


// 函数：fileWriteAligned。把当前状态写入缓存、缓冲、文件或着色器可消费的数据布局。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：写入路径应明确字节对齐、所有权和可见性，避免后续读取端解释错误。
inline void fileWriteAligned(bool& isValid, uint64_t& outAccumulatedSize, FILE* outFile, const std::span<const T>& view)
{

  assert(outAccumulatedSize % serialization::ALIGNMENT == 0);

  if(isValid)
  {
    union
    {
      uint64_t count;
      uint8_t  countData[serialization::ALIGNMENT];
    };
    memset(countData, 0, sizeof(countData));


    count = view.size();

    if(fwrite(countData, serialization::ALIGNMENT, 1, outFile) != 1)
    {
      isValid = false;
      return;
    }

    outAccumulatedSize += serialization::ALIGNMENT;

    if(view.size() && !fileWriteAligned(outAccumulatedSize, outFile, view.size_bytes(), view.data()))
    {
      isValid = false;
    }
  }
}


// 函数：Scene::storeCached。把当前状态写入缓存、缓冲、文件或着色器可消费的数据布局。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：写入路径应明确字节对齐、所有权和可见性，避免后续读取端解释错误。
uint64_t Scene::storeCached(const GeometryView& view, FILE* outFile)
{
  uint64_t dataSize = 0;

  bool isValid = fileWriteAligned(dataSize, outFile, sizeof(GeometryBase), (const GeometryBase*)&view);

  if(isValid)
  {

    fileWriteAligned(isValid, dataSize, outFile, view.groupData);

    fileWriteAligned(isValid, dataSize, outFile, view.groupInfos);

    fileWriteAligned(isValid, dataSize, outFile, view.lodLevels);

    fileWriteAligned(isValid, dataSize, outFile, view.lodNodes);

    fileWriteAligned(isValid, dataSize, outFile, view.lodNodeBboxes);

    fileWriteAligned(isValid, dataSize, outFile, view.localMaterialIDs);
  }

  return dataSize;
}


// 函数：Scene::loadCached。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
bool Scene::loadCached(GeometryView& view, uint64_t dataSize, const void* data)
{
  uint64_t dataAddress = reinterpret_cast<uint64_t>(data);
  uint64_t dataEnd     = dataAddress + dataSize;

  bool isValid = true;

  if(dataAddress % serialization::ALIGNMENT == 0 && dataAddress + sizeof(GeometryBase) <= dataEnd)
  {
    memcpy((GeometryBase*)&view, data, sizeof(GeometryBase));
    dataAddress += (sizeof(GeometryBase) + serialization::ALIGN_MASK) & ~serialization::ALIGN_MASK;
  }
  else
  {
    view = {};
    return false;
  }

  if(isValid)
  {

    serialization::loadAndAdvance(isValid, dataAddress, dataEnd, view.groupData);

    serialization::loadAndAdvance(isValid, dataAddress, dataEnd, view.groupInfos);

    serialization::loadAndAdvance(isValid, dataAddress, dataEnd, view.lodLevels);

    serialization::loadAndAdvance(isValid, dataAddress, dataEnd, view.lodNodes);

    serialization::loadAndAdvance(isValid, dataAddress, dataEnd, view.lodNodeBboxes);

    serialization::loadAndAdvance(isValid, dataAddress, dataEnd, view.localMaterialIDs);
  }

  return isValid;
}


// 函数：Scene::CacheFileView::init。初始化本模块所需状态、资源或 GPU 侧绑定。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
bool Scene::CacheFileView::init(uint64_t dataSize, const void* data)
{
  m_dataSize  = dataSize;
  m_dataBytes = reinterpret_cast<const uint8_t*>(data);

  if(dataSize <= sizeof(CacheFileHeader) + sizeof(uint64_t))
  {
    m_dataSize = 0;
    return false;
  }

  const CacheFileHeader* fileHeader = (const CacheFileHeader*)data;

  if(!fileHeader->isValid())
  {
    m_dataSize = 0;
    return false;
  }

  m_geometryCount = *getPointer<uint64_t>(m_dataSize - sizeof(uint64_t));

  if(!m_geometryCount || (dataSize <= (sizeof(CacheFileHeader) + sizeof(uint64_t) * (m_geometryCount * 2 + 1))))
  {
    m_dataSize = 0;
    return false;
  }

  m_tableStart = m_dataSize - sizeof(uint64_t) * (m_geometryCount * 2 + 1);

  return true;
}


// 函数：Scene::CacheFileView::getSceneConfig。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void Scene::CacheFileView::getSceneConfig(SceneConfig& settings) const
{
  const CacheFileHeader* cacheHeader = (const CacheFileHeader*)(m_dataBytes);

  settings = cacheHeader->config;
}


// 函数：Scene::CacheFileView::getHistograms。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void Scene::CacheFileView::getHistograms(Histograms& histograms) const
{
  const CacheFileHeader* cacheHeader = (const CacheFileHeader*)(m_dataBytes);

  histograms = cacheHeader->histograms;
}

void Scene::CacheFileView::getProcessingStats(ProcessingStatsSnapshot& stats) const
{
  const CacheFileHeader* cacheHeader = (const CacheFileHeader*)(m_dataBytes);

  stats = cacheHeader->processingStats;
}


// 函数：Scene::CacheFileView::getGeometryView。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
bool Scene::CacheFileView::getGeometryView(GeometryView& view, uint64_t geometryIndex) const
{
  constexpr uint64_t ALIGN_MASK = serialization::ALIGNMENT - 1;

  if(geometryIndex >= m_geometryCount)
  {

    assert(0);
    return false;
  }

  const uint64_t* geometryOffsets = getPointer<uint64_t>(m_tableStart, m_geometryCount * 2);
  uint64_t        base            = geometryOffsets[geometryIndex * 2 + 0];

  if(base + sizeof(GeometryBase) > m_tableStart)
  {


    assert(0);
    return false;
  }

  uint64_t geometryTotalSize = geometryOffsets[geometryIndex * 2 + 1];
  uint64_t baseEnd           = base + geometryTotalSize;

  const uint8_t* geoData = getPointer<uint8_t>(base, geometryTotalSize);

  return Scene::loadCached(view, geometryTotalSize, geoData);
}


// 函数：Scene::checkCache。返回条件判断结果，用于调用方选择后续分支或验证输入状态。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：谓词函数应保持无副作用或低副作用，使调用方可以安全地把它用于断言、过滤和早退。
bool Scene::checkCache(const GeometryLodInput& info, size_t geometryIndex)
{
  if(m_cacheFileView.isValid() && geometryIndex < m_cacheFileView.getGeometryCount())
  {
    GeometryView cacheView = {};
    if(!m_cacheFileView.getGeometryView(cacheView, geometryIndex))
    {
      return false;
    }

    return memcmp(&info, &cacheView.lodInfo, sizeof(cacheView.lodInfo)) == 0;
  }
  return false;
}

template <typename T>


// 函数：fillVector。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
static inline void fillVector(std::vector<T>& storageVec, const std::span<const T>& viewSpan)
{
  storageVec.resize(viewSpan.size());
  memcpy(storageVec.data(), viewSpan.data(), viewSpan.size_bytes());
}


// 函数：Scene::loadCachedGeometry。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
void Scene::loadCachedGeometry(GeometryStorage& storage, size_t geometryIndex)
{
  GeometryView view = {};

  m_cacheFileView.getGeometryView(view, geometryIndex);
  (GeometryBase&)storage = view;


  fillVector(storage.groupData, view.groupData);

  fillVector(storage.groupInfos, view.groupInfos);

  fillVector(storage.lodLevels, view.lodLevels);

  fillVector(storage.lodNodes, view.lodNodes);

  fillVector(storage.lodNodeBboxes, view.lodNodeBboxes);

  fillVector(storage.localMaterialIDs, view.localMaterialIDs);
}


// 函数：Scene::openCache。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void Scene::openCache()
{
  if(m_cacheFileMapping.open(m_cacheFilePath))
  {
    m_cacheFileView.init(m_cacheFileMapping.size(), m_cacheFileMapping.data());
    if(m_cacheFileView.isValid())
    {


      m_loadedFromCache = true;

      m_cacheFileSize   = m_cacheFileMapping.size();


      std::string cacheFileName = nvutils::utf8FromPath(m_cacheFilePath);
      LOGI("Scene::init using cache file:\n  %s\n", cacheFileName.c_str());

      if(m_cacheFileSize > size_t(2) * 1024 * 1024 * 1024)
      {
        m_loaderConfig.memoryMappedCache = true;
      }
    }
    else
    {

      m_cacheFileView.deinit();

      m_cacheFileMapping.close();
    }
  }
}


// 函数：Scene::closeCache。释放或回收前面初始化的资源，保持生命周期成对管理。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
void Scene::closeCache()
{
  if(m_cacheFileView.isValid())
  {

    m_cacheFileView.deinit();

    m_cacheFileMapping.close();
  }
}


// 函数：Scene::saveCache。把当前状态写入缓存、缓冲、文件或着色器可消费的数据布局。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：写入路径应明确字节对齐、所有权和可见性，避免后续读取端解释错误。
bool Scene::saveCache() const
{
  uint64_t dataOffset = sizeof(Scene::CacheFileHeader);

  std::vector<uint64_t> geometryOffsets;
  geometryOffsets.reserve(m_geometryViews.size() * 2 + 1);

  for(const GeometryView& geom : m_geometryViews)
  {

    uint64_t geomDataSize = geom.getCachedSize();

    geometryOffsets.push_back(dataOffset);

    geometryOffsets.push_back(geomDataSize);

    dataOffset += geomDataSize;
  }
  geometryOffsets.push_back(m_geometryViews.size());

  uint64_t tableOffset = dataOffset;

  dataOffset += geometryOffsets.size() * sizeof(uint64_t);

  nvutils::FileReadOverWriteMapping outMapping;

  std::string outFilename = nvutils::utf8FromPath(m_filePath) + ".zippp";

  if(!outMapping.open(outFilename.c_str(), dataOffset))
  {
    LOGE("Scene::saveCache failed to save file %s\n", outFilename.c_str());
    return false;
  }

  uint8_t* mappingData = static_cast<uint8_t*>(outMapping.data());
  Scene::CacheFileHeader cacheHeader;
  cacheHeader.config          = m_config;
  cacheHeader.histograms      = m_histograms;
  cacheHeader.processingStats = m_processingStats;
  memcpy(mappingData, &cacheHeader, sizeof(cacheHeader));
  memcpy(mappingData + tableOffset, geometryOffsets.data(), sizeof(uint64_t) * geometryOffsets.size());

  bool hadError = false;
  nvutils::parallel_batches(m_geometryViews.size(), [&](uint64_t idx) {
    const GeometryView& view = m_geometryViews[idx];

    uint64_t dataOffset = geometryOffsets[idx * 2 + 0];
    uint64_t dataSize   = geometryOffsets[idx * 2 + 1];

    if(!Scene::storeCached(view, dataSize, mappingData + dataOffset))
    {
      hadError = true;
    }
  });

  if(hadError)
  {
    LOGE("Scene::saveCache had errors while saving %s\n", outFilename.c_str());
  }
  else
  {
    LOGI("Scene::saveCache saved %s\n", outFilename.c_str());
  }

  return !hadError;
}


// 函数：Scene::beginProcessingOnly。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void Scene::beginProcessingOnly(size_t geometryCount)
{
  if(!m_loaderConfig.processingOnly || m_cacheFileView.isValid())
  {
    return;
  }


  std::string outFilename        = nvutils::utf8FromPath(m_cacheFilePath);

  std::string outPartialFilename = nvutils::utf8FromPath(m_cachePartialFilePath);


  bool partialExists = m_loaderConfig.processingAllowPartial && std::filesystem::exists(m_cachePartialFilePath)

                       && std::filesystem::exists(m_cacheFilePath);

  const char* mode = partialExists ? "ab" : "wb";

  m_processingOnlyPartialCompleted = 0;
  m_processingOnlyFileOffset       = sizeof(Scene::CacheFileHeader);


  m_processingOnlyGeometryOffsets.resize(geometryCount * 2 + 1);
  m_processingOnlyGeometryOffsets[geometryCount * 2] = geometryCount;

  if(partialExists)
  {
    nvutils::FileReadMapping mapping;
    if(mapping.open(m_cachePartialFilePath) && mapping.size())
    {
      size_t                   entryCount = mapping.size() / sizeof(CachePartialEntry);
      const CachePartialEntry* entries    = reinterpret_cast<const CachePartialEntry*>(mapping.data());


      LOGI("Scene::beginProcessingOnly partial resuming - %llu completed\n", entryCount);

      for(size_t i = 0; i < entryCount; i++)
      {
        const CachePartialEntry& entry = entries[i];

        m_processingOnlyGeometryOffsets[entry.geometryIndex * 2 + 0] = entry.offset;
        m_processingOnlyGeometryOffsets[entry.geometryIndex * 2 + 1] = entry.dataSize;


        m_processingOnlyFileOffset = std::max(m_processingOnlyFileOffset, entry.offset + entry.dataSize);
      }

      mapping.close();

      m_processingOnlyPartialCompleted = entryCount;

      std::filesystem::resize_file(m_cacheFilePath, m_processingOnlyFileOffset);
      std::filesystem::resize_file(m_cachePartialFilePath, sizeof(CachePartialEntry) * entryCount);
    }
  }

  m_processingOnlyFile        = nullptr;
  m_processingOnlyPartialFile = nullptr;
  int result                  = 0;
#ifdef WIN32
  result = fopen_s(&m_processingOnlyFile, outFilename.c_str(), mode) == 0;
#else
  m_processingOnlyFile = fopen(outFilename.c_str(), mode);
  result               = (m_processingOnlyFile) != nullptr;
#endif

  if(!result)
  {
    LOGE("Scene::beginProcessOnlySave failed to save file:\n   %s\n", outFilename.c_str());
    return;
  }

  if(!partialExists)
  {
    Scene::CacheFileHeader header;
    header.config          = m_config;
    header.histograms      = m_histograms;
    header.processingStats = m_processingStats;

    fwrite(&header, sizeof(header), 1, m_processingOnlyFile);
  }

  if(m_loaderConfig.processingAllowPartial)
  {
#ifdef WIN32
    result = fopen_s(&m_processingOnlyPartialFile, outPartialFilename.c_str(), mode) == 0;
#else
    m_processingOnlyPartialFile = fopen(outPartialFilename.c_str(), mode);
    result                      = (m_processingOnlyFile) != nullptr;
#endif

    if(!result)
    {

      fclose(m_processingOnlyFile);
      m_processingOnlyFile = nullptr;

      LOGE("Scene::beginProcessOnlySave failed to save file:\n  %s\n", outPartialFilename.c_str());
      return;
    }
  }

  LOGI("Scene::beginProcessOnlySave started save file:\n  %s\n", outFilename.c_str());
}


// 函数：Scene::saveProcessingOnly。把当前状态写入缓存、缓冲、文件或着色器可消费的数据布局。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：写入路径应明确字节对齐、所有权和可见性，避免后续读取端解释错误。
void Scene::saveProcessingOnly(ProcessingInfo& processingInfo, size_t geometryIndex)
{
  {


    // 函数：lock。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    std::lock_guard lock(processingInfo.processOnlySaveMutex);


    uint64_t dataSize = storeCached(m_geometryViews[geometryIndex], m_processingOnlyFile);

    fflush(m_processingOnlyFile);

    m_processingOnlyGeometryOffsets[geometryIndex * 2 + 0] = m_processingOnlyFileOffset;
    m_processingOnlyGeometryOffsets[geometryIndex * 2 + 1] = dataSize;

    if(m_processingOnlyPartialFile)
    {
      CachePartialEntry entry = {geometryIndex, m_processingOnlyFileOffset, dataSize};

      fwrite(&entry, sizeof(entry), 1, m_processingOnlyPartialFile);

      fflush(m_processingOnlyPartialFile);
    }


    m_processingOnlyFileOffset += dataSize;
  }
  m_geometryViews[geometryIndex]    = {};
  m_geometryStorages[geometryIndex] = {};
}


// 函数：Scene::endProcessingOnly。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
bool Scene::endProcessingOnly(ProcessingInfo& processingInfo, bool hadError)
{
  if(!m_processingOnlyFile)
    return false;

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

  if(!hadError)
  {
    fwrite(m_processingOnlyGeometryOffsets.data(), m_processingOnlyGeometryOffsets.size() * sizeof(uint64_t), 1, m_processingOnlyFile);
  }
  fseek(m_processingOnlyFile, offsetof(CacheFileHeader, histograms), SEEK_SET);
  fwrite(&m_histograms, sizeof(m_histograms), 1, m_processingOnlyFile);
  fwrite(&m_processingStats, sizeof(m_processingStats), 1, m_processingOnlyFile);


  fclose(m_processingOnlyFile);
  if(m_processingOnlyPartialFile)
  {

    fclose(m_processingOnlyPartialFile);
  }


  m_geometryStorages.clear();

  m_geometryViews.clear();

  m_processingOnlyFile             = nullptr;
  m_processingOnlyPartialFile      = nullptr;
  m_processingOnlyPartialCompleted = 0;

  m_processingOnlyGeometryOffsets.clear();


  std::string outFilename        = nvutils::utf8FromPath(m_cacheFilePath);

  std::string outPartialFilename = nvutils::utf8FromPath(m_cachePartialFilePath);


  LOGI("Scene::endProcessOnlySave completed %s\n", hadError ? "with errors" : "successfully");

  if(std::filesystem::exists(m_cachePartialFilePath))
  {

    std::filesystem::remove(m_cachePartialFilePath);

    LOGI("  deleted: %s\n", outPartialFilename.c_str());
  }

  if(hadError)
  {

    std::filesystem::remove(m_cacheFilePath);

    LOGI("  deleted: %s\n", outFilename.c_str());
  }
  else
  {
    LOGI("  saved: %s\n", outFilename.c_str());
  }

  return true;
}

}
