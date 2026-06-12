//==============================================================================
// 文件：src/app/lodclusters_scene.cpp
// 模块定位：场景加载、缓存管理、RenderScene 驻留模式选择和相机初始化实现。
// 数据流：输入是 glTF/cfg 路径、SceneConfig 和 StreamingConfig；输出是 Scene、RenderScene、场景网格、相机和缓存文件。
// 方法说明：该层把离线几何处理和在线渲染驻留解耦：Scene 表示 CPU 侧语义数据，RenderScene 表示 GPU 可访问布局。
// 正确性约束：异步加载期间不得释放正在使用的 Scene；preload 失败时必须可回退到 流式加载；相机裁剪面要覆盖 grid 后的整体包围盒。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <thread>
#include <volk.h>
#include <nvutils/file_operations.hpp>
#include <nvgui/camera.hpp>
#include "lodclusters.hpp"


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace lodclusters {


// 函数：LodClusters::initScene。初始化本模块所需状态、资源或 GPU 侧绑定。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
void LodClusters::initScene(std::filesystem::path filePath, std::string cacheSuffix, bool configChange)
{

  deinitScene();


  std::string fileName = nvutils::utf8FromPath(filePath);

  if(!fileName.empty())
  {
    LOGI("Loading scene %s\n", fileName.c_str());

    m_scene         = nullptr;
    m_sceneLoading  = true;
    m_sceneProgress = 0;

    std::thread([=, this]() {
      auto scene = std::make_unique<Scene>();
      if(scene->init(filePath, m_sceneConfig, m_sceneLoaderConfig, cacheSuffix, configChange) != Scene::SCENE_RESULT_SUCCESS)
      {
        scene = nullptr;

        LOGW("Loading scene failed\n");
      }
      else
      {

        m_scene               = std::move(scene);
        m_sceneFilePath       = filePath;

        m_tweak.clusterConfig = findSceneClusterConfig(m_scene->m_config);

        m_scene->updateSceneGrid(m_sceneGridConfig);
        m_sceneGridConfigLast = m_sceneGridConfig;

        updatedSceneGrid();
        m_renderSceneCanPreload = ScenePreloaded::canPreload(m_resources.getDeviceLocalHeapSize(), m_scene.get());

        if(!configChange)
        {
          m_sceneConfig = m_scene->m_config;

          postInitNewScene();
          m_tweakLast       = m_tweak;
          m_sceneConfigLast = m_sceneConfig;
          m_sceneConfigEdit = m_sceneConfig;
        }
      }
      m_sceneLoading = false;

    }).detach();

    return;
  }

  return;
}


// 函数：LodClusters::initRenderScene。初始化本模块所需状态、资源或 GPU 侧绑定。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
void LodClusters::initRenderScene()
{

  assert(m_scene);

  m_renderScene = std::make_unique<RenderScene>();

  bool success = m_renderScene->init(&m_resources, m_scene.get(), m_streamingConfig, m_tweak.useStreaming);


  if(!m_tweak.useStreaming && !success)
  {

    m_tweak.useStreaming     = true;
    m_tweakLast.useStreaming = true;

    if(!m_renderScene->init(&m_resources, m_scene.get(), m_streamingConfig, true))
    {

      LOGW("Init renderscene failed\n");

      deinitRenderScene();
    }
  }
  else if(!success && m_tweak.useStreaming)
  {

    LOGW("Init renderscene failed\n");

    deinitRenderScene();
  }

  m_streamingConfigLast = m_streamingConfig;
}


// 函数：LodClusters::deinitRenderScene。释放或回收前面初始化的资源，保持生命周期成对管理。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
void LodClusters::deinitRenderScene()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));
  if(m_renderScene)
  {

    m_renderScene->deinit();
    m_renderScene = nullptr;
  }
}


// 函数：LodClusters::deinitScene。释放或回收前面初始化的资源，保持生命周期成对管理。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
void LodClusters::deinitScene()
{

  deinitRenderScene();

  if(m_scene)
  {

    m_scene->deinit();
    m_scene = nullptr;
  }
}


// 函数：LodClusters::postInitNewScene。初始化本模块所需状态、资源或 GPU 侧绑定。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
void LodClusters::postInitNewScene()
{

  assert(m_scene);

  glm::vec3 extent         = m_scene->m_bbox.hi - m_scene->m_bbox.lo;
  glm::vec3 center         = (m_scene->m_bbox.hi + m_scene->m_bbox.lo) * 0.5f;

  float     sceneDimension = glm::length(extent);

  m_frameConfig.frameConstants.wLightPos = center + sceneDimension;

  m_frameConfig.frameConstants.sceneSize = glm::length(m_scene->m_bbox.hi - m_scene->m_bbox.lo);

  setSceneCamera(m_sceneFilePath);
  m_frames                    = 0;
  m_streamingConfig.maxGroups = std::max(m_streamingConfig.maxGroups, uint32_t(m_scene->getActiveGeometryCount()));

  if(!m_scene->m_hasVertexNormals)
    m_tweak.facetShading = true;


  m_frameConfig.frameConstants.skyParams.sunDirection = glm::normalize(m_frameConfig.frameConstants.skyParams.sunDirection);
}


// 函数：LodClusters::saveCacheFile。把当前状态写入缓存、缓冲、文件或着色器可消费的数据布局。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：写入路径应明确字节对齐、所有权和可见性，避免后续读取端解释错误。
void LodClusters::saveCacheFile()
{
  if(m_scene)
  {

    m_scene->saveCache();
  }
}


// 函数：LodClusters::onFileDrop。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void LodClusters::onFileDrop(const std::filesystem::path& filePath)
{
  if(filePath.empty())
    return;

  if(!m_sceneLoadFromConfig)
  {

    if(!m_sceneFilePathDropLast.empty() && filePath != m_sceneFilePathDropLast)
    {

      m_sceneGridConfig.uniqueGeometriesForCopies = false;

      m_cameraSpeed             = 0;
      m_cameraString            = {};
      m_cameraStringLast        = {};
      m_cameraStringCommandLine = {};
    }
    m_sceneFilePathDropLast = filePath;
    m_sceneFilePathDropNew  = filePath;
  }

  if(filePath.extension() == ".cfg")
  {

    std::string filePathString = nvutils::utf8FromPath(filePath);

    LOGI("Loading config: %s\n", filePathString.c_str());

    std::vector<const char*> args;

    args.push_back("--configfile");
    args.push_back(filePathString.c_str());

    std::filesystem::path oldFilePath = m_sceneFilePathDropNew;


    m_info.parameterParser->parse(std::span(args), false, {}, {}, true);

    if(!m_cameraStringCommandLine.empty())
    {

      m_cameraString = m_cameraStringCommandLine;
    }

    if(m_sceneFilePathDropNew != m_sceneFilePathDropLast)
    {
      bool oldState = m_sceneLoadFromConfig;

      m_sceneLoadFromConfig             = true;
      std::filesystem::path cfgFilePath = m_sceneFilePathDropNew;

      onFileDrop(cfgFilePath);

      m_sceneLoadFromConfig  = oldState;
      m_sceneFilePathDropNew = oldFilePath;
    }
    return;
  }

  LOGI("Loading model: %s\n", nvutils::utf8FromPath(filePath).c_str());

  deinitRenderer();


  initScene(filePath, m_sceneCacheSuffix, false);
}


// 函数：LodClusters::doProcessingOnly。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void LodClusters::doProcessingOnly()
{

  setFromClusterConfig(m_sceneConfig, m_tweak.clusterConfig);

  assert(m_app == nullptr);
  m_scene = std::make_unique<Scene>();

  m_scene->init(m_sceneFilePathDropNew, m_sceneConfig, m_sceneLoaderConfig, m_sceneCacheSuffix, false);
}


const LodClusters::ClusterInfo LodClusters::s_clusterInfos[NUM_CLUSTER_CONFIGS] = {
    {32, 32, CLUSTER_32T_32V}, {32, 64, CLUSTER_32T_64V}, {32, 96, CLUSTER_32T_96V}, {32, 128, CLUSTER_32T_128V}, {32, 160, CLUSTER_32T_160V}, {32, 192, CLUSTER_32T_192V}, {32, 224, CLUSTER_32T_224V}, {32, 256, CLUSTER_32T_256V},
    {64, 32, CLUSTER_64T_32V}, {64, 64, CLUSTER_64T_64V}, {64, 96, CLUSTER_64T_96V}, {64, 128, CLUSTER_64T_128V}, {64, 160, CLUSTER_64T_160V}, {64, 192, CLUSTER_64T_192V}, {64, 224, CLUSTER_64T_224V}, {64, 256, CLUSTER_64T_256V},
    {96, 32, CLUSTER_96T_32V}, {96, 64, CLUSTER_96T_64V}, {96, 96, CLUSTER_96T_96V}, {96, 128, CLUSTER_96T_128V}, {96, 160, CLUSTER_96T_160V}, {96, 192, CLUSTER_96T_192V}, {96, 224, CLUSTER_96T_224V}, {96, 256, CLUSTER_96T_256V},
    {128, 32, CLUSTER_128T_32V}, {128, 64, CLUSTER_128T_64V}, {128, 96, CLUSTER_128T_96V}, {128, 128, CLUSTER_128T_128V}, {128, 160, CLUSTER_128T_160V}, {128, 192, CLUSTER_128T_192V}, {128, 224, CLUSTER_128T_224V}, {128, 256, CLUSTER_128T_256V},
    {160, 32, CLUSTER_160T_32V}, {160, 64, CLUSTER_160T_64V}, {160, 96, CLUSTER_160T_96V}, {160, 128, CLUSTER_160T_128V}, {160, 160, CLUSTER_160T_160V}, {160, 192, CLUSTER_160T_192V}, {160, 224, CLUSTER_160T_224V}, {160, 256, CLUSTER_160T_256V},
    {192, 32, CLUSTER_192T_32V}, {192, 64, CLUSTER_192T_64V}, {192, 96, CLUSTER_192T_96V}, {192, 128, CLUSTER_192T_128V}, {192, 160, CLUSTER_192T_160V}, {192, 192, CLUSTER_192T_192V}, {192, 224, CLUSTER_192T_224V}, {192, 256, CLUSTER_192T_256V},
    {224, 32, CLUSTER_224T_32V}, {224, 64, CLUSTER_224T_64V}, {224, 96, CLUSTER_224T_96V}, {224, 128, CLUSTER_224T_128V}, {224, 160, CLUSTER_224T_160V}, {224, 192, CLUSTER_224T_192V}, {224, 224, CLUSTER_224T_224V}, {224, 256, CLUSTER_224T_256V},
    {256, 32, CLUSTER_256T_32V}, {256, 64, CLUSTER_256T_64V}, {256, 96, CLUSTER_256T_96V}, {256, 128, CLUSTER_256T_128V}, {256, 160, CLUSTER_256T_160V}, {256, 192, CLUSTER_256T_192V}, {256, 224, CLUSTER_256T_224V}, {256, 256, CLUSTER_256T_256V},
};


// 函数：LodClusters::findSceneClusterConfig。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
LodClusters::ClusterConfig LodClusters::findSceneClusterConfig(const SceneConfig& sceneConfig)
{
  for(uint32_t i = 0; i < NUM_CLUSTER_CONFIGS; i++)
  {
    const ClusterInfo& entry = s_clusterInfos[i];
    if(sceneConfig.clusterTriangles <= entry.tris && sceneConfig.clusterVertices <= entry.verts)
    {
      return entry.cfg;
    }
  }

  return CLUSTER_256T_256V;
}


// 函数：LodClusters::setFromClusterConfig。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void LodClusters::setFromClusterConfig(SceneConfig& sceneConfig, ClusterConfig clusterConfig)
{
  for(uint32_t i = 0; i < NUM_CLUSTER_CONFIGS; i++)
  {
    if(s_clusterInfos[i].cfg == clusterConfig)
    {
      sceneConfig.clusterTriangles = s_clusterInfos[i].tris;
      sceneConfig.clusterVertices  = s_clusterInfos[i].verts;
      return;
    }
  }
}


// 函数：LodClusters::updatedSceneGrid。根据最新状态刷新缓存数据、GPU 地址、描述符或统计信息。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：更新函数负责把“旧状态”推进到“当前状态”，因此要避免部分更新造成 CPU/GPU 视图不一致。
void LodClusters::updatedSceneGrid()
{
  {
    glm::vec3 gridExtent = m_scene->m_gridBbox.hi - m_scene->m_gridBbox.lo;
    float     gridRadius = glm::length(gridExtent) * 0.5f;

    glm::vec3 modelExtent = m_scene->m_bbox.hi - m_scene->m_bbox.lo;
    float     modelRadius = glm::length(modelExtent) * 0.5f;

    bool bigScene = m_scene->m_isBig;

    if(!m_cameraSpeed)
      m_info.cameraManipulator->setSpeed(modelRadius * (bigScene ? 0.0025f : 0.25f));

    if(m_cameraString.empty())
      m_info.cameraManipulator->setClipPlanes(
          glm::vec2((bigScene ? 0.0001f : 0.01F) * modelRadius,
                    bigScene ? gridRadius * 1.2f : std::max(50.0f * modelRadius, gridRadius * 1.2f)));

  }

}


// 函数：LodClusters::setSceneCamera。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void LodClusters::setSceneCamera(const std::filesystem::path& filePath)
{

  nvgui::SetCameraJsonFile(filePath);

  glm::vec3 modelExtent = m_scene->m_bbox.hi - m_scene->m_bbox.lo;
  float     modelRadius = glm::length(modelExtent) * 0.5f;
  glm::vec3 modelCenter = (m_scene->m_bbox.hi + m_scene->m_bbox.lo) * 0.5f;

  bool bigScene = m_scene->m_isBig;

  if(!m_scene->m_cameras.empty())
  {
    auto& c = m_scene->m_cameras[0];

    m_info.cameraManipulator->setFov(c.fovy);


    c.eye              = glm::vec3(c.worldMatrix[3]);

    float     distance = glm::length(modelCenter - c.eye);

    glm::mat3 rotMat   = glm::mat3(c.worldMatrix);
    c.center           = {0, 0, -distance};
    c.center           = c.eye + (rotMat * c.center);
    c.up               = {0, 1, 0};

    m_info.cameraManipulator->setCamera({c.eye, c.center, c.up, static_cast<float>(glm::degrees(c.fovy))});

    nvgui::SetHomeCamera({c.eye, c.center, c.up, static_cast<float>(glm::degrees(c.fovy))});
    for(auto& cam : m_scene->m_cameras)
    {

      cam.eye            = glm::vec3(cam.worldMatrix[3]);

      float     distance = glm::length(modelCenter - cam.eye);

      glm::mat3 rotMat   = glm::mat3(cam.worldMatrix);
      cam.center         = {0, 0, -distance};
      cam.center         = cam.eye + (rotMat * cam.center);
      cam.up             = {0, 1, 0};


      nvgui::AddCamera({cam.eye, cam.center, cam.up, static_cast<float>(glm::degrees(cam.fovy))});
    }
  }
  else
  {
    glm::vec3 up  = {0, 1, 0};
    glm::vec3 dir = {1.0f, bigScene ? 0.33f : 0.75f, 1.0f};

    m_info.cameraManipulator->setLookat(modelCenter + dir * (modelRadius * (bigScene ? 0.5f : 1.f)), modelCenter, up);
    nvgui::SetHomeCamera(m_info.cameraManipulator->getCamera());
  }

  if(m_cameraSpeed)
  {

    m_info.cameraManipulator->setSpeed(m_cameraSpeed);
  }

  if(!m_cameraString.empty())
  {

    applyCameraString();
  }
}


// 函数：LodClusters::decodePickingDepth。在紧凑编码和逻辑结构之间转换，减少带宽或便于着色器访问。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：编码位宽、符号位和特殊值必须与写入端/读取端完全一致，否则会产生难以定位的跨阶段错误。
float LodClusters::decodePickingDepth(const shaderio::Readback& readback)
{
  if(!isPickingValid(readback))
  {
    return 0.f;
  }
  uint32_t bits = readback._packedDepth0;
  bits ^= ~(int(bits) >> 31) | 0x80000000u;
  float res = *(float*)&bits;
  return 1.f - res;
}


// 函数：LodClusters::isPickingValid。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
bool LodClusters::isPickingValid(const shaderio::Readback& readback)
{
  return readback._packedDepth0 != 0u;
}

}
