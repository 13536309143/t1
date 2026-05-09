// LodClusters 主文件 - 实现基于LOD (Level of Detail) 技术的集群渲染系统
// 该文件包含了LOD集群的核心功能，包括场景加载、渲染器初始化、相机控制等
#include <algorithm>
#include <thread>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <volk.h>
#include <fmt/format.h>
#include <nvutils/file_operations.hpp>
#include <nvgui/camera.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "lodclusters.hpp"
bool g_verbose = false;
namespace lodclusters {
namespace {
bool sceneClusterConfigMatches(const Scene& scene, const SceneConfig& config);
}
// LodClusters 构造函数
// 初始化LOD集群系统，设置参数注册表和默认配置
// 参数: info - 包含应用程序信息的结构体
LodClusters::LodClusters(const Info& info)
    : m_info(info)
{
  nvutils::ProfilerTimeline::CreateInfo createInfo;
  createInfo.name = "graphics";
  m_profilerTimeline = m_info.profilerManager->createTimeline(createInfo);
  m_info.parameterRegistry->add({"scene"}, {".gltf", ".glb", ".cfg", ".lscene"}, &m_sceneFilePathDropNew);
  m_info.parameterRegistry->add({"renderer"}, (int*)&m_tweak.renderer);
  m_info.parameterRegistry->add({"verbose"}, &g_verbose, true);
  m_info.parameterRegistry->add({"resetstats"}, &m_tweak.autoResetTimers);
  m_info.parameterRegistry->add({"supersample"}, &m_tweak.supersample);
  m_info.parameterRegistry->add({"debugui"}, &m_showDebugUI);
  m_info.parameterRegistry->add({"sequencescreenshot", "save screenshot at end of each sequence. 0 disabled (default), 1 full window, 2 rendered viewport"},(int*)&m_sequenceScreenshotMode, true);
  m_info.parameterRegistry->add({"dumpspirv", "dumps compiled spirv into working directory"}, &m_resources.m_dumpSpirv);
  m_info.parameterRegistry->add({"camerastring"}, &m_cameraString);
  m_info.parameterRegistry->add({"cameraspeed"}, &m_cameraSpeed);
  m_info.parameterRegistry->addVector({"sundirection"}, &m_frameConfig.frameConstants.skyParams.sunDirection);
  m_info.parameterRegistry->addVector({"suncolor"}, &m_frameConfig.frameConstants.skyParams.sunColor);
  m_info.parameterRegistry->add({"streaming"}, &m_tweak.useStreaming);
  m_info.parameterRegistry->add({"sim"}, &m_simulation.enabled);
  m_info.parameterRegistry->add({"simmotion"}, &m_simulation.motion);
  m_info.parameterRegistry->add({"simtarget"}, &m_simulation.target);
  m_info.parameterRegistry->add({"simscale"}, &m_simulation.timeScale);
  m_info.parameterRegistry->add({"simspin"}, &m_simulation.spinDegrees);
  m_info.parameterRegistry->add({"simamplitude"}, &m_simulation.amplitude);
  m_info.parameterRegistry->add({"simorbit"}, &m_simulation.orbitRadius);
  m_info.parameterRegistry->add({"gridcopies"}, &m_sceneGridConfig.numCopies);
  m_info.parameterRegistry->add({"gridconfig"}, &m_sceneGridConfig.gridBits);
  m_info.parameterRegistry->add({"gridunique"}, &m_sceneGridConfig.uniqueGeometriesForCopies);
  m_info.parameterRegistry->add({"clusterconfig"}, (int*)&m_tweak.clusterConfig);
  m_info.parameterRegistry->add({"clustergroupsize"}, &m_sceneConfig.clusterGroupSize);
  m_info.parameterRegistry->add({"simplifyuvweight"}, &m_sceneConfig.simplifyTexCoordWeight);
  m_info.parameterRegistry->add({"simplifynormalweight"}, &m_sceneConfig.simplifyNormalWeight);
  m_info.parameterRegistry->add({"simplifytangentweight"}, &m_sceneConfig.simplifyTangentWeight);
  m_info.parameterRegistry->add({"simplifytangentsignweight"}, &m_sceneConfig.simplifyTangentSignWeight);
  m_info.parameterRegistry->add({"attributes"}, &m_sceneConfig.enabledAttributes);

  m_info.parameterRegistry->add({"loderrormergeprevious"}, &m_sceneConfig.lodErrorMergePrevious);
  m_info.parameterRegistry->add({"loderrormergeadditive"}, &m_sceneConfig.lodErrorMergeAdditive);
  m_info.parameterRegistry->add({"loderroredgelimit"}, &m_sceneConfig.lodErrorEdgeLimit);
  m_info.parameterRegistry->add({"lodnodewidth"}, &m_sceneConfig.preferredNodeWidth);
  m_info.parameterRegistry->add({"loddecimationfactor"}, &m_sceneConfig.lodLevelDecimationFactor);
  m_info.parameterRegistry->add({"meshoptfillweight"}, &m_sceneConfig.meshoptFillWeight);
  /////////////////////////////////////////////
  //开启lod优化
  m_info.parameterRegistry->add({ "curvatureadaptive" }, &m_sceneConfig.curvatureAdaptiveStrength);
  m_info.parameterRegistry->add({ "curvaturewindow" }, &m_sceneConfig.curvatureWindowRadius);
  m_info.parameterRegistry->add({ "featureedge" }, &m_sceneConfig.featureEdgeThreshold);
  m_info.parameterRegistry->add({ "perceptualweight" }, &m_sceneConfig.perceptualWeight);
  m_info.parameterRegistry->add({ "silhouettepreserve" }, &m_sceneConfig.silhouettePreservation);
  ////////////////////////////////////////////////////
  m_info.parameterRegistry->add({"loderror"}, &m_frameConfig.lodPixelError);
  m_info.parameterRegistry->add({"shadowray"}, &m_frameConfig.frameConstants.doShadow);
  m_info.parameterRegistry->add({"maxtransfermegabytes"}, (uint32_t*)&m_streamingConfig.maxTransferMegaBytes);
  m_info.parameterRegistry->add({"maxblascachingmegabytes"}, (uint32_t*)&m_streamingConfig.maxBlasCachingMegaBytes);
  m_info.parameterRegistry->add({"maxgeomegabytes"}, (uint32_t*)&m_streamingConfig.maxGeometryMegaBytes);
  m_info.parameterRegistry->add({"maxresidentgroups"}, &m_streamingConfig.maxGroups);
  m_info.parameterRegistry->add({"maxframeloadrequests"}, &m_streamingConfig.maxPerFrameLoadRequests);
  m_info.parameterRegistry->add({"maxframeunloadrequests"}, &m_streamingConfig.maxPerFrameUnloadRequests);
  m_info.parameterRegistry->add({"cullederrorscale"}, &m_frameConfig.culledErrorScale);
  m_info.parameterRegistry->add({"culling"}, &m_rendererConfig.useCulling);
  //two
  m_info.parameterRegistry->add({"primitiveculling"}, &m_rendererConfig.usePrimitiveCulling);
  m_info.parameterRegistry->add({"twopassculling"}, &m_rendererConfig.useTwoPassCulling);
  m_info.parameterRegistry->add({"forcedinvisculling"}, &m_rendererConfig.useForcedInvisibleCulling);
  m_info.parameterRegistry->add({"separategroups"}, &m_rendererConfig.useSeparateGroups);
  m_info.parameterRegistry->add({"sharingpushculled"}, &m_frameConfig.sharingPushCulled);
  m_info.parameterRegistry->add({"sharingenabledlevels"}, &m_frameConfig.sharingEnabledLevels);
  m_info.parameterRegistry->add({"sharingtolerantlevels"}, &m_frameConfig.sharingTolerantLevels);
  m_info.parameterRegistry->add({"cachingenabledlevels"}, &m_frameConfig.cachingEnabledLevels);
  m_info.parameterRegistry->add({"instancesorting"}, &m_rendererConfig.useSorting);
  m_info.parameterRegistry->add({"renderclusterbits"}, &m_rendererConfig.numRenderClusterBits);
  m_info.parameterRegistry->add({"rendertraversalbits"}, &m_rendererConfig.numTraversalTaskBits);
  m_info.parameterRegistry->add({"visualize"}, &m_frameConfig.visualize);
  m_info.parameterRegistry->add({"swraster"}, &m_rendererConfig.useComputeRaster);
  m_info.parameterRegistry->add({"adaptiveraster"}, &m_rendererConfig.useAdaptiveRasterRouting);
  m_info.parameterRegistry->add({"swrasterdensity"}, &m_frameConfig.swRasterTriangleDensityThreshold);
  m_info.parameterRegistry->add({"swrasterfeedback"}, &m_frameConfig.swRasterFeedbackEnabled);
  m_info.parameterRegistry->add({"swrastertargetshare"}, &m_frameConfig.swRasterFeedbackTargetTriangleShare);
  m_info.parameterRegistry->add({"renderstats"}, &m_rendererConfig.useRenderStats);
  m_info.parameterRegistry->add({"extmeshshader"}, &m_rendererConfig.useEXTmeshShader);
  m_info.parameterRegistry->add({"forcepreprocessmegabytes"}, (uint32_t*)&m_sceneLoaderConfig.forcePreprocessMiB);
  m_info.parameterRegistry->add({"facetshading"}, &m_tweak.facetShading);
  m_info.parameterRegistry->add({"flipwinding"}, &m_rendererConfig.flipWinding);
  m_info.parameterRegistry->add({"forcetwosided"}, &m_rendererConfig.forceTwoSided);
  m_info.parameterRegistry->add({"autosharing", "automatically set blas sharing based on scene's instancing usage. default true"},&m_tweak.autoSharing);
  m_info.parameterRegistry->add({"autosavecache", "automatically store cache file for loaded scene. default true"},&m_sceneLoaderConfig.autoSaveCache);
  m_info.parameterRegistry->add({"autoloadcache", "automatically load cache file if found. default true"},&m_sceneLoaderConfig.autoLoadCache);
  m_info.parameterRegistry->add({"mappedcache", "work from memory mapped cache file, otherwise load to sysmem. default false"},&m_sceneLoaderConfig.memoryMappedCache);
  m_info.parameterRegistry->add({"processingonly", "directly terminate app once cache file was saved. default false"},&m_sceneLoaderConfig.processingOnly);
  m_info.parameterRegistry->add({"processingpartial", "in processingonly mode also allow partial/resuming processing. default false"},&m_sceneLoaderConfig.processingAllowPartial);
  m_info.parameterRegistry->add({"processingmode", "0 auto, -1 inner (within geometry), +1 outer (over geometries) parallelism. default 0"},&m_sceneLoaderConfig.processingMode);
  m_info.parameterRegistry->add({"processingthreadpct", "float percentage of threads during initial file load and processing into lod clusters, default 0.5 == 50 %"},&m_sceneLoaderConfig.processingThreadsPct);
  m_info.parameterRegistry->add({"compressed"}, &m_sceneConfig.useCompressedData);
  m_info.parameterRegistry->add({"compressedpositionbits"}, &m_sceneConfig.compressionPosDropBits);
  m_info.parameterRegistry->add({"compressedtexcoordbits"}, &m_sceneConfig.compressionTexDropBits);
  m_info.parameterRegistry->add({"cachesuffix", "default is .zippp"}, &m_sceneCacheSuffix);
  {
    // HACK as zorah.cfg ships with some deprecated settings
    static bool dummy;
    m_info.parameterRegistry->add({"twosided", "deprecated - now detecting doubleSided materials - there is a new forcetwosided"},
                                  &dummy);
  }

  m_frameConfig.frameConstants                         = {};
  m_frameConfig.frameConstants.wireThickness           = 2.f;
  m_frameConfig.frameConstants.wireSmoothing           = 1.f;
  m_frameConfig.frameConstants.wireColor               = {118.f / 255.f, 185.f / 255.f, 0.f};
  m_frameConfig.frameConstants.wireStipple             = 0;
  m_frameConfig.frameConstants.wireBackfaceColor       = {0.5f, 0.5f, 0.5f};
  m_frameConfig.frameConstants.wireStippleRepeats      = 5;
  m_frameConfig.frameConstants.wireStippleLength       = 0.5f;
  m_frameConfig.frameConstants.doShadow                = 1;
  m_frameConfig.frameConstants.doWireframe             = 0;
  m_frameConfig.frameConstants.ambientOcclusionRadius  = 0.1f;
  m_frameConfig.frameConstants.ambientOcclusionSamples = 2;
  m_frameConfig.frameConstants.visualize               = VISUALIZE_LOD;
  m_frameConfig.frameConstants.facetShading            = 1;
  m_frameConfig.frameConstants.lightMixer = 0.5f;
  m_frameConfig.frameConstants.skyParams  = {};
  m_frameConfig.frameConstants.time = 0.0f;
  m_frameConfig.frameConstants.deltaTime = 0.0f;
  m_frameConfig.frameConstants.lodTransitionSpeed = 1.0f;
  m_frameConfig.swRasterThresholdEffective = m_frameConfig.swRasterThreshold;
  m_frameConfig.swRasterTriangleDensityThresholdEffective = m_frameConfig.swRasterTriangleDensityThreshold;
  m_lastAmbientOcclusionSamples = m_frameConfig.frameConstants.ambientOcclusionSamples;
  m_sceneLoaderConfig.progressPct = &m_sceneProgress;
}

// 初始化场景
// 加载指定路径的场景文件，支持.gltf、.glb和.cfg格式
// 参数: filePath - 场景文件路径
//       cacheSuffix - 缓存文件后缀
//       configChange - 是否仅更改配置而不重新加载场景
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

// 初始化渲染场景
// 创建并初始化RenderScene对象，处理场景的渲染相关设置
// 如果预加载失败，会自动尝试使用流式加载
glm::mat4 LodClusters::getModelMatrix(const ModelAsset& model) const
{
  glm::mat4 matrix(1.0f);
  matrix = glm::translate(matrix, model.translate);
  matrix = glm::rotate(matrix, glm::radians(model.rotateDeg.x), glm::vec3(1.0f, 0.0f, 0.0f));
  matrix = glm::rotate(matrix, glm::radians(model.rotateDeg.y), glm::vec3(0.0f, 1.0f, 0.0f));
  matrix = glm::rotate(matrix, glm::radians(model.rotateDeg.z), glm::vec3(0.0f, 0.0f, 1.0f));
  matrix = glm::scale(matrix, glm::max(model.scale, glm::vec3(0.001f)));
  return matrix;
}

int LodClusters::findModelIndexForInstance(uint32_t instanceId) const
{
  for(int i = 0; i < int(m_modelAssets.size()); i++)
  {
    const ModelAsset& model = m_modelAssets[i];
    if(!model.visible || model.compositeInstanceCount == 0)
    {
      continue;
    }

    const uint32_t first = model.compositeInstanceOffset;
    const uint32_t last  = first + model.compositeInstanceCount;
    if(instanceId >= first && instanceId < last)
    {
      return i;
    }
  }

  return -1;
}

glm::vec3 LodClusters::getModelCenter(int modelIndex) const
{
  if(!m_scene || modelIndex < 0 || modelIndex >= int(m_modelAssets.size()))
  {
    return glm::vec3(0.0f);
  }

  const ModelAsset& model = m_modelAssets[modelIndex];
  if(!model.visible || model.compositeInstanceCount == 0)
  {
    return model.translate;
  }

  glm::vec3 bboxLo(std::numeric_limits<float>::max());
  glm::vec3 bboxHi(-std::numeric_limits<float>::max());
  bool      valid = false;

  for(uint32_t i = 0; i < model.compositeInstanceCount; i++)
  {
    const uint32_t instanceIndex = model.compositeInstanceOffset + i;
    if(instanceIndex >= m_scene->m_instances.size())
    {
      continue;
    }

    const Scene::Instance& instance = m_scene->m_instances[instanceIndex];
    if(instance.geometryID >= m_scene->getActiveGeometryCount())
    {
      continue;
    }

    const shaderio::BBox& bbox = m_scene->getActiveGeometry(instance.geometryID).bbox;
    for(int x = 0; x < 2; x++)
    {
      for(int y = 0; y < 2; y++)
      {
        for(int z = 0; z < 2; z++)
        {
          const glm::vec3 corner(x ? bbox.hi.x : bbox.lo.x, y ? bbox.hi.y : bbox.lo.y, z ? bbox.hi.z : bbox.lo.z);
          const glm::mat4& matrix =
              m_simulationBaseMatrices.size() == m_scene->m_instances.size() ? m_simulationBaseMatrices[instanceIndex] :
                                                                               instance.matrix;
          const glm::vec3 world = glm::vec3(matrix * glm::vec4(corner, 1.0f));
          bboxLo                = glm::min(bboxLo, world);
          bboxHi                = glm::max(bboxHi, world);
          valid                 = true;
        }
      }
    }
  }

  return valid ? (bboxLo + bboxHi) * 0.5f : model.translate;
}

bool LodClusters::addModelToProject(const std::filesystem::path& filePath)
{
  if(filePath.empty())
  {
    return false;
  }

  LOGI("Importing model: %s\n", nvutils::utf8FromPath(filePath).c_str());

  auto scene = std::make_unique<Scene>();
  m_sceneProgress = 0;
  m_sceneLoading  = true;
  Scene::Result result = scene->init(filePath, m_sceneConfig, m_sceneLoaderConfig, m_sceneCacheSuffix, false);
  m_sceneLoading = false;

  if(result != Scene::SCENE_RESULT_SUCCESS)
  {
    LOGW("Importing model failed: %s\n", nvutils::utf8FromPath(filePath).c_str());
    return false;
  }

  bool savedCache = false;
  if(!sceneClusterConfigMatches(*scene, m_sceneConfig))
  {
    LOGI("Cached cluster config mismatch for %s; rebuilding clusters and cache\n",
         nvutils::utf8FromPath(filePath).c_str());
    LOGI("  scene setting: %u triangles, %u vertices; loaded cache: %u triangles, %u vertices; loaded max: %u triangles, %u vertices\n",
         m_sceneConfig.clusterTriangles, m_sceneConfig.clusterVertices, scene->m_config.clusterTriangles,
         scene->m_config.clusterVertices, scene->m_maxClusterTriangles, scene->m_maxClusterVertices);

    scene = std::make_unique<Scene>();
    m_sceneProgress = 0;
    m_sceneLoading  = true;
    result = scene->init(filePath, m_sceneConfig, m_sceneLoaderConfig, m_sceneCacheSuffix, true);
    m_sceneLoading = false;

    if(result != Scene::SCENE_RESULT_SUCCESS)
    {
      LOGW("Rebuilding model clusters failed: %s\n", nvutils::utf8FromPath(filePath).c_str());
      return false;
    }

    savedCache = scene->saveCache();

    if(!sceneClusterConfigMatches(*scene, m_sceneConfig))
    {
      LOGW("Rebuilt clusters still differ from requested limits: requested %uT/%uV, built max %uT/%uV\n",
           m_sceneConfig.clusterTriangles, m_sceneConfig.clusterVertices, scene->m_maxClusterTriangles,
           scene->m_maxClusterVertices);
    }
  }

  if(!savedCache && !scene->m_loadedFromCache && m_sceneLoaderConfig.autoSaveCache)
  {
    scene->saveCache();
  }

  ModelAsset model;
  model.filePath = filePath;
  model.name     = filePath.stem().string();
  model.scene    = std::move(scene);

  m_modelAssets.push_back(std::move(model));
  m_selectedModel = int(m_modelAssets.size()) - 1;
  rebuildProjectScene(true);
  return true;
}

void LodClusters::clearProject()
{
  deinitRenderer();
  deinitRenderScene();
  m_scene = nullptr;
  m_modelAssets.clear();
  m_selectedModel = -1;
  m_projectFilePath.clear();
  m_sceneFilePath.clear();
  m_renderSceneCanPreload = false;

  m_scene = std::make_unique<Scene>();
  m_scene->initEmpty(m_sceneConfig);
  captureSimulationBase();
}

void LodClusters::rebuildProjectScene(bool resetCamera)
{
  deinitRenderer();
  deinitRenderScene();

  m_scene = std::make_unique<Scene>();
  m_scene->initEmpty(m_sceneConfig);

  for(ModelAsset& model : m_modelAssets)
  {
    if(model.visible && model.scene)
    {
      model.compositeGeometryOffset = uint32_t(m_scene->getActiveGeometryCount());
      model.compositeInstanceOffset = uint32_t(m_scene->m_instances.size());
      model.compositeInstanceCount  = uint32_t(model.scene->m_instances.size());
      m_scene->appendScene(*model.scene, getModelMatrix(model));
    }
    else
    {
      model.compositeGeometryOffset = 0;
      model.compositeInstanceOffset = 0;
      model.compositeInstanceCount  = 0;
    }
  }

  m_renderSceneCanPreload = m_scene->getActiveGeometryCount() > 0
                            && ScenePreloaded::canPreload(m_resources.getDeviceLocalHeapSize(), m_scene.get());

  if(m_scene->m_instances.empty())
  {
    captureSimulationBase();
    return;
  }

  initRenderScene();
  initRenderer(m_tweak.renderer);

  if(resetCamera || m_frames == 0)
  {
    postInitNewScene();
  }
  else
  {
    m_frameConfig.frameConstants.sceneSize = glm::length(m_scene->m_bbox.hi - m_scene->m_bbox.lo);
    captureSimulationBase();
  }

  m_tweakLast           = m_tweak;
  m_rendererConfigLast  = m_rendererConfig;
  m_streamingConfigLast = m_streamingConfig;
  m_sceneConfigLast     = m_sceneConfig;
  m_sceneGridConfigLast = m_sceneGridConfig;
}

void LodClusters::updateModelTransform(int modelIndex)
{
  if(!m_scene || !m_renderer || modelIndex < 0 || modelIndex >= int(m_modelAssets.size()))
  {
    return;
  }

  ModelAsset& model = m_modelAssets[modelIndex];
  if(!model.visible || !model.scene || !model.compositeInstanceCount)
  {
    rebuildProjectScene(false);
    return;
  }

  const glm::mat4 modelMatrix = getModelMatrix(model);
  for(uint32_t i = 0; i < model.compositeInstanceCount; i++)
  {
    Scene::Instance instance = model.scene->m_instances[i];
    instance.geometryID += model.compositeGeometryOffset;
    instance.matrix = modelMatrix * instance.matrix;
    m_scene->m_instances[model.compositeInstanceOffset + i] = instance;
  }

  m_scene->refitBounds();
  m_frameConfig.frameConstants.sceneSize = glm::length(m_scene->m_bbox.hi - m_scene->m_bbox.lo);
  captureSimulationBase();
  m_renderer->updateRenderInstances(m_resources, *m_scene, m_rendererConfig);
}

bool LodClusters::saveProjectFile(const std::filesystem::path& filePath)
{
  if(filePath.empty())
  {
    return false;
  }

  std::ofstream out(filePath);
  if(!out)
  {
    LOGW("Saving project failed: %s\n", nvutils::utf8FromPath(filePath).c_str());
    return false;
  }

  out << "lodclusters_project 1\n";
  out << "model_count " << m_modelAssets.size() << "\n";
  for(const ModelAsset& model : m_modelAssets)
  {
    out << "model " << std::quoted(nvutils::utf8FromPath(model.filePath)) << "\n";
    out << "name " << std::quoted(model.name) << "\n";
    out << "visible " << (model.visible ? 1 : 0) << "\n";
    out << "translate " << model.translate.x << " " << model.translate.y << " " << model.translate.z << "\n";
    out << "rotate " << model.rotateDeg.x << " " << model.rotateDeg.y << " " << model.rotateDeg.z << "\n";
    out << "scale " << model.scale.x << " " << model.scale.y << " " << model.scale.z << "\n";
    out << "end_model\n";
  }

  m_projectFilePath = filePath;
  LOGI("Project saved: %s\n", nvutils::utf8FromPath(filePath).c_str());
  return true;
}

bool LodClusters::loadProjectFile(const std::filesystem::path& filePath)
{
  std::ifstream in(filePath);
  if(!in)
  {
    LOGW("Loading project failed: %s\n", nvutils::utf8FromPath(filePath).c_str());
    return false;
  }

  std::string token;
  int         version = 0;
  in >> token >> version;
  if(token != "lodclusters_project" || version != 1)
  {
    LOGW("Unsupported project file: %s\n", nvutils::utf8FromPath(filePath).c_str());
    return false;
  }

  clearProject();

  size_t modelCount = 0;
  in >> token >> modelCount;
  if(token != "model_count")
  {
    return false;
  }

  for(size_t i = 0; i < modelCount; i++)
  {
    std::string pathString;
    std::string name;
    bool        visible = true;
    glm::vec3   translate(0.0f);
    glm::vec3   rotateDeg(0.0f);
    glm::vec3   scale(1.0f);

    in >> token >> std::quoted(pathString);
    if(token != "model")
      return false;

    while(in >> token)
    {
      if(token == "end_model")
      {
        break;
      }
      if(token == "name")
      {
        in >> std::quoted(name);
      }
      else if(token == "visible")
      {
        int visibleInt = 1;
        in >> visibleInt;
        visible = visibleInt != 0;
      }
      else if(token == "translate")
      {
        in >> translate.x >> translate.y >> translate.z;
      }
      else if(token == "rotate")
      {
        in >> rotateDeg.x >> rotateDeg.y >> rotateDeg.z;
      }
      else if(token == "scale")
      {
        in >> scale.x >> scale.y >> scale.z;
      }
    }

    auto scene = std::make_unique<Scene>();
    std::filesystem::path modelPath = nvutils::pathFromUtf8(pathString);
    m_sceneProgress = 0;
    m_sceneLoading  = true;
    Scene::Result result = scene->init(modelPath, m_sceneConfig, m_sceneLoaderConfig, m_sceneCacheSuffix, false);
    m_sceneLoading = false;
    if(result != Scene::SCENE_RESULT_SUCCESS)
    {
      LOGW("Skipping model from project: %s\n", pathString.c_str());
      continue;
    }

    if(!sceneClusterConfigMatches(*scene, m_sceneConfig))
    {
      LOGI("Cached cluster config mismatch for %s; rebuilding clusters and cache\n", pathString.c_str());
      scene = std::make_unique<Scene>();
      m_sceneProgress = 0;
      m_sceneLoading  = true;
      result = scene->init(modelPath, m_sceneConfig, m_sceneLoaderConfig, m_sceneCacheSuffix, true);
      m_sceneLoading = false;
      if(result != Scene::SCENE_RESULT_SUCCESS)
      {
        LOGW("Skipping model after failed cluster rebuild: %s\n", pathString.c_str());
        continue;
      }
      scene->saveCache();
    }

    ModelAsset model;
    model.filePath  = modelPath;
    model.name      = name.empty() ? modelPath.stem().string() : name;
    model.visible   = visible;
    model.translate = translate;
    model.rotateDeg = rotateDeg;
    model.scale     = scale;
    model.scene     = std::move(scene);
    m_modelAssets.push_back(std::move(model));
  }

  m_projectFilePath = filePath;
  m_selectedModel   = m_modelAssets.empty() ? -1 : 0;
  rebuildProjectScene(true);
  LOGI("Project loaded: %s\n", nvutils::utf8FromPath(filePath).c_str());
  return true;
}

void LodClusters::initRenderScene()
{
  assert(m_scene);

  m_renderScene = std::make_unique<RenderScene>();

  bool success = m_renderScene->init(&m_resources, m_scene.get(), m_streamingConfig, m_tweak.useStreaming);

  // if preload fails, try streaming
  if(!m_tweak.useStreaming && !success)
  {
    // override to use streaming
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

void LodClusters::deinitRenderScene()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));
  if(m_renderScene)
  {
    m_renderScene->deinit();
    m_renderScene = nullptr;
  }
}

void LodClusters::deinitScene()
{
  deinitRenderScene();

  if(m_scene)
  {
    m_scene->deinit();
    m_scene = nullptr;
  }
  m_modelAssets.clear();
  m_selectedModel = -1;
}

void LodClusters::onResize(VkCommandBuffer cmd, const VkExtent2D& size)
{
  m_windowSize = size;
  m_resources.initFramebuffer(m_windowSize, m_tweak.supersample);
  updateImguiImage();
  if(m_renderer)
  {
    m_renderer->updatedFrameBuffer(m_resources, *m_renderScene);
    m_rendererFboChangeID = m_resources.m_fboChangeID;
  }
}

void LodClusters::updateImguiImage()
{
  if(m_imguiTexture)
  {
    ImGui_ImplVulkan_RemoveTexture(m_imguiTexture);
    m_imguiTexture = nullptr;
  }

  VkImageView imageView = m_resources.m_frameBuffer.useResolved ? m_resources.m_frameBuffer.imgColorResolved.descriptor.imageView :
                                                                  m_resources.m_frameBuffer.imgColor.descriptor.imageView;

  assert(imageView);

  m_imguiTexture = ImGui_ImplVulkan_AddTexture(m_imguiSampler, imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void LodClusters::resetSwRasterFeedback()
{
  m_swRasterFeedback.initialized            = false;
  m_swRasterFeedback.lastBaseExtent         = m_frameConfig.swRasterThreshold;
  m_swRasterFeedback.lastBaseDensity        = m_frameConfig.swRasterTriangleDensityThreshold;
  m_swRasterFeedback.effectiveExtent        = m_frameConfig.swRasterThreshold;
  m_swRasterFeedback.effectiveDensity       = m_frameConfig.swRasterTriangleDensityThreshold;
  m_swRasterFeedback.emaSwClusterShare      = 0.0f;
  m_swRasterFeedback.emaSwTriangleShare     = 0.0f;
  m_swRasterFeedback.emaSwTrianglesPerCluster = 0.0f;

  m_frameConfig.swRasterThresholdEffective = m_frameConfig.swRasterThreshold;
  m_frameConfig.swRasterTriangleDensityThresholdEffective = m_frameConfig.swRasterTriangleDensityThreshold;
}

void LodClusters::updateSwRasterFeedback()
{
  const float baseExtent  = std::max(m_frameConfig.swRasterThreshold, 1.0f);
  const float baseDensity = std::max(m_frameConfig.swRasterTriangleDensityThreshold, 0.01f);

  bool feedbackActive = m_renderer && m_rendererConfig.useComputeRaster && m_rendererConfig.useAdaptiveRasterRouting
                        && m_frameConfig.swRasterFeedbackEnabled;

  if(!feedbackActive)
  {
    resetSwRasterFeedback();
    return;
  }

  if(!m_swRasterFeedback.initialized || m_swRasterFeedback.lastBaseExtent != baseExtent
     || m_swRasterFeedback.lastBaseDensity != baseDensity)
  {
    resetSwRasterFeedback();
    m_swRasterFeedback.initialized = true;
  }

  shaderio::Readback readback;
  m_resources.getReadbackData(readback);

  const uint64_t totalClusters  = uint64_t(readback.numRenderedClusters) + uint64_t(readback.numRenderedClustersSW);
  const uint64_t totalTriangles = uint64_t(readback.numRenderedTriangles) + uint64_t(readback.numRenderedTrianglesSW);

  if(totalClusters == 0 || totalTriangles == 0)
  {
    m_frameConfig.swRasterThresholdEffective = m_swRasterFeedback.effectiveExtent;
    m_frameConfig.swRasterTriangleDensityThresholdEffective = m_swRasterFeedback.effectiveDensity;
    return;
  }

  const float alpha = 0.2f;
  const float swClusterShare = float(readback.numRenderedClustersSW) / float(totalClusters);
  const float swTriangleShare = float(readback.numRenderedTrianglesSW) / float(totalTriangles);
  const float swTrianglesPerCluster =
      readback.numRenderedClustersSW ? float(readback.numRenderedTrianglesSW) / float(readback.numRenderedClustersSW) : 0.0f;

  m_swRasterFeedback.emaSwClusterShare =
      m_swRasterFeedback.emaSwClusterShare * (1.0f - alpha) + swClusterShare * alpha;
  m_swRasterFeedback.emaSwTriangleShare =
      m_swRasterFeedback.emaSwTriangleShare * (1.0f - alpha) + swTriangleShare * alpha;
  m_swRasterFeedback.emaSwTrianglesPerCluster =
      m_swRasterFeedback.emaSwTrianglesPerCluster * (1.0f - alpha) + swTrianglesPerCluster * alpha;

  const bool enoughSamples = totalClusters >= 64;
  if(enoughSamples)
  {
    const float targetTriangleShare = glm::clamp(m_frameConfig.swRasterFeedbackTargetTriangleShare, 0.02f, 0.75f);
    const float deadzone            = std::max(0.015f, targetTriangleShare * 0.15f);
    const float shareError          = m_swRasterFeedback.emaSwTriangleShare - targetTriangleShare;
    const float stepExtent          = 0.35f;
    const float stepDensity         = 0.04f;
    const float highTrianglesPerCluster = std::min(float(m_sceneConfig.clusterTriangles) * 0.55f, 96.0f);

    float errorScale = 0.0f;
    if(shareError > deadzone)
    {
      errorScale = glm::clamp((shareError - deadzone) / std::max(1.0f - targetTriangleShare, 0.1f), 0.0f, 1.0f);
      m_swRasterFeedback.effectiveExtent -= stepExtent * (0.35f + errorScale);
      m_swRasterFeedback.effectiveDensity += stepDensity * (0.35f + errorScale);
    }
    else if(shareError < -deadzone)
    {
      errorScale = glm::clamp((-shareError - deadzone) / std::max(targetTriangleShare, 0.1f), 0.0f, 1.0f);
      m_swRasterFeedback.effectiveExtent += stepExtent * (0.35f + errorScale);
      m_swRasterFeedback.effectiveDensity -= stepDensity * (0.35f + errorScale);
    }

    if(m_swRasterFeedback.emaSwTrianglesPerCluster > highTrianglesPerCluster)
    {
      m_swRasterFeedback.effectiveExtent -= stepExtent * 0.5f;
      m_swRasterFeedback.effectiveDensity += stepDensity * 0.5f;
    }
  }

  const float minExtent   = std::max(1.0f, baseExtent * 0.5f);
  const float maxExtent   = std::max(baseExtent * 2.0f, baseExtent + 4.0f);
  const float minDensity  = std::max(0.05f, baseDensity * 0.35f);
  const float maxDensity  = std::max(baseDensity * 3.0f, baseDensity + 0.75f);

  m_swRasterFeedback.effectiveExtent  = glm::clamp(m_swRasterFeedback.effectiveExtent, minExtent, maxExtent);
  m_swRasterFeedback.effectiveDensity = glm::clamp(m_swRasterFeedback.effectiveDensity, minDensity, maxDensity);

  m_frameConfig.swRasterThresholdEffective = m_swRasterFeedback.effectiveExtent;
  m_frameConfig.swRasterTriangleDensityThresholdEffective = m_swRasterFeedback.effectiveDensity;
}

void LodClusters::onPreRender()
{
  updateSwRasterFeedback();
  m_profilerTimeline->frameAdvance();
}

void LodClusters::deinitRenderer()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));

  if(m_renderer)
  {
    m_renderer->deinit(m_resources);
    m_renderer = nullptr;
  }
}

void LodClusters::initRenderer(RendererType rtype)
{
  LOGI("Initializing renderer and compiling shaders\n");
  deinitRenderer();
  if(!m_renderScene)
    return;

  printf("init renderer %d\n", rtype);

  if(m_renderScene->useStreaming)
  {
    if(!m_renderScene->sceneStreaming.reloadShaders())
    {
      LOGE("RenderScene shaders failed\n");
      return;
    }
  }

  switch(rtype)
  {
    case RENDERER_RASTER_CLUSTERS_LOD:
      m_renderer = makeRendererRasterClustersLod();
      break;
  }

  if(m_renderer && !m_renderer->init(m_resources, *m_renderScene, m_rendererConfig))
  {
    m_renderer = nullptr;
    LOGE("Renderer init failed\n");
  }

  m_rendererFboChangeID = m_resources.m_fboChangeID;
}

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
  captureSimulationBase();
}

namespace {
glm::vec3 safeNormalize(const glm::vec3& v, const glm::vec3& fallback)
{
  const float len2 = glm::dot(v, v);
  return len2 > 0.000001f ? v * glm::inversesqrt(len2) : fallback;
}

bool sceneClusterConfigMatches(const Scene& scene, const SceneConfig& config)
{
  return scene.m_config.clusterTriangles == config.clusterTriangles && scene.m_config.clusterVertices == config.clusterVertices
         && scene.m_maxClusterTriangles <= config.clusterTriangles && scene.m_maxClusterVertices <= config.clusterVertices;
}
}

void LodClusters::captureSimulationBase()
{
  m_simulationBaseMatrices.clear();
  if(!m_scene)
  {
    return;
  }

  m_simulationBaseMatrices.reserve(m_scene->m_instances.size());
  for(const Scene::Instance& instance : m_scene->m_instances)
  {
    m_simulationBaseMatrices.push_back(instance.matrix);
  }

  const int maxInstance = std::max(0, int(m_scene->m_instances.size()) - 1);
  m_simulation.selected = glm::clamp(m_simulation.selected, 0, maxInstance);
  m_simulation.selectedGeom =
      glm::clamp(m_simulation.selectedGeom, 0, std::max(0, int(m_scene->getActiveGeometryCount()) - 1));
}

void LodClusters::resetSimulationPose()
{
  if(!m_scene)
  {
    return;
  }

  if(m_simulationBaseMatrices.size() != m_scene->m_instances.size())
  {
    captureSimulationBase();
  }

  for(size_t i = 0; i < m_scene->m_instances.size(); i++)
  {
    m_scene->m_instances[i].matrix = m_simulationBaseMatrices[i];
  }

  m_simulation.time            = 0.0f;
  m_simulation.manualTranslate = glm::vec3(0.0f);
  m_simulation.manualRotateDeg = glm::vec3(0.0f);
  m_simulation.manualScale     = glm::vec3(1.0f);
  uploadSimulationPose();
}

void LodClusters::uploadSimulationPose()
{
  if(m_renderer && m_scene)
  {
    m_renderer->updateRenderInstances(m_resources, *m_scene, m_rendererConfig);
    m_equalFrames = 0;
  }
}

void LodClusters::uploadSimulationPoseRange(uint32_t firstInstance, uint32_t instanceCount)
{
  if(m_renderer && m_scene)
  {
    m_renderer->updateRenderInstancesRange(m_resources, *m_scene, m_rendererConfig, firstInstance, instanceCount);
    m_equalFrames = 0;
  }
}

void LodClusters::updateSimulation(float deltaTime, bool forceUpdate)
{
  if(!m_scene || !m_renderer || m_scene->m_instances.empty())
  {
    return;
  }

  if(m_simulationBaseMatrices.size() != m_scene->m_instances.size())
  {
    captureSimulationBase();
    forceUpdate = true;
  }

  const bool animated = m_simulation.enabled && m_simulation.playing;
  if(!animated && !forceUpdate && !m_simulation.dirty)
  {
    return;
  }

  if(animated)
  {
    m_simulation.time += deltaTime * m_simulation.timeScale;
  }

  const glm::vec3 rotationAxis    = safeNormalize(m_simulation.rotationAxis, glm::vec3(0.0f, 1.0f, 0.0f));
  const glm::vec3 translationAxis = safeNormalize(m_simulation.translationAxis, glm::vec3(1.0f, 0.0f, 0.0f));

  const int maxInstance = int(m_scene->m_instances.size()) - 1;
  m_simulation.selected = glm::clamp(m_simulation.selected, 0, maxInstance);

  const float t = m_simulation.time;
  const bool  hasSelectedModel =
      m_selectedModel >= 0 && m_selectedModel < int(m_modelAssets.size()) && m_modelAssets[m_selectedModel].visible
      && m_modelAssets[m_selectedModel].compositeInstanceCount > 0;
  if(m_simulation.target == SIM_TARGET_MODEL && !hasSelectedModel)
  {
    m_simulation.dirty = false;
    return;
  }

  const bool needsSelectedModelCenter =
      m_simulation.target == SIM_TARGET_MODEL && (m_simulation.motion == SIM_MOTION_SPIN || m_simulation.motion == SIM_MOTION_WAVE);
  const glm::vec3 selectedModelCenter =
      needsSelectedModelCenter ? getModelCenter(m_selectedModel) : glm::vec3(0.0f);
  const auto      isSelectedModelInstance = [&](size_t instanceIndex) {
    if(!hasSelectedModel)
    {
      return false;
    }
    const ModelAsset& model = m_modelAssets[m_selectedModel];
    return instanceIndex >= model.compositeInstanceOffset
           && instanceIndex < size_t(model.compositeInstanceOffset + model.compositeInstanceCount);
  };

  uint32_t updateFirst = 0;
  uint32_t updateCount = uint32_t(m_scene->m_instances.size());
  bool     contiguousUpdate = false;

  if(m_simulation.target == SIM_TARGET_MODEL && hasSelectedModel)
  {
    const ModelAsset& model = m_modelAssets[m_selectedModel];
    updateFirst            = model.compositeInstanceOffset;
    updateCount            = model.compositeInstanceCount;
    contiguousUpdate       = true;
  }
  else if(m_simulation.target == SIM_TARGET_SELECTED)
  {
    updateFirst      = uint32_t(m_simulation.selected);
    updateCount      = 1;
    contiguousUpdate = true;
  }

  const size_t updateLast = std::min<size_t>(size_t(updateFirst) + updateCount, m_scene->m_instances.size());
  for(size_t i = updateFirst; i < updateLast; i++)
  {
    Scene::Instance& instance = m_scene->m_instances[i];
    const glm::mat4  base     = m_simulationBaseMatrices[i];

    bool affected = m_simulation.target == SIM_TARGET_ALL;
    affected = affected || (m_simulation.target == SIM_TARGET_SELECTED && int(i) == m_simulation.selected);
    affected = affected
               || (m_simulation.target == SIM_TARGET_GEOMETRY && int(instance.geometryID) == m_simulation.selectedGeom);
    affected = affected || (m_simulation.target == SIM_TARGET_MODEL && isSelectedModelInstance(i));

    glm::mat4 world = base;

    if(m_simulation.enabled && affected)
    {
      const Scene::GeometryView& geometry = m_scene->getActiveGeometry(instance.geometryID);
      const glm::vec3 objectCenter = (geometry.bbox.lo + geometry.bbox.hi) * 0.5f;
      const glm::vec3 center =
          m_simulation.target == SIM_TARGET_MODEL ? selectedModelCenter : glm::vec3(base * glm::vec4(objectCenter, 1.0f));
      const float     phase        = m_simulation.target == SIM_TARGET_MODEL ? 0.0f : float(i) * m_simulation.phaseStride;

      switch(m_simulation.motion)
      {
        case SIM_MOTION_SPIN:
        {
          const float angle = glm::radians(m_simulation.spinDegrees) * t + phase;
          world = glm::translate(glm::mat4(1.0f), center) * glm::rotate(glm::mat4(1.0f), angle, rotationAxis)
                  * glm::translate(glm::mat4(1.0f), -center) * base;
          break;
        }
        case SIM_MOTION_ORBIT:
        {
          const float angle = m_simulation.linearSpeed * t + phase;
          const glm::vec3 offset = glm::vec3(std::cos(angle), 0.0f, std::sin(angle)) * m_simulation.orbitRadius;
          world = glm::translate(glm::mat4(1.0f), offset) * base;
          break;
        }
        case SIM_MOTION_OSCILLATE:
        {
          const float wave = std::sin(m_simulation.linearSpeed * t + phase) * m_simulation.amplitude;
          world = glm::translate(glm::mat4(1.0f), translationAxis * wave) * base;
          break;
        }
        case SIM_MOTION_CONVEYOR:
        {
          const float travel = std::fmod(m_simulation.linearSpeed * t + phase, 2.0f) - 1.0f;
          world = glm::translate(glm::mat4(1.0f), translationAxis * travel * m_simulation.amplitude) * base;
          break;
        }
        case SIM_MOTION_WAVE:
        {
          const float wave  = std::sin(m_simulation.linearSpeed * t + phase) * m_simulation.amplitude;
          const float angle = glm::radians(m_simulation.spinDegrees) * t + phase;
          world = glm::translate(glm::mat4(1.0f), translationAxis * wave)
                  * glm::translate(glm::mat4(1.0f), center) * glm::rotate(glm::mat4(1.0f), angle, rotationAxis)
                  * glm::translate(glm::mat4(1.0f), -center) * base;
          break;
        }
      }
    }

    if(int(i) == m_simulation.selected)
    {
      glm::mat4 manual(1.0f);
      manual = glm::translate(manual, m_simulation.manualTranslate);
      manual = glm::rotate(manual, glm::radians(m_simulation.manualRotateDeg.x), glm::vec3(1.0f, 0.0f, 0.0f));
      manual = glm::rotate(manual, glm::radians(m_simulation.manualRotateDeg.y), glm::vec3(0.0f, 1.0f, 0.0f));
      manual = glm::rotate(manual, glm::radians(m_simulation.manualRotateDeg.z), glm::vec3(0.0f, 0.0f, 1.0f));
      manual = glm::scale(manual, glm::max(m_simulation.manualScale, glm::vec3(0.001f)));
      world  = manual * world;
    }

    instance.matrix = world;
  }

  if(contiguousUpdate)
  {
    uploadSimulationPoseRange(updateFirst, uint32_t(updateLast - updateFirst));
  }
  else
  {
    uploadSimulationPose();
  }
  m_simulation.dirty = false;
}

void LodClusters::onAttach(nvapp::Application* app)
{
  m_app = app;

  m_tweak.supersample = std::max(1, m_tweak.supersample);
  m_info.cameraManipulator->setMode(nvutils::CameraManipulator::Fly);
  m_renderer = nullptr;

  if(m_resources.m_supportsSmBuiltinsNV)
  {
    VkPhysicalDeviceProperties2 physicalProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceShaderSMBuiltinsPropertiesNV smProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV};
    physicalProperties.pNext = &smProperties;
    vkGetPhysicalDeviceProperties2(app->getPhysicalDevice(), &physicalProperties);
    // pseudo heuristic
    // larger GPUs seem better off with lower values
    if(smProperties.shaderSMCount * smProperties.shaderWarpsPerSM > 4096)
      m_frameConfig.traversalPersistentThreads = smProperties.shaderSMCount * smProperties.shaderWarpsPerSM * 2;
    else if(smProperties.shaderSMCount * smProperties.shaderWarpsPerSM > 2048 + 1024)
      m_frameConfig.traversalPersistentThreads = smProperties.shaderSMCount * smProperties.shaderWarpsPerSM * 4;
    else
      m_frameConfig.traversalPersistentThreads = smProperties.shaderSMCount * smProperties.shaderWarpsPerSM * 8;
  }

  {
    m_ui.enumAdd(GUI_RENDERER, RENDERER_RASTER_CLUSTERS_LOD, "Rasterization");
    m_ui.enumAdd(GUI_BUILDMODE, 0, "default");
    m_ui.enumAdd(GUI_BUILDMODE, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR, "fast build");
    m_ui.enumAdd(GUI_BUILDMODE, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR, "fast trace");

    {
      for(uint32_t i = 0; i < NUM_CLUSTER_CONFIGS; i++)
      {
        std::string enumStr = fmt::format("{}T_{}V", s_clusterInfos[i].tris, s_clusterInfos[i].verts);
        m_ui.enumAdd(GUI_MESHLET, s_clusterInfos[i].cfg, enumStr.c_str());
      }
    }

    m_ui.enumAdd(GUI_SUPERSAMPLE, 1, "none");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 2, "4x");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 720, "720p");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 1080, "1080p");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 1440, "1440p");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 2160, "2160p");
    m_ui.enumAdd(GUI_SUPERSAMPLE, 1024, "1024 sq");// 1024x1024 正方形分辨率
    m_ui.enumAdd(GUI_SUPERSAMPLE, 2048, "2048 sq");// 2048x2048 正方形分辨率
    m_ui.enumAdd(GUI_SUPERSAMPLE, 4096, "4096 sq");// 4096x4096 正方形分辨率
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_MATERIAL, "material");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_GREY, "grey");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_VIS_BUFFER, "visibility buffer");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_CLUSTER, "clusters");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_GROUP, "cluster groups");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_LOD, "lod levels");
    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_TRIANGLE, "triangles");
    // UI 可视化选项枚举添加完成
  }

  // Initialize core components

  m_profilerGpuTimer.init(m_profilerTimeline, app->getDevice(), app->getPhysicalDevice(), app->getQueue(0).familyIndex, true);
  m_resources.init(app->getDevice(), app->getPhysicalDevice(), app->getInstance(), app->getQueue(0), app->getQueue(1));

  {
    NVVK_CHECK(m_resources.m_samplerPool.acquireSampler(m_imguiSampler));
    NVVK_DBG_NAME(m_imguiSampler);
  }

  m_resources.initFramebuffer({128, 128}, m_tweak.supersample);
  updateImguiImage();

  setFromClusterConfig(m_sceneConfig, m_tweak.clusterConfig);

  if(!m_resources.m_supportsMeshShaderNV)
  {
    m_rendererConfig.useEXTmeshShader = true;
  }

  m_cameraStringCommandLine = m_cameraString;

  if(m_sceneFilePathDropNew.empty())
  {
    clearProject();
  }
  else
  {
    std::filesystem::path newFileDrop = m_sceneFilePathDropNew;
    onFileDrop(newFileDrop);
  }

  m_tweakLast          = m_tweak;
  m_sceneConfigLast    = m_sceneConfig;
  m_sceneConfigEdit    = m_sceneConfig;
  m_rendererConfigLast = m_rendererConfig;
}

void LodClusters::onDetach()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));

  deinitRenderer();
  deinitScene();

  m_resources.m_samplerPool.releaseSampler(m_imguiSampler);
  ImGui_ImplVulkan_RemoveTexture(m_imguiTexture);

  m_resources.deinit();

  m_profilerGpuTimer.deinit();
}

void LodClusters::saveCacheFile()
{
  if(m_selectedModel >= 0 && m_selectedModel < int(m_modelAssets.size()) && m_modelAssets[m_selectedModel].scene)
  {
    m_modelAssets[m_selectedModel].scene->saveCache();
  }
  else
  {
    for(ModelAsset& model : m_modelAssets)
    {
      if(model.scene)
      {
        model.scene->saveCache();
      }
    }
  }
}

void LodClusters::onFileDrop(const std::filesystem::path& filePath)
{
  if(filePath.empty())
    return;

  if(!m_sceneLoadFromConfig)
  {
    // avoid certain state to affect the new scene
    if(!m_sceneFilePathDropLast.empty() && filePath != m_sceneFilePathDropLast)
    {
      // reset grid parameter (in case scene is too large to be replicated)
      m_sceneGridConfig.numCopies                 = 1;
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

    // config parsing might change m_sceneFilePathDropNew
    // and m_cameraString
    m_info.parameterParser->parse(std::span(args), false, {}, {}, true);

    if(!m_cameraStringCommandLine.empty())
    {
      // override from command-line
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

  if(filePath.extension() == ".lscene")
  {
    loadProjectFile(filePath);
    m_sceneFilePathDropLast = filePath;
    m_sceneFilePathDropNew  = filePath;
    return;
  }

  addModelToProject(filePath);
}

void LodClusters::doProcessingOnly()
{
  setFromClusterConfig(m_sceneConfig, m_tweak.clusterConfig);
  assert(m_app == nullptr);
  m_scene = std::make_unique<Scene>();
  m_scene->init(m_sceneFilePathDropNew, m_sceneConfig, m_sceneLoaderConfig, m_sceneCacheSuffix, false);
}

void LodClusters::parameterSequenceCallback(const nvutils::ParameterSequencer::State& state)
{
  std::string message;
  message += fmt::format("MemoryReport {} \"{}\" = {{ \n", state.index, state.description);
  if(m_renderer)
  {
    Renderer::ResourceUsageInfo resourceActual   = m_renderer->getResourceUsage(false);
    Renderer::ResourceUsageInfo resourceReserved = m_renderer->getResourceUsage(true);
    message += fmt::format("Memory; Actual; Reserved;\n");
    message += fmt::format("Geometry; {}; {};\n", resourceActual.geometryMemBytes, resourceReserved.geometryMemBytes);
    if(m_renderScene->useStreaming)
    {
      StreamingStats stats;
      m_renderScene->sceneStreaming.getStats(stats);
      message += fmt::format("Resident; Actual; Reserved;\n");
      message += fmt::format("Groups; {}; {};\n", stats.residentGroups, stats.maxGroups);
      message += fmt::format("Clusters; {}; {};\n", stats.residentClusters, stats.maxClusters);
    }

    shaderio::Readback readback;
    m_resources.getReadbackData(readback);
    message += fmt::format("Traversal; Actual; Reserved;\n");
    message += fmt::format("Traversal Tasks; {}; {};\n", readback.numTraversalTasks, m_renderer->getMaxTraversalTasks());
    message += fmt::format("Traversal Clusters; {}; {};\n", readback.numRenderClusters, m_renderer->getMaxRenderClusters());

  }
  message += fmt::format("}}\n");

  nvutils::Logger::getInstance().log(nvutils::Logger::eSTATS, "%s", message.c_str());

  if(m_sequenceScreenshotMode != SCREENSHOT_OFF)
  {
    ScreenshotMode screenshotMode = m_sequenceScreenshotMode;
    if(m_app->isHeadless())
    {
      screenshotMode = SCREENSHOT_VIEWPORT;
    }

    std::string filename = fmt::format("screenshot_{}_{}.jpg", state.index, state.description);
    if(screenshotMode == SCREENSHOT_WINDOW)
    {
      m_app->saveScreenShot(std::filesystem::path(filename), 100);
    }
    else if(screenshotMode == SCREENSHOT_VIEWPORT)
    {
      m_app->saveImageToFile(m_resources.m_frameBuffer.useResolved ? m_resources.m_frameBuffer.imgColorResolved.image :
                                                                     m_resources.m_frameBuffer.imgColor.image,
                             m_resources.m_frameBuffer.windowSize, filename, 100, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
  }
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
        //m_info.cameraManipulator->setClipPlanes(glm::vec2(0.0001f, 10000.0f));
  }

}

void LodClusters::handleChanges()
{
  if(m_sceneLoading)
    return;

  if(m_sceneFilePathDropLast != m_sceneFilePathDropNew)
  {
    std::filesystem::path newFilePath = m_sceneFilePathDropNew;
    onFileDrop(newFilePath);
  }

  if(!m_resources.m_supportsMeshShaderNV)
  {
    m_rendererConfig.useEXTmeshShader = true;
  }


  if((m_frameConfig.visualize == VISUALIZE_VIS_BUFFER || m_frameConfig.visualize == VISUALIZE_DEPTH_ONLY)
     && m_rendererConfig.useShading)
  {
    m_rendererConfig.useShading = false;
  }
  if(!(m_frameConfig.visualize == VISUALIZE_VIS_BUFFER || m_frameConfig.visualize == VISUALIZE_DEPTH_ONLY)
     && !m_rendererConfig.useShading)
  {
    m_rendererConfig.useShading = true;
  }
  m_rendererConfig.useDepthOnly = m_frameConfig.visualize == VISUALIZE_DEPTH_ONLY;


  if(m_rendererConfig.useComputeRaster
     && (!m_rendererConfig.useSeparateGroups || !m_rendererConfig.useCulling || m_rendererConfig.useShading))
  {
    m_rendererConfig.useComputeRaster = false;
  }
  if(!m_rendererConfig.useComputeRaster)
  {
    m_rendererConfig.useAdaptiveRasterRouting = false;
  }


  bool frameBufferChanged = false;
  if(tweakChanged(m_tweak.supersample))
  {
    m_resources.initFramebuffer(m_windowSize, m_tweak.supersample);
    updateImguiImage();

    frameBufferChanged = true;
  }

  bool shaderChanged = false;
  if(m_reloadShaders)
  {
    shaderChanged   = true;
    m_reloadShaders = false;
  }

  bool sceneChanged = false;
  if(memcmp(&m_sceneConfig, &m_sceneConfigLast, sizeof(m_sceneConfig)))
  {
    sceneChanged = true;

    deinitRenderer();
    for(ModelAsset& model : m_modelAssets)
    {
      if(model.filePath.empty())
        continue;

      auto scene = std::make_unique<Scene>();
      if(scene->init(model.filePath, m_sceneConfig, m_sceneLoaderConfig, m_sceneCacheSuffix, true) == Scene::SCENE_RESULT_SUCCESS)
      {
        model.scene = std::move(scene);
      }
    }
    rebuildProjectScene(false);
  }

  if(!m_cameraString.empty() && m_cameraString != m_cameraStringLast)
  {
    applyCameraString();
  }

  bool sceneGridChanged = false;
  if(m_scene)
  {
    const bool hasRenderableScene = m_scene->getActiveGeometryCount() > 0 && !m_scene->m_instances.empty();
    if(!hasRenderableScene)
    {
      if(m_renderer)
      {
        deinitRenderer();
      }
      if(m_renderScene)
      {
        deinitRenderScene();
      }
      m_renderSceneCanPreload = false;
    }
    else
    {
    if(!m_renderScene)
    {
      sceneGridChanged = true;
    }

    if(!sceneChanged && memcmp(&m_sceneGridConfig, &m_sceneGridConfigLast, sizeof(m_sceneGridConfig)))
    {
      sceneGridChanged = true;

      deinitRenderer();
      m_scene->updateSceneGrid(m_sceneGridConfig);
      updatedSceneGrid();
      captureSimulationBase();
      m_simulation.dirty = true;
    }

    bool renderSceneChanged = false;
    bool streamingChanged   = tweakChanged(m_tweak.useStreaming)
                            || (memcmp(&m_streamingConfig, &m_streamingConfigLast, sizeof(m_streamingConfig)));
    if(sceneGridChanged || streamingChanged)
    {
      if(!sceneChanged || !sceneGridChanged)
      {
        deinitRenderer();
      }

      renderSceneChanged = true;
      deinitRenderScene();
      initRenderScene();

      if(streamingChanged)
      {
        m_streamGeometryHistogramMax = 0;
      }
    }
    // 检查场景、着色器或渲染场景是否有变化
    if(sceneChanged || shaderChanged || renderSceneChanged || tweakChanged(m_tweak.renderer) || tweakChanged(m_tweak.supersample)
       || rendererCfgChanged(m_rendererConfig.flipWinding) || rendererCfgChanged(m_rendererConfig.useDebugVisualization)
       || rendererCfgChanged(m_rendererConfig.useCulling) || rendererCfgChanged(m_rendererConfig.forceTwoSided)
       || rendererCfgChanged(m_rendererConfig.useSorting) || rendererCfgChanged(m_rendererConfig.numRenderClusterBits)
       || rendererCfgChanged(m_rendererConfig.numTraversalTaskBits) || rendererCfgChanged(m_rendererConfig.useShading)
       || rendererCfgChanged(m_rendererConfig.useRenderStats)
       || rendererCfgChanged(m_rendererConfig.useSeparateGroups) 
       || rendererCfgChanged(m_rendererConfig.useEXTmeshShader)
       || rendererCfgChanged(m_rendererConfig.useComputeRaster) || rendererCfgChanged(m_rendererConfig.useAdaptiveRasterRouting)
       || rendererCfgChanged(m_rendererConfig.usePrimitiveCulling)
        //|| rendererCfgChanged(m_rendererConfig.useComputeRaster) || rendererCfgChanged(m_rendererConfig.usePrimitiveCulling))
       || rendererCfgChanged(m_rendererConfig.useTwoPassCulling) || rendererCfgChanged(m_rendererConfig.useDepthOnly)
       || rendererCfgChanged(m_rendererConfig.useForcedInvisibleCulling))
    {

      initRenderer(m_tweak.renderer);
    }
    else if(m_renderer && frameBufferChanged)
    {
      m_renderer->updatedFrameBuffer(m_resources, *m_renderScene);
      m_rendererFboChangeID = m_resources.m_fboChangeID;
    }
    }
  }


  bool hadChange = shaderChanged || memcmp(&m_tweakLast, &m_tweak, sizeof(m_tweak))
                   || memcmp(&m_rendererConfigLast, &m_rendererConfig, sizeof(m_rendererConfig))
                   || memcmp(&m_sceneConfigLast, &m_sceneConfig, sizeof(m_sceneConfig))
                   || memcmp(&m_streamingConfigLast, &m_streamingConfig, sizeof(m_streamingConfig))
                   || memcmp(&m_sceneGridConfigLast, &m_sceneGridConfig, sizeof(m_sceneGridConfig));
  m_tweakLast           = m_tweak;
  m_rendererConfigLast  = m_rendererConfig;
  m_streamingConfigLast = m_streamingConfig;
  m_sceneConfigLast     = m_sceneConfig;
  m_sceneGridConfigLast = m_sceneGridConfig;

  if(hadChange)
  {
    m_equalFrames = 0;
    if(m_tweak.autoResetTimers)
    {
      m_info.profilerManager->resetFrameSections(8);
    }
  }
}

void LodClusters::applyCameraString()
{
  nvutils::CameraManipulator::Camera cam = m_info.cameraManipulator->getCamera();
  if(cam.setFromString(m_cameraString))
  {
    m_info.cameraManipulator->setCamera(cam);
    nvgui::SetHomeCamera(m_info.cameraManipulator->getCamera());
  }
  m_cameraStringLast = m_cameraString;
}

// 渲染函数
// 执行场景渲染，包括设置帧常量、更新相机矩阵、执行渲染器渲染等
// 参数: cmd - Vulkan命令缓冲区
void LodClusters::onRender(VkCommandBuffer cmd)
{
  double time = m_clock.getSeconds();
  static double lastTime = 0.0;
  float deltaTime = static_cast<float>(time - lastTime);
  lastTime = time;

  // 开始新帧
  m_resources.beginFrame(m_app->getFrameCycleIndex());

  // 设置窗口大小
  m_frameConfig.windowSize = m_windowSize;

  // 检查渲染器是否存在
  if(m_renderer)
  {
    // 检查帧缓冲区是否有变化
    if(m_rendererFboChangeID != m_resources.m_fboChangeID)
    {
      m_renderer->updatedFrameBuffer(m_resources, *m_renderScene);
      m_rendererFboChangeID = m_resources.m_fboChangeID;
    }

    updateSimulation(deltaTime);

    shaderio::FrameConstants& frameConstants = m_frameConfig.frameConstants;

    // for motion always use last
    frameConstants.viewProjMatrixPrev = frameConstants.viewProjMatrix;

    if(m_frames)
    {
      m_frameConfig.frameConstantsLast = m_frameConfig.frameConstants;
    }

    int supersample = m_tweak.supersample;
    //uint32_t windowWidth = m_resources.m_frameBuffer.windowSize.width;
    //uint32_t windowHeight = m_resources.m_frameBuffer.windowSize.height;
    uint32_t renderWidth  = m_resources.m_frameBuffer.renderSize.width;
    uint32_t renderHeight = m_resources.m_frameBuffer.renderSize.height;

    uint32_t targetWidth  = m_resources.m_frameBuffer.targetSize.width;
    uint32_t targetHeight = m_resources.m_frameBuffer.targetSize.height;
    // 设置帧常量的渲染属性
    frameConstants.facetShading = m_tweak.facetShading ? 1 : 0;
    frameConstants.visualize    = m_frameConfig.visualize;
    frameConstants.frame        = m_frames;
    
    // 更新时间相关字段用于LOD平滑过渡
    static float accumulatedTime = 0.0f;
    accumulatedTime += deltaTime;
    frameConstants.time = accumulatedTime;
    frameConstants.deltaTime = deltaTime;

    {
      frameConstants.visFilterClusterID  = ~0;
      frameConstants.visFilterInstanceID = ~0;
    }

    frameConstants.bgColor   = m_resources.m_bgColor;
    frameConstants.viewport  = glm::ivec2(renderWidth, renderHeight);
    frameConstants.viewportf = glm::vec2(renderWidth, renderHeight);
    //frameConstants.supersample = m_tweak.supersample;
    frameConstants.nearPlane = m_info.cameraManipulator->getClipPlanes().x;
    frameConstants.farPlane  = m_info.cameraManipulator->getClipPlanes().y;
    frameConstants.wUpDir    = m_info.cameraManipulator->getUp();
    frameConstants.fov = glm::radians(m_info.cameraManipulator->getFov());
    //glm::perspectiveRH_ZO(glm::radians(m_info.cameraManipulator->getFov()), float(windowWidth) / float(windowHeight),
    //glm::perspectiveRH_ZO(glm::radians(m_info.cameraManipulator->getFov()), float(targetWidth) / float(targetHeight),
    glm::mat4 projection = glm::perspectiveRH_ZO(frameConstants.fov, float(targetWidth) / float(targetHeight), frameConstants.farPlane, frameConstants.nearPlane);
    projection[1][1] *= -1;
    glm::mat4 view  = m_info.cameraManipulator->getViewMatrix();
    glm::mat4 viewI = glm::inverse(view);

    frameConstants.viewProjMatrix  = projection * view;
    frameConstants.viewProjMatrixI = glm::inverse(frameConstants.viewProjMatrix);
    frameConstants.viewMatrix      = view;
    frameConstants.viewMatrixI     = viewI;
    frameConstants.projMatrix      = projection;
    frameConstants.projMatrixI     = glm::inverse(projection);

    glm::mat4 viewNoTrans         = view;
    viewNoTrans[3]                = {0.0f, 0.0f, 0.0f, 1.0f};
    frameConstants.skyProjMatrixI = glm::inverse(projection * viewNoTrans);

    glm::vec4 hPos   = projection * glm::vec4(1.0f, 1.0f, -frameConstants.farPlane, 1.0f);
    glm::vec2 hCoord = glm::vec2(hPos.x / hPos.w, hPos.y / hPos.w);
    glm::vec2 dim    = glm::abs(hCoord);

    // helper to quickly get footprint of a point at a given distance
    //
    // __.__hPos (far plane is width x height)
    // \ | /
    //  \|/
    //   x camera
    //
    // here: viewPixelSize / point.w = size of point in pixels
    // * 0.5f because renderWidth/renderHeight represents [-1,1] but we need half of frustum
    frameConstants.viewPixelSize = dim * (glm::vec2(float(renderWidth), float(renderHeight)) * 0.5f) * frameConstants.farPlane;
    // here: viewClipSize / point.w = size of point in clip-space units
    // no extra scale as half clip space is 1.0 in extent
    frameConstants.viewClipSize = dim * frameConstants.farPlane;

    frameConstants.viewPos = frameConstants.viewMatrixI[3];  // position of eye in the world
    frameConstants.viewDir = -viewI[2];

    frameConstants.viewPlane   = frameConstants.viewDir;
    frameConstants.viewPlane.w = -glm::dot(glm::vec3(frameConstants.viewPos), glm::vec3(frameConstants.viewDir));

    frameConstants.wLightPos = frameConstants.viewMatrixI[3];  // place light at position of eye in the world

    {
      // hiz
      //m_resources.m_hizUpdate.farInfo.getShaderFactors((float*)&frameConstants.hizSizeFactors);
      //frameConstants.hizSizeMax = m_resources.m_hizUpdate.farInfo.getSizeMax();
      m_resources.m_hizUpdate[0].farInfo.getShaderFactors((float*)&frameConstants.hizSizeFactors);
      frameConstants.hizSizeMax = m_resources.m_hizUpdate[0].farInfo.getSizeMax();
      // 注：在 resources.hpp 中定义了：
      // NVHizVK::Update m_hizUpdate[2];
      // [0] = 前一帧 HiZ
      // [1] = 当前帧 HiZ（用于时间上的平滑过渡）
    }


    if(!m_frames)
    {
      // on first frame replicate last
      m_frameConfig.frameConstantsLast = m_frameConfig.frameConstants;
    }

    if(!m_frameConfig.freezeLoD)
    {
      m_frameConfig.traversalViewMatrix = m_frameConfig.frameConstants.viewMatrix;
    }
    if(!m_frameConfig.freezeCulling)
    {
      m_frameConfig.cullViewProjMatrix     = m_frameConfig.frameConstants.viewProjMatrix;
      m_frameConfig.cullViewProjMatrixLast = m_frameConfig.frameConstantsLast.viewProjMatrix;
    }

    if(m_frames)
    {
      shaderio::FrameConstants frameCurrent = m_frameConfig.frameConstants;

      if(memcmp(&frameCurrent, &m_frameConfig.frameConstantsLast, sizeof(shaderio::FrameConstants)))
        m_equalFrames = 0;
      else
        m_equalFrames++;
    }

    m_renderer->render(cmd, m_resources, *m_renderScene, m_frameConfig, m_profilerGpuTimer);
  }
  else
  {
    m_resources.emptyFrame(cmd, m_frameConfig, m_profilerGpuTimer);
  }

  {
    m_resources.postProcessFrame(cmd, m_frameConfig, m_profilerGpuTimer);
  }

  m_resources.endFrame();

  // signal new semaphore state with this command buffer's submit
  VkSemaphoreSubmitInfo semSubmit = m_resources.m_queueStates.primary.advanceSignalSubmit(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);
  m_app->addSignalSemaphore(semSubmit);
  // but also enqueue waits if there are any
  while(!m_resources.m_queueStates.primary.m_pendingWaits.empty())
  {
    m_app->addWaitSemaphore(m_resources.m_queueStates.primary.m_pendingWaits.back());
    m_resources.m_queueStates.primary.m_pendingWaits.pop_back();
  }

  m_lastTime = time;
  m_frames++;
}

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

bool LodClusters::isPickingValid(const shaderio::Readback& readback)
{
  return readback._packedDepth0 != 0u;
}

} 
