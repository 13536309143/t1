//==============================================================================
// 文件：src/app/lodclusters_config.cpp
// 模块定位：构造函数和默认参数注册实现，定义应用可从命令行、配置文件和 UI 调整的主要实验变量。
// 数据流：输入是 Info 中的参数注册器、解析器和 性能分析器；输出是默认 FrameConfig、SceneConfig、RendererConfig 与 StreamingConfig。
// 方法说明：该文件把渲染实验的控制变量显式参数化，便于复现实验、批处理序列和性能对比。
// 正确性约束：默认值必须与 着色器 侧常量和 UI 选项保持一致；新增参数需要同步考虑配置加载、序列化和统计输出。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include "lodclusters.hpp"

bool g_verbose = false;


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace lodclusters {


LodClusters::LodClusters(const Info& info)

    : m_info(info)
{
  nvutils::ProfilerTimeline::CreateInfo createInfo;
  createInfo.name = "graphics";

  m_profilerTimeline = m_info.profilerManager->createTimeline(createInfo);
  m_info.parameterRegistry->add({"scene"}, {".gltf", ".glb", ".cfg"}, &m_sceneFilePathDropNew);
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
  m_info.parameterRegistry->add({"assemblymininstances"}, &m_sceneConfig.assemblyCullingMinInstances);
  m_info.parameterRegistry->add({"assemblylodpixels"}, &m_sceneConfig.assemblyLodPixelThreshold);
  m_info.parameterRegistry->add({"meshoptfillweight"}, &m_sceneConfig.meshoptFillWeight);
  m_info.parameterRegistry->add({"learnedimportance"}, &m_sceneConfig.learnedImportanceEnable);
  m_info.parameterRegistry->add({"learnedstrength"}, &m_sceneConfig.learnedImportanceStrength);
  m_info.parameterRegistry->add({"learnedprotect"}, &m_sceneConfig.learnedImportanceProtectThreshold);
  m_info.parameterRegistry->add({"learnedtargetboost"}, &m_sceneConfig.learnedImportanceTargetBoost);
  m_info.parameterRegistry->add({"learnederrorscale"}, &m_sceneConfig.learnedImportanceErrorScale);
  m_info.parameterRegistry->add({"learnedtopologyedges"}, &m_sceneConfig.learnedImportanceTopologyEdgeLimit);

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

}
