//==============================================================================
// 文件：src/app/lodclusters_runtime.cpp
// 模块定位：每帧运行时调度实现，负责配置变更合并、帧常量更新、渲染调用、后处理和时间线推进。
// 数据流：输入是 UI/参数改变、相机状态、窗口尺寸和上一帧 回读数据；输出是当前帧 FrameConstants 与 renderer 命令流。
// 方法说明：该文件实现“帧级状态归约”：把多个来源的可变状态规约成一次稳定的 GPU 提交，保证实验参数可追踪。
// 正确性约束：冻结剔除或 LOD 时必须复用上一帧矩阵；SW raster feedback 只能调节 effective 阈值，不能改写用户配置基准值。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <volk.h>
#include <nvgui/camera.hpp>
#include "lodclusters.hpp"


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace lodclusters {


// 函数：LodClusters::resetSwRasterFeedback。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
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


// 函数：LodClusters::updateSwRasterFeedback。根据最新状态刷新缓存数据、GPU 地址、描述符或统计信息。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：更新函数负责把“旧状态”推进到“当前状态”，因此要避免部分更新造成 CPU/GPU 视图不一致。
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


// 函数：LodClusters::onPreRender。录制或执行渲染相关工作，把准备好的数据提交到当前渲染阶段。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：渲染函数通常处于帧级关键路径，必须尊重前序计算阶段写出的计数、地址和同步屏障。
void LodClusters::onPreRender()
{

  updateSwRasterFeedback();

  m_profilerTimeline->frameAdvance();
}


// 函数：LodClusters::handleChanges。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
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

    initScene(m_sceneFilePath, m_scene->m_cacheSuffix, true);
  }

  if(!m_cameraString.empty() && m_cameraString != m_cameraStringLast)
  {

    applyCameraString();
  }

  bool sceneGridChanged = false;
  if(m_scene)
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


// 函数：LodClusters::applyCameraString。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
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


// 函数：LodClusters::onRender。录制或执行渲染相关工作，把准备好的数据提交到当前渲染阶段。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：渲染函数通常处于帧级关键路径，必须尊重前序计算阶段写出的计数、地址和同步屏障。
void LodClusters::onRender(VkCommandBuffer cmd)
{

  double time = m_clock.getSeconds();
  static double lastTime = 0.0;
  float deltaTime = static_cast<float>(time - lastTime);
  lastTime = time;


  m_resources.beginFrame(m_app->getFrameCycleIndex());


  m_frameConfig.windowSize = m_windowSize;


  if(m_renderer)
  {

    if(m_rendererFboChangeID != m_resources.m_fboChangeID)
    {

      m_renderer->updatedFrameBuffer(m_resources, *m_renderScene);
      m_rendererFboChangeID = m_resources.m_fboChangeID;
    }


    shaderio::FrameConstants& frameConstants = m_frameConfig.frameConstants;


    frameConstants.viewProjMatrixPrev = frameConstants.viewProjMatrix;

    if(m_frames)
    {
      m_frameConfig.frameConstantsLast = m_frameConfig.frameConstants;
    }

    int supersample = m_tweak.supersample;


    uint32_t renderWidth  = m_resources.m_frameBuffer.renderSize.width;
    uint32_t renderHeight = m_resources.m_frameBuffer.renderSize.height;

    uint32_t targetWidth  = m_resources.m_frameBuffer.targetSize.width;
    uint32_t targetHeight = m_resources.m_frameBuffer.targetSize.height;

    frameConstants.facetShading = m_tweak.facetShading ? 1 : 0;
    frameConstants.visualize    = m_frameConfig.visualize;
    frameConstants.frame        = m_frames;


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

    frameConstants.nearPlane = m_info.cameraManipulator->getClipPlanes().x;
    frameConstants.farPlane  = m_info.cameraManipulator->getClipPlanes().y;

    frameConstants.wUpDir    = m_info.cameraManipulator->getUp();
    frameConstants.fov = glm::radians(m_info.cameraManipulator->getFov());


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


    frameConstants.viewPixelSize = dim * (glm::vec2(float(renderWidth), float(renderHeight)) * 0.5f) * frameConstants.farPlane;


    frameConstants.viewClipSize = dim * frameConstants.farPlane;

    frameConstants.viewPos = frameConstants.viewMatrixI[3];
    frameConstants.viewDir = -viewI[2];

    frameConstants.viewPlane   = frameConstants.viewDir;
    frameConstants.viewPlane.w = -glm::dot(glm::vec3(frameConstants.viewPos), glm::vec3(frameConstants.viewDir));

    frameConstants.wLightPos = frameConstants.viewMatrixI[3];

    {


      m_resources.m_hizUpdate[0].farInfo.getShaderFactors((float*)&frameConstants.hizSizeFactors);

      frameConstants.hizSizeMax = m_resources.m_hizUpdate[0].farInfo.getSizeMax();


    }


    if(!m_frames)
    {

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


  VkSemaphoreSubmitInfo semSubmit = m_resources.m_queueStates.primary.advanceSignalSubmit(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);

  m_app->addSignalSemaphore(semSubmit);

  while(!m_resources.m_queueStates.primary.m_pendingWaits.empty())
  {
    m_app->addWaitSemaphore(m_resources.m_queueStates.primary.m_pendingWaits.back());

    m_resources.m_queueStates.primary.m_pendingWaits.pop_back();
  }

  m_lastTime = time;
  m_frames++;
}

}
