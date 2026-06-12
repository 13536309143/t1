//==============================================================================
// 文件：src/app/lodclusters_lifecycle.cpp
// 模块定位：应用挂载、释放、窗口尺寸变化、渲染器重建和 ImGui 纹理桥接的生命周期实现。
// 数据流：输入是 nvapp 回调和窗口大小；输出是已匹配当前 帧缓冲 的资源、描述符和 renderer 实例。
// 方法说明：生命周期函数把 Vulkan 对象的创建/销毁拓扑固定下来，避免图像、描述符和 管线 之间出现悬垂引用。
// 正确性约束：释放前需要等待 GPU 空闲；帧缓冲 变化必须刷新 ImGui 图像和 renderer 描述符；renderer 销毁早于 Resources 销毁。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <volk.h>
#include <fmt/format.h>
#include <nvutils/file_operations.hpp>
#include "lodclusters.hpp"


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace lodclusters {


// 函数：LodClusters::onResize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
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


// 函数：LodClusters::updateImguiImage。根据最新状态刷新缓存数据、GPU 地址、描述符或统计信息。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：更新函数负责把“旧状态”推进到“当前状态”，因此要避免部分更新造成 CPU/GPU 视图不一致。
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


// 函数：LodClusters::deinitRenderer。释放或回收前面初始化的资源，保持生命周期成对管理。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
void LodClusters::deinitRenderer()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));

  if(m_renderer)
  {

    m_renderer->deinit(m_resources);
    m_renderer = nullptr;
  }
}


// 函数：LodClusters::initRenderer。初始化本模块所需状态、资源或 GPU 侧绑定。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
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


// 函数：LodClusters::onAttach。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
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

    m_ui.enumAdd(GUI_SUPERSAMPLE, 1024, "1024 sq");

    m_ui.enumAdd(GUI_SUPERSAMPLE, 2048, "2048 sq");

    m_ui.enumAdd(GUI_SUPERSAMPLE, 4096, "4096 sq");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_MATERIAL, "material");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_GREY, "grey");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_VIS_BUFFER, "visibility buffer");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_CLUSTER, "clusters");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_GROUP, "cluster groups");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_LOD, "lod levels");

    m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_TRIANGLE, "triangles");

  }


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


  if(m_sceneFilePathDropNew.empty())
  {

    const std::filesystem::path              exeDirectoryPath   = nvutils::getExecutablePath().parent_path();
    const std::vector<std::filesystem::path> defaultSearchPaths = {

        std::filesystem::absolute(exeDirectoryPath / TARGET_EXE_TO_DOWNLOAD_DIRECTORY),

        std::filesystem::absolute(exeDirectoryPath / "resources"),
    };


    m_sceneFilePathDefault = m_sceneFilePathDropNew = nvutils::findFile("bunny_v2/bunny.gltf", defaultSearchPaths);


    m_sceneGridConfig.uniqueGeometriesForCopies = false;

    if(m_sceneGridConfig.numCopies == 1)
    {
      if(m_resources.getDeviceLocalHeapSize() >= 8ull * 1024 * 1024 * 1024)
      {
        m_sceneGridConfig.numCopies = 1;
      }
      else
      {
        m_sceneGridConfig.numCopies =1;
      }
    }
  }

  m_cameraStringCommandLine = m_cameraString;

  std::filesystem::path newFileDrop = m_sceneFilePathDropNew;

  onFileDrop(newFileDrop);

  m_tweakLast          = m_tweak;
  m_sceneConfigLast    = m_sceneConfig;
  m_sceneConfigEdit    = m_sceneConfig;
  m_rendererConfigLast = m_rendererConfig;
}


// 函数：LodClusters::onDetach。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
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


// 函数：LodClusters::parameterSequenceCallback。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
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

}
