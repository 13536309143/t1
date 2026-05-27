/*
 * lodclusters.hpp
 * 
 * LOD Clusters主类定义，负责管理场景、渲染器和用户界面
 * 
 * 主要功能：
 * - 场景加载和管理
 * - 渲染器初始化和配置
 * - 用户界面管理
 * - 性能分析和统计
 */

#pragma once
#include <backends/imgui_impl_vulkan.h>
#include <nvapp/application.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/parameter_sequencer.hpp>
#include <nvvk/context.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvgui/enum_registry.hpp>
#include "renderer.hpp"

namespace lodclusters {

class LodClusters : public nvapp::IAppElement
{
public:
  enum RendererType
  {
    RENDERER_RASTER_CLUSTERS_LOD,
  };

  enum ClusterConfig
  {
    CLUSTER_32T_32V,
    CLUSTER_32T_64V,
    CLUSTER_32T_96V,
    CLUSTER_32T_128V,
    CLUSTER_32T_160V,
    CLUSTER_32T_192V,
    CLUSTER_32T_224V,
    CLUSTER_32T_256V,
    CLUSTER_64T_32V,
    CLUSTER_64T_64V,
    CLUSTER_64T_96V,
    CLUSTER_64T_128V,
    CLUSTER_64T_160V,
    CLUSTER_64T_192V,
    CLUSTER_64T_224V,
    CLUSTER_64T_256V,
    CLUSTER_96T_32V,
    CLUSTER_96T_64V,
    CLUSTER_96T_96V,
    CLUSTER_96T_128V,
    CLUSTER_96T_160V,
    CLUSTER_96T_192V,
    CLUSTER_96T_224V,
    CLUSTER_96T_256V,
    CLUSTER_128T_32V,
    CLUSTER_128T_64V,
    CLUSTER_128T_96V,
    CLUSTER_128T_128V,
    CLUSTER_128T_160V,
    CLUSTER_128T_192V,
    CLUSTER_128T_224V,
    CLUSTER_128T_256V,
    CLUSTER_160T_32V,
    CLUSTER_160T_64V,
    CLUSTER_160T_96V,
    CLUSTER_160T_128V,
    CLUSTER_160T_160V,
    CLUSTER_160T_192V,
    CLUSTER_160T_224V,
    CLUSTER_160T_256V,
    CLUSTER_192T_32V,
    CLUSTER_192T_64V,
    CLUSTER_192T_96V,
    CLUSTER_192T_128V,
    CLUSTER_192T_160V,
    CLUSTER_192T_192V,
    CLUSTER_192T_224V,
    CLUSTER_192T_256V,
    CLUSTER_224T_32V,
    CLUSTER_224T_64V,
    CLUSTER_224T_96V,
    CLUSTER_224T_128V,
    CLUSTER_224T_160V,
    CLUSTER_224T_192V,
    CLUSTER_224T_224V,
    CLUSTER_224T_256V,
    CLUSTER_256T_32V,
    CLUSTER_256T_64V,
    CLUSTER_256T_96V,
    CLUSTER_256T_128V,
    CLUSTER_256T_160V,
    CLUSTER_256T_192V,
    CLUSTER_256T_224V,
    CLUSTER_256T_256V,
    NUM_CLUSTER_CONFIGS,
  };

  struct ClusterInfo
  {
    uint32_t      tris;
    uint32_t      verts;
    ClusterConfig cfg;
  };

  static const ClusterInfo s_clusterInfos[NUM_CLUSTER_CONFIGS];

  enum GuiEnums
  {
    GUI_RENDERER,
    GUI_BUILDMODE,
    GUI_SUPERSAMPLE,
    GUI_MESHLET,
    GUI_VISUALIZE,
  };

  enum ScreenshotMode
  {
    SCREENSHOT_OFF,
    SCREENSHOT_WINDOW,
    SCREENSHOT_VIEWPORT
  };

  struct Tweak
  {
    ClusterConfig clusterConfig = CLUSTER_128T_128V;

    RendererType renderer    = RENDERER_RASTER_CLUSTERS_LOD;
    int          supersample = 2;

    bool facetShading = true;
    bool useStreaming = false;

    bool autoResetTimers = false;
    bool autoSharing     = true;

    float clickSpeedScale = 0.33f;
  };
  struct ViewPoint
  {
    std::string name;
    glm::mat4   mat;
    float       sceneScale;
    float       fov;
  };

  struct TargetImage
  {
    VkImage     image;
    VkImageView view;
    VkFormat    format;
  };

  struct Info
  {
    nvutils::ProfilerManager*                   profilerManager{};
    nvutils::ParameterRegistry*                 parameterRegistry{};
    nvutils::ParameterParser*                   parameterParser{};
    std::shared_ptr<nvutils::CameraManipulator> cameraManipulator;
  };

  LodClusters(const Info& info);

  ~LodClusters() override { m_info.profilerManager->destroyTimeline(m_profilerTimeline); }

  void onAttach(nvapp::Application* app) override;
  void onDetach() override;
  void onUIMenu() override;
  void onUIRender() override;
  void onPreRender() override;
  void onRender(VkCommandBuffer cmd) override;
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override;
  void onFileDrop(const std::filesystem::path& filename) override;
  void setSupportsBarycentrics(bool supported) { m_resources.m_supportsBarycentrics = supported; }
  void setSupportsMeshShaderNV(bool supported) { m_resources.m_supportsMeshShaderNV = supported; }
  void setSupportsSmBuiltinsNV(bool supported) { m_resources.m_supportsSmBuiltinsNV = supported; }
  bool getShowDebugUI() const { return m_showDebugUI; }

  bool isProcessingOnly() const { return !m_sceneFilePathDropNew.empty() && m_sceneLoaderConfig.processingOnly; }
  void doProcessingOnly();

  void parameterSequenceCallback(const nvutils::ParameterSequencer::State& state);

private:
  VkExtent2D                 m_windowSize;
  Info                       m_info;
  nvutils::ProfilerTimeline* m_profilerTimeline{};
  nvvk::ProfilerGpuTimer     m_profilerGpuTimer{};
  nvapp::Application*        m_app{};

  //////////////////////////////////////////////////////////////////////////

  // key components

  Resources                 m_resources;
  FrameConfig               m_frameConfig;
  double                    m_lastTime = 0;
  VkDescriptorSet           m_imguiTexture{};
  VkSampler                 m_imguiSampler{};
  nvgui::EnumRegistry       m_ui;
  nvutils::PerformanceTimer m_clock;

  bool m_reloadShaders = false;
#ifndef NDEBUG
  bool m_showDebugUI = true;
#else
  bool m_showDebugUI = false;
#endif
  int            m_frames                 = 0;
  double         m_animTime               = 0;
  ScreenshotMode m_sequenceScreenshotMode = SCREENSHOT_OFF;

  Tweak m_tweak;
  Tweak m_tweakLast;

  uint32_t m_lastAmbientOcclusionSamples = 0;

  std::unique_ptr<Scene> m_scene;
  std::filesystem::path  m_sceneFilePath;
  std::filesystem::path  m_sceneFilePathDefault;
  std::filesystem::path  m_sceneFilePathDropLast;
  std::filesystem::path  m_sceneFilePathDropNew;
  std::string            m_sceneCacheSuffix = ".zippp";
  SceneLoaderConfig      m_sceneLoaderConfig;
  SceneConfig            m_sceneConfig;
  SceneConfig            m_sceneConfigLast;
  SceneConfig            m_sceneConfigEdit;
  glm::vec3              m_sceneUpVector = glm::vec3(0, 1, 0);
  SceneGridConfig        m_sceneGridConfig;
  SceneGridConfig        m_sceneGridConfigLast;
  std::atomic_bool       m_sceneLoading        = false;
  std::atomic_uint32_t   m_sceneProgress       = 0;
  bool                   m_sceneLoadFromConfig = false;

  std::string m_cameraString;
  std::string m_cameraStringLast;
  std::string m_cameraStringCommandLine;
  float       m_cameraSpeed = 0;
  //std::filesystem::path  m_cameraFilePath;

  std::unique_ptr<RenderScene> m_renderScene;
  bool                         m_renderSceneCanPreload = false;

  StreamingConfig m_streamingConfig;
  StreamingConfig m_streamingConfigLast;

  std::unique_ptr<Renderer> m_renderer;
  uint64_t                  m_rendererFboChangeID{};
  RendererConfig            m_rendererConfig;
  RendererConfig            m_rendererConfigLast;

  std::vector<uint32_t> m_streamClasHistogram;
  std::vector<uint32_t> m_streamGeometryHistogram;
  uint32_t              m_streamGeometryHistogramMax;
  int32_t               m_streamHistogramOffset = 0;

  uint32_t m_equalFrames = 0;

  struct SwRasterFeedbackState
  {
    bool  initialized = false;
    float lastBaseExtent = 0.0f;
    float lastBaseDensity = 0.0f;
    float effectiveExtent = 0.0f;
    float effectiveDensity = 0.0f;
    float emaSwClusterShare = 0.0f;
    float emaSwTriangleShare = 0.0f;
    float emaSwTrianglesPerCluster = 0.0f;
  } m_swRasterFeedback;
  // // 拾取相关
  // struct PickedInfo
  // {
  //   bool      valid = false;
  //   uint32_t  instanceId = 0;
  //   std::string name;
  //   uint32_t  vertexCount = 0;
  //   uint32_t  triangleCount = 0;
  //   uint32_t  clusterCount = 0;
  // } m_pickedInfo;
  // use by-value copies for flexibility
  void initScene(std::filesystem::path filePath, std::string cacheSuffix, bool configChange);

  void setSceneCamera(const std::filesystem::path& filePath);
  void saveCacheFile();
  void deinitScene();
  void postInitNewScene();

  void initRenderScene();
  void deinitRenderScene();

  void initRenderer(RendererType rtype);
  void deinitRenderer();

  void updateImguiImage();

  ClusterConfig findSceneClusterConfig(const SceneConfig& sceneConfig);
  void          setFromClusterConfig(SceneConfig& sceneConfig, ClusterConfig clusterConfig);
  void          updatedSceneGrid();

  void handleChanges();
  void applyCameraString();
  void resetSwRasterFeedback();
  void updateSwRasterFeedback();

  float decodePickingDepth(const shaderio::Readback& readback);
  bool  isPickingValid(const shaderio::Readback& readback);

  void viewportUI(ImVec2 corner);

  void loadingUI();

  template <typename T>
  bool sceneChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_sceneConfig);
    assert(offset < sizeof(m_sceneConfig));
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_sceneConfigLast) + offset, sizeof(T)) != 0;
  }

  template <typename T>
  bool tweakChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_tweak);
    assert(offset < sizeof(m_tweak));
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_tweakLast) + offset, sizeof(T)) != 0;
  }

  template <typename T>
  bool rendererCfgChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_rendererConfig);
    assert(offset < sizeof(m_rendererConfig));
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_rendererConfigLast) + offset, sizeof(T)) != 0;
  }

  template <typename T>
  bool streamingCfgChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_streamingConfig);
    assert(offset < sizeof(m_rendererConfig));
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_streamingConfigLast) + offset, sizeof(T)) != 0;
  }
};
}  // namespace lodclusters
