//==============================================================================
// 文件：src/app/lodclusters.hpp
// 模块定位：顶层应用元素声明，集中描述场景、渲染器、流式加载、帧配置、用户界面和相机状态的所有权关系。
// 数据流：上层 nvapp 通过生命周期回调驱动该类型；该类型再把状态变更传播到 Scene、RenderScene、Renderer 和 Resources。
// 方法说明：这里是应用状态机的接口边界。配置结构、缓存路径、渲染资源和 UI 状态被集中持有，以便每帧对变更进行一致性合并。
// 正确性约束：成员的 last/edit/current 三类状态用于检测 UI 或命令行引起的变化；析构时 性能分析器 时间线 必须仍可由 管理器 释放。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#pragma once


// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <backends/imgui_impl_vulkan.h>
#include <nvapp/application.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/parameter_sequencer.hpp>
#include <nvvk/context.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvgui/enum_registry.hpp>
#include "renderer.hpp"


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace lodclusters {


// 类型：LodClusters。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class LodClusters : public nvapp::IAppElement
{
public:


  // 枚举：RendererType。集中定义本模块可选模式或状态值，避免调用点使用裸整数。
  // 设计意图：把实验开关、渲染模式或阶段编号显式命名，使配置文件、UI 和代码路径可以互相对应。
  // 使用约束：新增枚举值时需要同步 UI 文本、参数解析和相关 switch 分支。
  enum RendererType
  {
    RENDERER_RASTER_CLUSTERS_LOD,
  };


  // 枚举：ClusterConfig。集中定义本模块可选模式或状态值，避免调用点使用裸整数。
  // 设计意图：把实验开关、渲染模式或阶段编号显式命名，使配置文件、UI 和代码路径可以互相对应。
  // 使用约束：新增枚举值时需要同步 UI 文本、参数解析和相关 switch 分支。
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


  // 结构：ClusterInfo。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct ClusterInfo
  {
    uint32_t      tris;
    uint32_t      verts;
    ClusterConfig cfg;
  };

  static const ClusterInfo s_clusterInfos[NUM_CLUSTER_CONFIGS];


  // 枚举：GuiEnums。集中定义本模块可选模式或状态值，避免调用点使用裸整数。
  // 设计意图：把实验开关、渲染模式或阶段编号显式命名，使配置文件、UI 和代码路径可以互相对应。
  // 使用约束：新增枚举值时需要同步 UI 文本、参数解析和相关 switch 分支。
  enum GuiEnums
  {
    GUI_RENDERER,
    GUI_BUILDMODE,
    GUI_SUPERSAMPLE,
    GUI_MESHLET,
    GUI_VISUALIZE,
  };


  // 枚举：ScreenshotMode。集中定义本模块可选模式或状态值，避免调用点使用裸整数。
  // 设计意图：把实验开关、渲染模式或阶段编号显式命名，使配置文件、UI 和代码路径可以互相对应。
  // 使用约束：新增枚举值时需要同步 UI 文本、参数解析和相关 switch 分支。
  enum ScreenshotMode
  {
    SCREENSHOT_OFF,
    SCREENSHOT_WINDOW,
    SCREENSHOT_VIEWPORT
  };


  // 结构：Tweak。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
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


  // 结构：ViewPoint。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct ViewPoint
  {
    std::string name;
    glm::mat4   mat;
    float       sceneScale;
    float       fov;
  };


  // 结构：TargetImage。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct TargetImage
  {
    VkImage     image;
    VkImageView view;
    VkFormat    format;
  };


  // 结构：Info。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct Info
  {
    nvutils::ProfilerManager*                   profilerManager{};
    nvutils::ParameterRegistry*                 parameterRegistry{};
    nvutils::ParameterParser*                   parameterParser{};
    std::shared_ptr<nvutils::CameraManipulator> cameraManipulator;
  };


  LodClusters(const Info& info);

  ~LodClusters() override { m_info.profilerManager->destroyTimeline(m_profilerTimeline); }


  // 函数：onAttach。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void onAttach(nvapp::Application* app) override;


  // 函数：onDetach。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void onDetach() override;


  // 函数：onUIMenu。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void onUIMenu() override;


  // 函数：onUIRender。录制或执行渲染相关工作，把准备好的数据提交到当前渲染阶段。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：渲染函数通常处于帧级关键路径，必须尊重前序计算阶段写出的计数、地址和同步屏障。
  void onUIRender() override;


  // 函数：onPreRender。录制或执行渲染相关工作，把准备好的数据提交到当前渲染阶段。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：渲染函数通常处于帧级关键路径，必须尊重前序计算阶段写出的计数、地址和同步屏障。
  void onPreRender() override;


  // 函数：onRender。录制或执行渲染相关工作，把准备好的数据提交到当前渲染阶段。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：渲染函数通常处于帧级关键路径，必须尊重前序计算阶段写出的计数、地址和同步屏障。
  void onRender(VkCommandBuffer cmd) override;


  // 函数：onResize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override;


  // 函数：onFileDrop。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void onFileDrop(const std::filesystem::path& filename) override;
  void setSupportsBarycentrics(bool supported) { m_resources.m_supportsBarycentrics = supported; }
  void setSupportsMeshShaderNV(bool supported) { m_resources.m_supportsMeshShaderNV = supported; }
  void setSupportsSmBuiltinsNV(bool supported) { m_resources.m_supportsSmBuiltinsNV = supported; }
  bool getShowDebugUI() const { return m_showDebugUI; }

  bool isProcessingOnly() const { return !m_sceneFilePathDropNew.empty() && m_sceneLoaderConfig.processingOnly; }


  // 函数：doProcessingOnly。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void doProcessingOnly();


  // 函数：parameterSequenceCallback。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void parameterSequenceCallback(const nvutils::ParameterSequencer::State& state);

private:
  VkExtent2D                 m_windowSize;
  Info                       m_info;
  nvutils::ProfilerTimeline* m_profilerTimeline{};
  nvvk::ProfilerGpuTimer     m_profilerGpuTimer{};
  nvapp::Application*        m_app{};


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


  // 结构：SwRasterFeedbackState。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
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


  // 函数：initScene。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void initScene(std::filesystem::path filePath, std::string cacheSuffix, bool configChange);


  // 函数：saveCacheFile。把当前状态写入缓存、缓冲、文件或着色器可消费的数据布局。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：写入路径应明确字节对齐、所有权和可见性，避免后续读取端解释错误。
  void saveCacheFile();


  // 函数：deinitScene。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinitScene();


  // 函数：postInitNewScene。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void postInitNewScene();


  // 函数：setSceneCamera。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void setSceneCamera(const std::filesystem::path& filePath);


  // 函数：findSceneClusterConfig。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  ClusterConfig findSceneClusterConfig(const SceneConfig& sceneConfig);


  // 函数：setFromClusterConfig。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void          setFromClusterConfig(SceneConfig& sceneConfig, ClusterConfig clusterConfig);


  // 函数：updatedSceneGrid。根据最新状态刷新缓存数据、GPU 地址、描述符或统计信息。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：更新函数负责把“旧状态”推进到“当前状态”，因此要避免部分更新造成 CPU/GPU 视图不一致。
  void          updatedSceneGrid();


  // 函数：initRenderScene。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void initRenderScene();


  // 函数：deinitRenderScene。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinitRenderScene();


  // 函数：initRenderer。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void initRenderer(RendererType rtype);


  // 函数：deinitRenderer。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinitRenderer();


  // 函数：updateImguiImage。根据最新状态刷新缓存数据、GPU 地址、描述符或统计信息。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：更新函数负责把“旧状态”推进到“当前状态”，因此要避免部分更新造成 CPU/GPU 视图不一致。
  void updateImguiImage();


  // 函数：handleChanges。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void handleChanges();


  // 函数：applyCameraString。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void applyCameraString();


  // 函数：resetSwRasterFeedback。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void resetSwRasterFeedback();


  // 函数：updateSwRasterFeedback。根据最新状态刷新缓存数据、GPU 地址、描述符或统计信息。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：更新函数负责把“旧状态”推进到“当前状态”，因此要避免部分更新造成 CPU/GPU 视图不一致。
  void updateSwRasterFeedback();


  // 函数：decodePickingDepth。在紧凑编码和逻辑结构之间转换，减少带宽或便于着色器访问。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：编码位宽、符号位和特殊值必须与写入端/读取端完全一致，否则会产生难以定位的跨阶段错误。
  float decodePickingDepth(const shaderio::Readback& readback);


  // 函数：isPickingValid。返回条件判断结果，用于调用方选择后续分支或验证输入状态。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：谓词函数应保持无副作用或低副作用，使调用方可以安全地把它用于断言、过滤和早退。
  bool  isPickingValid(const shaderio::Readback& readback);


  // 函数：viewportUI。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void viewportUI(ImVec2 corner);


  // 函数：loadingUI。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
  void loadingUI();

  template <typename T>


  // 函数：sceneChanged。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  bool sceneChanged(const T& val) const
  {

    size_t offset = size_t(&val) - size_t(&m_sceneConfig);
    assert(offset < sizeof(m_sceneConfig));
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_sceneConfigLast) + offset, sizeof(T)) != 0;
  }

  template <typename T>


  // 函数：tweakChanged。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  bool tweakChanged(const T& val) const
  {

    size_t offset = size_t(&val) - size_t(&m_tweak);
    assert(offset < sizeof(m_tweak));
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_tweakLast) + offset, sizeof(T)) != 0;
  }

  template <typename T>


  // 函数：rendererCfgChanged。录制或执行渲染相关工作，把准备好的数据提交到当前渲染阶段。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：渲染函数通常处于帧级关键路径，必须尊重前序计算阶段写出的计数、地址和同步屏障。
  bool rendererCfgChanged(const T& val) const
  {

    size_t offset = size_t(&val) - size_t(&m_rendererConfig);
    assert(offset < sizeof(m_rendererConfig));
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_rendererConfigLast) + offset, sizeof(T)) != 0;
  }

  template <typename T>


  // 函数：streamingCfgChanged。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  bool streamingCfgChanged(const T& val) const
  {

    size_t offset = size_t(&val) - size_t(&m_streamingConfig);
    assert(offset < sizeof(m_rendererConfig));
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_streamingConfigLast) + offset, sizeof(T)) != 0;
  }
};
}
