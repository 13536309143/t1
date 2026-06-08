//==============================================================================
// 文件：src/app/main.cpp
// 模块定位：应用程序入口，负责把窗口系统、参数系统、Vulkan 上下文、性能分析器和 LodClusters 应用元素连接成完整运行体。
// 数据流：输入来自命令行、配置文件和运行环境设备能力；输出是已经初始化的 nvapp 应用、逻辑设备、队列、界面布局和主循环。
// 方法说明：该文件体现“组合根”模式：算法模块不在入口处展开，而是通过明确的初始化顺序建立依赖图，避免跨层资源生命周期失配。
// 正确性约束：Vulkan 特性链必须在创建设备前完成；processing-only 模式必须在无窗口路径下提前结束；清理顺序应与初始化顺序相反。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#ifndef NDEBUG


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    fprintf(stderr, (format), __VA_ARGS__);                                                                            \
    fprintf(stderr, "\n");                                                                                             \
  } while(false)
#endif


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define VMA_IMPLEMENTATION

#if __INTELLISENSE__
#undef VK_NO_PROTOTYPES
#endif


// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <volk.h>
#include <imgui/imgui.h>
#include <nvvk/validation_settings.hpp>
#include <nvapp/elem_logger.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvapp/elem_sequencer.hpp>
#include <nvutils/parameter_parser.hpp>

#include "lodclusters.hpp"

using namespace lodclusters;


// 函数：main。作为程序入口，串联初始化、运行和清理流程。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该入口位于控制流根部，调用顺序决定后续资源生命周期和数据依赖。
int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;
  appInfo.name    = TARGET_NAME;
  appInfo.useMenu = true;
  appInfo.vSync = false;
  VkPhysicalDeviceShaderSMBuiltinsFeaturesNV smNV = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES_NV};
  VkPhysicalDeviceMeshShaderFeaturesNV       meshNV  = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV};
  VkPhysicalDeviceMeshShaderFeaturesEXT      meshEXT = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accKHR = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR rayQueryKHR = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceClusterAccelerationStructureFeaturesNV clustersNV = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV};
  VkPhysicalDeviceShaderClockFeaturesKHR clockKHR = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
  VkPhysicalDeviceFragmentShadingRateFeaturesKHR shadingRateFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR};
  VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentricFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
  VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT shaderImageAtomic64Features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT};

  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {{VK_KHR_SWAPCHAIN_EXTENSION_NAME}},
      .queues             = {VK_QUEUE_GRAPHICS_BIT, VK_QUEUE_TRANSFER_BIT},
  };

  vkSetup.deviceExtensions.push_back({VK_EXT_MESH_SHADER_EXTENSION_NAME, &meshEXT});
  vkSetup.deviceExtensions.push_back({VK_KHR_SHADER_CLOCK_EXTENSION_NAME, &clockKHR});
  vkSetup.deviceExtensions.push_back({VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, &atomicFloatFeatures});


  vkSetup.deviceExtensions.push_back({VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, &shadingRateFeatures});

  vkSetup.deviceExtensions.push_back({VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME, &shaderImageAtomic64Features});

#if 1


  vkSetup.deviceExtensions.push_back({VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME, &clustersNV, false, 2});

  vkSetup.deviceExtensions.push_back({VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME, &smNV, false});
  vkSetup.deviceExtensions.push_back({VK_NV_MESH_SHADER_EXTENSION_NAME, &meshNV, false});

  vkSetup.deviceExtensions.push_back({VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, &barycentricFeatures, false});
#endif


  nvutils::ProfilerManager                    profilerManager;
  std::shared_ptr<nvutils::CameraManipulator> cameraManipulator = std::make_shared<nvutils::CameraManipulator>();

  nvutils::ParameterRegistry            parameterRegistry;
  nvutils::ParameterParser              parameterParser;
  nvutils::ParameterSequencer::InitInfo sequencerInfo{
                                                      .parameterParser   = &parameterParser,
                                                      .parameterRegistry = &parameterRegistry,

                                                      .profilerManager = &profilerManager};

  nvvk::ValidationSettings::LayerPresets validationPreset = nvvk::ValidationSettings::LayerPresets::eStandard;

  parameterRegistry.add({"validation"}, &vkSetup.enableValidationLayers);
  parameterRegistry.add({"validationpreset"}, (int*)&validationPreset);
  parameterRegistry.add({"vsync"}, &appInfo.vSync);
  parameterRegistry.add({"device", "force a vulkan device via index into the device list"}, &vkSetup.forceGPU);
  parameterRegistry.add({"headless"}, &appInfo.headless, true);
  parameterRegistry.add({"headlessframes"}, &appInfo.headlessFrameCount);

  LodClusters::Info sampleInfo;
  sampleInfo.cameraManipulator               = cameraManipulator;
  sampleInfo.profilerManager                 = &profilerManager;
  sampleInfo.parameterRegistry               = &parameterRegistry;
  sampleInfo.parameterParser                 = &parameterParser;
  std::shared_ptr<LodClusters> sampleElement = std::make_shared<LodClusters>(sampleInfo);


  sequencerInfo.registerScriptParameters(parameterRegistry, parameterParser);


  sequencerInfo.postCallbacks.emplace_back(
      [&](const nvutils::ParameterSequencer::State& state) { sampleElement->parameterSequenceCallback(state); });


  parameterParser.add(parameterRegistry);

  parameterParser.setVerbose(true);

  parameterParser.parse(argc, argv);


  auto elemSequencer = std::make_shared<nvapp::ElementSequencer>(sequencerInfo);


  if(sampleElement->isProcessingOnly())
  {

    sampleElement->doProcessingOnly();
    return 0;
  }

  nvvk::ValidationSettings validationSettings;
  if(vkSetup.enableValidationLayers)
  {

    validationSettings.setPreset(validationPreset);
    validationSettings.duplicate_message_limit = 3;
    validationSettings.message_id_filter = {"VUID-RuntimeSpirv-storageInputOutput16-06334", "VUID-VkShaderModuleCreateInfo-pCode-08740"};


    vkSetup.instanceCreateInfoExt = validationSettings.buildPNextChain();
  }


  nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
  nvvk::Context vkContext;


  NVVK_CHECK(volkInitialize());

  {


    // 函数：st。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    nvutils::ScopedTimer st("Creating Vulkan Context");
    VkResult result{};
    vkContext.contextInfo = vkSetup;

    result = vkContext.createInstance();

    result = vkContext.selectPhysicalDevice();

    result = vkContext.createDevice();

    NVVK_CHECK(result);
    nvvk::DebugUtil::getInstance().init(vkContext.getDevice());
    if(vkContext.contextInfo.verbose)
    {
      NVVK_CHECK(nvvk::Context::printVulkanVersion());
      NVVK_CHECK(nvvk::Context::printInstanceLayers());
      NVVK_CHECK(nvvk::Context::printInstanceExtensions(vkContext.contextInfo.instanceExtensions));
      NVVK_CHECK(nvvk::Context::printDeviceExtensions(vkContext.getPhysicalDevice(), vkContext.contextInfo.deviceExtensions));
    }
    {
      NVVK_CHECK(nvvk::Context::printGpus(vkContext.getInstance(), vkContext.getPhysicalDevice()));

      LOGI("_________________________________________________\n");
    }
  }

  sampleElement->setSupportsBarycentrics(vkContext.hasExtensionEnabled(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME));
  sampleElement->setSupportsMeshShaderNV(vkContext.hasExtensionEnabled(VK_NV_MESH_SHADER_EXTENSION_NAME));
  sampleElement->setSupportsSmBuiltinsNV(vkContext.hasExtensionEnabled(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME));

  appInfo.instance       = vkContext.getInstance();

  appInfo.device         = vkContext.getDevice();

  appInfo.physicalDevice = vkContext.getPhysicalDevice();

  appInfo.queues         = vkContext.getQueueInfos();


  bool hasDebugUI = sampleElement->getShowDebugUI();


  appInfo.dockSetup = [&hasDebugUI](ImGuiID viewportID) {
    if(hasDebugUI)
    {


      ImGuiID debugID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Left, 0.15F, nullptr, &viewportID);

      ImGui::DockBuilderDockWindow("Debug", debugID);
    }


    ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.25F, nullptr, &viewportID);

    ImGui::DockBuilderDockWindow("Settings", settingID);

    ImGui::DockBuilderDockWindow("Misc Settings", settingID);


    ImGuiID loggerID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Down, 0.35F, nullptr, &viewportID);

    ImGui::DockBuilderDockWindow("Log", loggerID);

    ImGuiID profilerID = ImGui::DockBuilderSplitNode(loggerID, ImGuiDir_Right, 0.75F, nullptr, &loggerID);

    ImGui::DockBuilderDockWindow("Profiler", profilerID);

    ImGuiID streamingID = ImGui::DockBuilderSplitNode(profilerID, ImGuiDir_Right, 0.66F, nullptr, &profilerID);

    ImGui::DockBuilderDockWindow("Streaming memory", streamingID);

    ImGuiID statisticsID = ImGui::DockBuilderSplitNode(streamingID, ImGuiDir_Right, 0.5F, nullptr, &streamingID);

    ImGui::DockBuilderDockWindow("Statistics", statisticsID);
  };


  nvapp::Application app;

  app.init(appInfo);

  auto                  logger      = std::make_shared<nvapp::ElementLogger>();

  nvapp::ElementLogger* loggerDeref = logger.get();
  nvutils::Logger::getInstance().setLogCallback([&](nvutils::Logger::LogLevel logLevel, const std::string& text) {
    loggerDeref->addLog(logLevel, "%s", text.c_str());
  });

  auto profilerUiSettings          = std::make_shared<nvapp::ElementProfiler::ViewSettings>();
  profilerUiSettings->table.levels = 1u;

  app.addElement(elemSequencer);
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>());

  app.addElement(sampleElement);

  app.addElement(logger);
  app.addElement(std::make_shared<nvapp::ElementCamera>(cameraManipulator));
  app.addElement(std::make_shared<nvapp::ElementProfiler>(&profilerManager, profilerUiSettings));

  app.run();

  nvutils::Logger::getInstance().setLogCallback(nullptr);


  app.deinit();

  vkContext.deinit();

  return 0;
}
