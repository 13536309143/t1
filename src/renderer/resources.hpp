//==============================================================================
// 文件：src/renderer/resources.hpp
// 模块定位：Vulkan 资源管理声明，封装设备、队列、帧缓冲、上传器、着色器编译、内存统计和通用命令辅助。
// 数据流：上层 renderer 请求资源创建、上传和同步；Resources 负责转换为 Vulkan 句柄和命令缓冲操作。
// 方法说明：该类是渲染系统的资源中枢，将 Vulkan 的显式生命周期、同步和内存管理封装成项目可复用接口。
// 正确性约束：所有 GPU 对象必须由同一 device 释放；临时命令提交要推进时间线；帧缓冲 尺寸变化必须重建依赖图像视图。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#pragma once


// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <span>
#if __INTELLISENSE__
#undef VK_NO_PROTOTYPES
#endif
#include <glm/glm.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/alignment.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/physical_device.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvkglsl/glsl.hpp>
#include <vk_radix_sort.h>

#if VK_HEADER_VERSION < 309
#error Update Vulkan SDK >= 1.4.309.0
#endif
#include "hiz.hpp"
#include "shaderio.h"


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace lodclusters {


// 结构：FrameConfig。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct FrameConfig
{
  VkExtent2D windowSize;

  bool  showInstanceBboxes = false;
  bool  showClusterBboxes  = false;
  bool  freezeCulling      = false;
  bool  freezeLoD          = false;
  float lodPixelError      = 1.0f;

  float culledErrorScale = 2.0f;

  float swRasterThreshold = 8.0f;

  float swRasterTriangleDensityThreshold = 0.5f;
  bool  swRasterFeedbackEnabled = false;
  float swRasterFeedbackTargetTriangleShare = 0.15f;
  float swRasterThresholdEffective = 8.0f;
  float swRasterTriangleDensityThresholdEffective = 0.5f;


  uint32_t streamingAgeThreshold = 16;


  uint32_t traversalPersistentThreads = 2048;

  uint32_t sharingTolerantLevels = 7;
  uint32_t sharingEnabledLevels  = 8;
  bool     sharingPushCulled     = true;

  uint32_t cachingEnabledLevels = 8;
  uint32_t cachingAgeThreshold  = 16;

  uint32_t visualize = VISUALIZE_LOD;

  shaderio::FrameConstants frameConstants;
  shaderio::FrameConstants frameConstantsLast;
  glm::mat4                traversalViewMatrix;
  glm::mat4                cullViewProjMatrix;
  glm::mat4                cullViewProjMatrixLast;
};


// 函数：cmdCopyBuffer。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
inline void cmdCopyBuffer(VkCommandBuffer cmd, const nvvk::Buffer& src, const nvvk::Buffer& dst)
{
  VkBufferCopy cpy = {0, 0, src.bufferSize};

  vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &cpy);
}


// 函数：formatMemorySize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
std::string formatMemorySize(size_t sizeInBytes);


// 函数：logMemoryUsage。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
inline size_t logMemoryUsage(size_t size, const char* memtype, const char* what)
{
  LOGI("%s memory: %s - %s\n", memtype, formatMemorySize(size).c_str(), what);
  return size;
}


// 结构：BufferRanges。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct BufferRanges
{
  VkDeviceSize tempOffset = 0;

  VkDeviceSize beginOffset = 0;
  VkDeviceSize splitOffset = 0;


  // 函数：append。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  VkDeviceSize append(VkDeviceSize size, VkDeviceSize alignment)
  {

    tempOffset = nvutils::align_up(tempOffset, alignment);

    VkDeviceSize offset = tempOffset;
    tempOffset += size;

    return offset;
  }


  // 函数：beginOverlap。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void beginOverlap()
  {
    beginOffset = tempOffset;
    splitOffset = 0;
  }


  // 函数：splitOverlap。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void splitOverlap()
  {

    splitOffset = std::max(splitOffset, tempOffset);
    tempOffset  = beginOffset;
  }
  void endOverlap() { tempOffset = std::max(splitOffset, tempOffset); }

  VkDeviceSize getSize(VkDeviceSize alignment = 4) { return nvutils::align_up(tempOffset, alignment); }
};


// 类型：QueueState。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class QueueState
{
public:
  VkDevice    m_device            = nullptr;
  VkQueue     m_queue             = nullptr;
  uint32_t    m_familyIndex       = 0;
  VkSemaphore m_timelineSemaphore = nullptr;
  uint64_t    m_timelineValue     = 1;

  std::vector<VkSemaphoreSubmitInfo> m_pendingWaits;


  // 函数：init。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void init(VkDevice device, VkQueue queue, uint32_t familyIndex, uint64_t initialValue);


  // 函数：deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinit();


  // 函数：getTimelineValue。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  VkResult getTimelineValue(uint64_t& timelineValue) const
  {
    return vkGetSemaphoreCounterValue(m_device, m_timelineSemaphore, &timelineValue);
  }


  // 函数：getCurrentState。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  nvvk::SemaphoreState getCurrentState() const
  {
    return nvvk::SemaphoreState::makeFixed(m_timelineSemaphore, m_timelineValue);
  }


  VkSemaphoreSubmitInfo getWaitSubmit(VkPipelineStageFlags2 stageMask, uint32_t deviceIndex = 0) const;


  VkSemaphoreSubmitInfo advanceSignalSubmit(VkPipelineStageFlags2 stageMask, uint32_t deviceIndex = 0);
};


// 结构：QueueStateManager。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct QueueStateManager
{
  QueueState primary;
  QueueState transfer;
};


// 类型：Resources。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class Resources
{
public:
  static constexpr VkPipelineStageFlags2 ALL_SHADER_STAGES =
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;


  // 结构：FrameBuffer。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct FrameBuffer
  {
    VkExtent2D renderSize{};
    VkExtent2D targetSize{};
    VkExtent2D windowSize{};


    int supersample = 0;

    bool  useResolved = false;
    float pixelScale  = 1;

    VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat depthStencilFormat;

    VkViewport viewport;
    VkRect2D   scissor;

    nvvk::Image imgColor         = {};
    nvvk::Image imgColorResolved = {};
    nvvk::Image imgDepthStencil  = {};

    VkImageView viewDepth = VK_NULL_HANDLE;
    nvvk::Image imgRasterAtomic = {};


    nvvk::Image imgHizFar[2] = {};

    VkPipelineRenderingCreateInfo pipelineRenderingInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  };


  // 函数：init。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void init(VkDevice device, VkPhysicalDevice physicalDevice, VkInstance instance, const nvvk::QueueInfo& queue, const nvvk::QueueInfo& queueTransfer);


  // 函数：deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinit();


  // 函数：initFramebuffer。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  bool initFramebuffer(const VkExtent2D& windowSize, int supersample);


  // 函数：updateFramebufferRenderSizeDependent。录制或执行渲染相关工作，把准备好的数据提交到当前渲染阶段。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：渲染函数通常处于帧级关键路径，必须尊重前序计算阶段写出的计数、地址和同步屏障。
  void updateFramebufferRenderSizeDependent(VkCommandBuffer cmd);


  // 函数：deinitFramebufferRenderSizeDependent。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinitFramebufferRenderSizeDependent();


  // 函数：deinitFramebuffer。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinitFramebuffer();


  // 函数：getFramebufferWindow2RenderScale。录制或执行渲染相关工作，把准备好的数据提交到当前渲染阶段。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：渲染函数通常处于帧级关键路径，必须尊重前序计算阶段写出的计数、地址和同步屏障。
  glm::vec2 getFramebufferWindow2RenderScale() const;


  // 函数：beginFrame。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void beginFrame(uint32_t cycleIndex);


  // 函数：postProcessFrame。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void postProcessFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);


  // 函数：emptyFrame。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void emptyFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);


  // 函数：endFrame。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void endFrame();


  // 函数：cmdBuildHiz。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
  void cmdBuildHiz(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler, uint32_t idx);


  // 函数：cmdLinearDispatch。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
  void cmdLinearDispatch(VkCommandBuffer cmd, uint32_t count) const
  {
    if(!count)
      return;

    if(!m_use16bitDispatch || count <= 0xFFFF)
    {

      vkCmdDispatch(cmd, count, 1, 1);
    }
    else
    {

      glm::uvec3 grid = shaderio::fit16bitLaunchGrid(count);

      assert(grid.x <= 0xFFFF && grid.y <= 0xFFFF && grid.z <= 0xFFFF);

      vkCmdDispatch(cmd, grid.x, grid.y, grid.z);
    }
  }


  // 函数：getReadbackData。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
  void getReadbackData(shaderio::Readback& readback);


  shaderc::CompileOptions makeCompilerOptions() { return shaderc::CompileOptions(m_glslCompiler.options()); }

  bool compileShader(shaderc::SpvCompilationResult& compiled,
                     VkShaderStageFlagBits          shader,
                     const std::filesystem::path&   filePath,
                     shaderc::CompileOptions*       options = nullptr);


  // 函数：verifyShaders。返回条件判断结果，用于调用方选择后续分支或验证输入状态。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：谓词函数应保持无副作用或低副作用，使调用方可以安全地把它用于断言、过滤和早退。
  bool verifyShaders(size_t numShaders, shaderc::SpvCompilationResult* shaders)
  {
    for(size_t i = 0; i < numShaders; i++)
    {
      if(shaders[i].GetCompilationStatus() != shaderc_compilation_status_null_result_object
         && shaders[i].GetCompilationStatus() != shaderc_compilation_status_success)
        return false;
    }

    return true;
  }
  template <typename T>


  // 函数：verifyShaders。返回条件判断结果，用于调用方选择后续分支或验证输入状态。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：谓词函数应保持无副作用或低副作用，使调用方可以安全地把它用于断言、过滤和早退。
  bool verifyShaders(T& container)
  {
    return verifyShaders(sizeof(T) / sizeof(shaderc::SpvCompilationResult), (shaderc::SpvCompilationResult*)&container);
  }


  // 函数：destroyPipelines。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void destroyPipelines(size_t numPipelines, VkPipeline* pipelines)
  {
    for(size_t i = 0; i < numPipelines; i++)
    {

      vkDestroyPipeline(m_device, pipelines[i], nullptr);
      pipelines[i] = nullptr;
    }
  }
  template <typename T>


  // 函数：destroyPipelines。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void destroyPipelines(T& container)
  {
    destroyPipelines(sizeof(T) / sizeof(VkPipeline), (VkPipeline*)&container);
  }


  // 函数：createTempCmdBuffer。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
  VkCommandBuffer createTempCmdBuffer();


  // 函数：tempSyncSubmit。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void            tempSyncSubmit(VkCommandBuffer cmd);


  void cmdBeginRendering(VkCommandBuffer    cmd,
                         bool               hasSecondary = false,
                         VkAttachmentLoadOp loadOpColor  = VK_ATTACHMENT_LOAD_OP_CLEAR,
                         VkAttachmentLoadOp loadOpDepth  = VK_ATTACHMENT_LOAD_OP_CLEAR);


  void cmdImageTransition(VkCommandBuffer cmd, nvvk::Image& rimg, VkImageAspectFlags aspects, VkImageLayout newLayout, bool needBarrier = false) const;


  template <typename T>
  VkResult createBufferTyped(nvvk::BufferTyped<T>&     buffer,
                             size_t                    elementCount,
                             VkBufferUsageFlagBits2    bufferUsageFlags,
                             VmaMemoryUsage            vmaMemUsage   = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                             VmaAllocationCreateFlags  vmaAllocFlags = 0,
                             VkDeviceSize              minAlignment  = 0,
                             std::span<const uint32_t> queueFamilies = {})
  {
    VkDeviceSize size = elementCount * nvvk::BufferTyped<T>::value_size;

    trackMemoryUsage(size, vmaMemUsage);
    return m_allocator.createBuffer(buffer, size, bufferUsageFlags,
                                    vmaMemUsage, vmaAllocFlags, minAlignment, queueFamilies);
  }

  VkResult createBuffer(nvvk::Buffer&             buffer,
                        VkDeviceSize              bufferSize,
                        VkBufferUsageFlagBits2    bufferUsageFlags,
                        VmaMemoryUsage            vmaMemUsage   = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                        VmaAllocationCreateFlags  vmaAllocFlags = 0,
                        VkDeviceSize              minAlignment  = 0,
                        std::span<const uint32_t> queueFamilies = {})
  {

    trackMemoryUsage(bufferSize, vmaMemUsage);
    return m_allocator.createBuffer(buffer, bufferSize, bufferUsageFlags, vmaMemUsage, vmaAllocFlags, minAlignment, queueFamilies);
  }


  // 函数：createLargeBuffer。创建对象、资源或描述符，并返回后续流程可直接使用的句柄。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：创建函数需要把失败路径和资源所有权定义清楚，便于调用方在异常或重建时回收。
  VkResult createLargeBuffer(nvvk::LargeBuffer& buffer, VkDeviceSize bufferSize, VkBufferUsageFlagBits2 bufferUsageFlags)
  {

    trackMemoryUsage(bufferSize, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    return m_allocator.createLargeBuffer(buffer, bufferSize, bufferUsageFlags, m_queue.queue);
  }


  // 函数：getDeviceLocalHeapSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  VkDeviceSize getDeviceLocalHeapSize() const;


  // 函数：isBufferSizeValid。返回条件判断结果，用于调用方选择后续分支或验证输入状态。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：谓词函数应保持无副作用或低副作用，使调用方可以安全地把它用于断言、过滤和早退。
  bool isBufferSizeValid(VkDeviceSize size) const;


  // 函数：simpleUploadBuffer。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
  void simpleUploadBuffer(const nvvk::Buffer& buffer, void* data)
  {

    VkCommandBuffer cmd = createTempCmdBuffer();

    m_uploader.appendBuffer(buffer, 0, buffer.bufferSize, data);

    m_uploader.cmdUploadAppended(cmd);

    tempSyncSubmit(cmd);

    m_uploader.releaseStaging();
  }


  // 函数：simpleUploadBuffer。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
  void simpleUploadBuffer(const nvvk::Buffer& buffer, size_t offset, size_t sz, void* data)
  {

    VkCommandBuffer cmd = createTempCmdBuffer();

    m_uploader.appendBuffer(buffer, offset, sz, data);

    m_uploader.cmdUploadAppended(cmd);

    tempSyncSubmit(cmd);

    m_uploader.releaseStaging();
  }


  // 枚举：FlushState。集中定义本模块可选模式或状态值，避免调用点使用裸整数。
  // 设计意图：把实验开关、渲染模式或阶段编号显式命名，使配置文件、UI 和代码路径可以互相对应。
  // 使用约束：新增枚举值时需要同步 UI 文本、参数解析和相关 switch 分支。
  enum FlushState
  {
    ALLOW_FLUSH,
    DONT_FLUSH,
  };


  // 类型：MemoryPool。封装本模块的长期状态、资源所有权和对外操作接口。
  // 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
  // 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
  class MemoryPool
{
public:


  // 结构：MemoryBlock。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct MemoryBlock
  {
    void* ptr;
    size_t size;
    bool inUse;
  };


  MemoryPool(size_t blockSize, size_t initialBlocks = 32, size_t alignment = 16)
      : m_blockSize(nvutils::align_up(blockSize, alignment))

      , m_alignment(alignment)
  {


    m_blocks.reserve(initialBlocks * 2);

    m_freeBlocks.reserve(initialBlocks);

    for(size_t i = 0; i < initialBlocks; i++)
    {
      void* ptr = nullptr;
      #ifdef _WIN32


      ptr = _aligned_malloc(m_blockSize, alignment);
      #else


      ptr = aligned_alloc(alignment, m_blockSize);
      #endif
      if(ptr)
      {
        m_blocks.push_back({ptr, m_blockSize, false});
        m_freeBlocks.push_back(m_blocks.size() - 1);
      }
    }
  }


  ~MemoryPool()
  {
    for(auto& block : m_blocks)
    {
      if(block.ptr)
      {
        #ifdef _WIN32


        _aligned_free(block.ptr);
        #else


        free(block.ptr);
        #endif
      }
    }
  }


  // 函数：allocate。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void* allocate()
  {
    if(!m_freeBlocks.empty())
    {

      size_t index = m_freeBlocks.back();

      m_freeBlocks.pop_back();
      m_blocks[index].inUse = true;
      return m_blocks[index].ptr;
    }


    void* ptr = nullptr;
    #ifdef _WIN32


    ptr = _aligned_malloc(m_blockSize, m_alignment);
    #else


    ptr = aligned_alloc(m_alignment, m_blockSize);
    #endif
    if(ptr)
    {
      m_blocks.push_back({ptr, m_blockSize, true});
      return ptr;
    }

    return nullptr;
  }


  // 函数：free。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void free(void* ptr)
  {
    if(!ptr) return;

    for(size_t i = 0; i < m_blocks.size(); i++)
    {
      if(m_blocks[i].ptr == ptr && m_blocks[i].inUse)
      {
        m_blocks[i].inUse = false;

        m_freeBlocks.push_back(i);
        break;
      }
    }
  }


  // 函数：allocateMultiple。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  std::vector<void*> allocateMultiple(size_t count)
  {
    std::vector<void*> results;

    results.reserve(count);

    for(size_t i = 0; i < count; i++)
    {

      void* ptr = allocate();
      if(ptr)

        results.push_back(ptr);
      else
        break;
    }

    return results;
  }


  // 函数：freeMultiple。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void freeMultiple(const std::vector<void*>& ptrs)
  {
    for(void* ptr : ptrs)
    {

      free(ptr);
    }
  }

  size_t getBlockSize() const { return m_blockSize; }
  size_t getTotalBlocks() const { return m_blocks.size(); }
  size_t getFreeBlocks() const { return m_freeBlocks.size(); }
  size_t getAlignment() const { return m_alignment; }

private:
  size_t m_blockSize;
  size_t m_alignment;
  std::vector<MemoryBlock> m_blocks;
  std::vector<size_t> m_freeBlocks;
};


// 类型：BatchedUploader。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class BatchedUploader
{
public:

  BatchedUploader(Resources& resources, VkDeviceSize maxBatchSize = 128 * 1024 * 1024)

      : m_resources(resources)

      , m_maxBatchSize(maxBatchSize)
  {
  }


  // 函数：getCmd。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
  VkCommandBuffer getCmd()
  {
    if(!m_cmd)
    {

      m_cmd = m_resources.createTempCmdBuffer();
    }
    return m_cmd;
  }

  template <typename T>

  T* uploadBuffer(const nvvk::Buffer& dst, size_t offset, size_t sz, const T* src, FlushState flushState = FlushState::ALLOW_FLUSH)
  {
    if(sz)
    {
      if(m_resources.m_uploader.checkAppendedSize(m_maxBatchSize, sz) && flushState == FlushState::ALLOW_FLUSH)
      {

        flush();
      }

      if(!m_cmd)
      {

        m_cmd = m_resources.createTempCmdBuffer();
      }
      T* mapping = nullptr;
      NVVK_CHECK(m_resources.m_uploader.appendBufferMapping(dst, offset, sz, mapping));

      if(src)
      {

        memcpy(mapping, src, sz);
      }

      return mapping;
    }
    return nullptr;
  }

  template <typename T>

  T* uploadBuffer(const nvvk::Buffer& dst, const T* src, FlushState flushState = FlushState::ALLOW_FLUSH)
  {
    return uploadBuffer(dst, 0, dst.bufferSize, src, flushState);
  }


  // 函数：fillBuffer。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void fillBuffer(const nvvk::Buffer& dst, uint32_t fillValue)
  {
    if(!m_cmd)
    {

      m_cmd = m_resources.createTempCmdBuffer();
    }

    vkCmdFillBuffer(m_cmd, dst.buffer, 0, dst.bufferSize, fillValue);
  }


  // 函数：flush。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void flush()
  {
    if(m_cmd)
    {

      m_resources.m_uploader.cmdUploadAppended(m_cmd);

      m_resources.tempSyncSubmit(m_cmd);

      m_resources.m_uploader.releaseStaging();
      m_cmd = nullptr;
    }
  }


  // 函数：abort。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void abort()
  {

    m_resources.m_uploader.cancelAppended();

    m_resources.m_uploader.releaseStaging();
  }

  ~BatchedUploader() { assert(!m_cmd && "must call flush at end"); }

private:
  Resources&      m_resources;
  VkDeviceSize    m_maxBatchSize = 0;
  VkCommandBuffer m_cmd          = nullptr;
};


  static constexpr VkPipelineStageFlags2 s_supportedShaderStages =
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT
      | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

  VkDevice         m_device          = {};
  VkPhysicalDevice m_physicalDevice  = {};
  nvvk::QueueInfo  m_queue           = {};
  nvvk::QueueInfo  m_queueTransfer   = {};
  VkCommandPool    m_tempCommandPool = {};

  nvvk::ResourceAllocator m_allocator     = {};
  nvvk::SamplerPool       m_samplerPool   = {};
  VkSampler               m_samplerLinear = {};
  nvvkglsl::GlslCompiler  m_glslCompiler  = {};
  nvvk::StagingUploader   m_uploader      = {};

  FrameBuffer m_frameBuffer;


  // 结构：CommonBuffers。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct CommonBuffers
  {
    nvvk::BufferTyped<shaderio::FrameConstants> frameConstants;
    nvvk::BufferTyped<shaderio::Readback>       readBack;
    nvvk::BufferTyped<shaderio::Readback>       readBackHost;
  } m_commonBuffers;

  nvvk::PhysicalDeviceInfo         m_physicalDeviceInfo = {};
  VkPhysicalDeviceMemoryProperties m_memoryProperties   = {};
  nvvk::GraphicsPipelineState      m_basicGraphicsState = {};
  uint32_t                         m_cycleIndex         = 0;
  size_t                           m_fboChangeID        = ~0;
  glm::vec4                        m_bgColor            = {0, 0, 0, 1.0};

  VkPhysicalDeviceMeshShaderPropertiesEXT m_meshShaderPropsEXT = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
  VkPhysicalDeviceMeshShaderPropertiesNV m_meshShaderPropsNV = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_NV};

  bool m_use16bitDispatch          = false;
  bool m_supportsMeshShaderNV      = false;
  bool m_supportsBarycentrics      = false;
  bool m_supportsSmBuiltinsNV      = false;
  bool m_dumpSpirv                 = false;
  NVHizVK                       m_hiz;

  NVHizVK::Update               m_hizUpdate[2];
  shaderc::SpvCompilationResult m_hizShaders[NVHizVK::SHADER_COUNT];

  QueueStateManager m_queueStates;
  VrdxSorter        m_vrdxSorter{};


  std::unique_ptr<MemoryPool> m_tempCmdBufferPool;
  std::vector<VkCommandBuffer> m_cmdBuffers;
  std::vector<bool> m_cmdBuffersInUse;
  uint32_t m_cmdBufferCount = 0;


  // 结构：MemoryUsage。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct MemoryUsage
  {
    size_t deviceLocal = 0;
    size_t hostVisible = 0;
    size_t hostCached = 0;
    size_t total = 0;
  } m_memoryUsage;


  // 函数：trackMemoryUsage。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void trackMemoryUsage(VkDeviceSize size, VmaMemoryUsage usage);


  // 函数：logMemoryUsage。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void logMemoryUsage() const;

private:
};


}
