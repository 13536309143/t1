#pragma once
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
#include "../shaders/shaderio.h"

namespace lodclusters {

struct FrameConfig
{
  VkExtent2D windowSize;

  bool  showInstanceBboxes = false;
  bool  showClusterBboxes  = false;
  bool  highlightSelectedInstance = false;
  bool  freezeCulling      = false;
  bool  freezeLoD          = false;
  uint32_t selectedInstanceID = ~0u;
  float lodPixelError      = 1.0f;
  // increase error by this for instances not having primary visibility in ray tracing
  float culledErrorScale = 2.0f;
  // if less pixels than this, use sw raster
  float swRasterThreshold = 8.0f;
  // if more triangles than this per projected pixel, prefer sw raster for tiny dense clusters
  float swRasterTriangleDensityThreshold = 0.5f;
  bool  swRasterFeedbackEnabled = false;
  float swRasterFeedbackTargetTriangleShare = 0.15f;
  float swRasterThresholdEffective = 8.0f;
  float swRasterTriangleDensityThresholdEffective = 0.5f;

  // how many frames until we schedule a group for unloading
  uint32_t streamingAgeThreshold = 16;

  // how much threads to use in the persistent kernels
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

//////////////////////////////////////////////////////////////////////////

inline void cmdCopyBuffer(VkCommandBuffer cmd, const nvvk::Buffer& src, const nvvk::Buffer& dst)
{
  VkBufferCopy cpy = {0, 0, src.bufferSize};
  vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &cpy);
}

std::string formatMemorySize(size_t sizeInBytes);

inline size_t logMemoryUsage(size_t size, const char* memtype, const char* what)
{
  LOGI("%s memory: %s - %s\n", memtype, formatMemorySize(size).c_str(), what);
  return size;
}

//////////////////////////////////////////////////////////////////////////

struct BufferRanges
{
  VkDeviceSize tempOffset = 0;

  VkDeviceSize beginOffset = 0;
  VkDeviceSize splitOffset = 0;

  VkDeviceSize append(VkDeviceSize size, VkDeviceSize alignment)
  {
    tempOffset = nvutils::align_up(tempOffset, alignment);

    VkDeviceSize offset = tempOffset;
    tempOffset += size;

    return offset;
  }

  void beginOverlap()
  {
    beginOffset = tempOffset;
    splitOffset = 0;
  }
  void splitOverlap()
  {
    splitOffset = std::max(splitOffset, tempOffset);
    tempOffset  = beginOffset;
  }
  void endOverlap() { tempOffset = std::max(splitOffset, tempOffset); }

  VkDeviceSize getSize(VkDeviceSize alignment = 4) { return nvutils::align_up(tempOffset, alignment); }
};

//////////////////////////////////////////////////////////////////////////

class QueueState
{
public:
  VkDevice    m_device            = nullptr;
  VkQueue     m_queue             = nullptr;
  uint32_t    m_familyIndex       = 0;
  VkSemaphore m_timelineSemaphore = nullptr;
  uint64_t    m_timelineValue     = 1;

  std::vector<VkSemaphoreSubmitInfo> m_pendingWaits;

  void init(VkDevice device, VkQueue queue, uint32_t familyIndex, uint64_t initialValue);
  void deinit();

  VkResult getTimelineValue(uint64_t& timelineValue) const
  {
    return vkGetSemaphoreCounterValue(m_device, m_timelineSemaphore, &timelineValue);
  }

  nvvk::SemaphoreState getCurrentState() const
  {
    return nvvk::SemaphoreState::makeFixed(m_timelineSemaphore, m_timelineValue);
  }

  VkSemaphoreSubmitInfo getWaitSubmit(VkPipelineStageFlags2 stageMask, uint32_t deviceIndex = 0) const;

  // increments timeline
  VkSemaphoreSubmitInfo advanceSignalSubmit(VkPipelineStageFlags2 stageMask, uint32_t deviceIndex = 0);
};

struct QueueStateManager
{
  QueueState primary;
  QueueState transfer;
};

//////////////////////////////////////////////////////////////////////////

class Resources
{
public:
  static constexpr VkPipelineStageFlags2 ALL_SHADER_STAGES =
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;


  struct FrameBuffer
  {
    VkExtent2D renderSize{};
    VkExtent2D targetSize{};
    VkExtent2D windowSize{};

    // typically super resolution with respect to the window size
    // 0: off - use window resolution
    // 1: off - use window resolution
    // 2: 2x resolution along width and height
    // 720:  fix render resolution to 1280 x 720, aspect from window
    // 1080: fix render resolution to 1920 x 1080, aspect from window
    // 1440: fix render resolution to 2560 x 1440, aspect from window
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
    //nvvk::Image imgHizFar = {};
    // 现在：双 HIZ 图像数组
    nvvk::Image imgHizFar[2] = {};

    VkPipelineRenderingCreateInfo pipelineRenderingInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  };

  void init(VkDevice device, VkPhysicalDevice physicalDevice, VkInstance instance, const nvvk::QueueInfo& queue, const nvvk::QueueInfo& queueTransfer);
  void deinit();

  bool initFramebuffer(const VkExtent2D& windowSize, int supersample);
  void updateFramebufferRenderSizeDependent(VkCommandBuffer cmd);
  void deinitFramebufferRenderSizeDependent();
  void deinitFramebuffer();
  glm::vec2 getFramebufferWindow2RenderScale() const;
  void beginFrame(uint32_t cycleIndex);
  void postProcessFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);
  void emptyFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);
  void endFrame();
  //void cmdBuildHiz(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);添加索引参数
  void cmdBuildHiz(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler, uint32_t idx);

  // some vulkan implementations only support 16 bit per grid component
  // need to convert the 1D intended launch into a grid.
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

  void getReadbackData(shaderio::Readback& readback);

  //////////////////////////////////////////////////////////////////////////

  shaderc::CompileOptions makeCompilerOptions() { return shaderc::CompileOptions(m_glslCompiler.options()); }

  bool compileShader(shaderc::SpvCompilationResult& compiled,
                     VkShaderStageFlagBits          shader,
                     const std::filesystem::path&   filePath,
                     shaderc::CompileOptions*       options = nullptr);

  // tests if all shaders compiled well, returns false if not
  // also destroys all shaders if not all were successful.
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
  bool verifyShaders(T& container)
  {
    return verifyShaders(sizeof(T) / sizeof(shaderc::SpvCompilationResult), (shaderc::SpvCompilationResult*)&container);
  }

  void destroyPipelines(size_t numPipelines, VkPipeline* pipelines)
  {
    for(size_t i = 0; i < numPipelines; i++)
    {
      vkDestroyPipeline(m_device, pipelines[i], nullptr);
      pipelines[i] = nullptr;
    }
  }
  template <typename T>
  void destroyPipelines(T& container)
  {
    destroyPipelines(sizeof(T) / sizeof(VkPipeline), (VkPipeline*)&container);
  }

  //////////////////////////////////////////////////////////////////////////

  VkCommandBuffer createTempCmdBuffer();
  void            tempSyncSubmit(VkCommandBuffer cmd);

  //////////////////////////////////////////////////////////////////////////

  void cmdBeginRendering(VkCommandBuffer    cmd,
                         bool               hasSecondary = false,
                         VkAttachmentLoadOp loadOpColor  = VK_ATTACHMENT_LOAD_OP_CLEAR,
                         VkAttachmentLoadOp loadOpDepth  = VK_ATTACHMENT_LOAD_OP_CLEAR);

  void cmdImageTransition(VkCommandBuffer cmd, nvvk::Image& rimg, VkImageAspectFlags aspects, VkImageLayout newLayout, bool needBarrier = false) const;

  //////////////////////////////////////////////////////////////////////////

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

  VkResult createLargeBuffer(nvvk::LargeBuffer& buffer, VkDeviceSize bufferSize, VkBufferUsageFlagBits2 bufferUsageFlags)
  {
    trackMemoryUsage(bufferSize, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    return m_allocator.createLargeBuffer(buffer, bufferSize, bufferUsageFlags, m_queue.queue);
  }

  VkDeviceSize getDeviceLocalHeapSize() const;

  bool isBufferSizeValid(VkDeviceSize size) const;

  //////////////////////////////////////////////////////////////////////////

  void simpleUploadBuffer(const nvvk::Buffer& buffer, void* data)
  {
    VkCommandBuffer cmd = createTempCmdBuffer();
    m_uploader.appendBuffer(buffer, 0, buffer.bufferSize, data);
    m_uploader.cmdUploadAppended(cmd);
    tempSyncSubmit(cmd);
    m_uploader.releaseStaging();
  }

  void simpleUploadBuffer(const nvvk::Buffer& buffer, size_t offset, size_t sz, void* data)
  {
    VkCommandBuffer cmd = createTempCmdBuffer();
    m_uploader.appendBuffer(buffer, offset, sz, data);
    m_uploader.cmdUploadAppended(cmd);
    tempSyncSubmit(cmd);
    m_uploader.releaseStaging();
  }

  enum FlushState
  {
    ALLOW_FLUSH,
    DONT_FLUSH,
  };

  class MemoryPool
{
public:
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
    // 预分配更大的初始块数，减少动态扩容
    m_blocks.reserve(initialBlocks * 2);
    m_freeBlocks.reserve(initialBlocks);
    
    for(size_t i = 0; i < initialBlocks; i++)
    {
      void* ptr = nullptr;
      #ifdef _WIN32
      // Windows 平台使用 _aligned_malloc
      ptr = _aligned_malloc(m_blockSize, alignment);
      #else
      // 其他平台使用 aligned_alloc
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
        // Windows 平台使用 _aligned_free
        _aligned_free(block.ptr);
        #else
        // 其他平台使用 free
        free(block.ptr);
        #endif
      }
    }
  }

  void* allocate()
  {
    if(!m_freeBlocks.empty())
    {
      size_t index = m_freeBlocks.back();
      m_freeBlocks.pop_back();
      m_blocks[index].inUse = true;
      return m_blocks[index].ptr;
    }

    // No free blocks, allocate new one
    void* ptr = nullptr;
    #ifdef _WIN32
    // Windows 平台使用 _aligned_malloc
    ptr = _aligned_malloc(m_blockSize, m_alignment);
    #else
    // 其他平台使用 aligned_alloc
    ptr = aligned_alloc(m_alignment, m_blockSize);
    #endif
    if(ptr)
    {
      m_blocks.push_back({ptr, m_blockSize, true});
      return ptr;
    }

    return nullptr;
  }

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

  // 批量分配多个块
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

  // 批量释放多个块
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

class BatchedUploader
{
public:
  BatchedUploader(Resources& resources, VkDeviceSize maxBatchSize = 128 * 1024 * 1024)
      : m_resources(resources)
      , m_maxBatchSize(maxBatchSize)
  {
  }

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

  void fillBuffer(const nvvk::Buffer& dst, uint32_t fillValue)
  {
    if(!m_cmd)
    {
      m_cmd = m_resources.createTempCmdBuffer();
    }
    vkCmdFillBuffer(m_cmd, dst.buffer, 0, dst.bufferSize, fillValue);
  }

  // must call flush at end of operations
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

//////////////////////////////////////////////////////////////////////////

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
  //NVHizVK::Update               m_hizUpdate;
  NVHizVK::Update               m_hizUpdate[2];
  shaderc::SpvCompilationResult m_hizShaders[NVHizVK::SHADER_COUNT];

  QueueStateManager m_queueStates;
  VrdxSorter        m_vrdxSorter{};

  // Memory pools for efficient memory management
  std::unique_ptr<MemoryPool> m_tempCmdBufferPool;
  std::vector<VkCommandBuffer> m_cmdBuffers;
  std::vector<bool> m_cmdBuffersInUse;
  uint32_t m_cmdBufferCount = 0;

  // Memory usage tracking
  struct MemoryUsage
  {
    size_t deviceLocal = 0;
    size_t hostVisible = 0;
    size_t hostCached = 0;
    size_t total = 0;
  } m_memoryUsage;

  void trackMemoryUsage(VkDeviceSize size, VmaMemoryUsage usage);
  void logMemoryUsage() const;

private:
};


}  // namespace lodclusters
