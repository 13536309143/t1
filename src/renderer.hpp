#pragma once
#if __INTELLISENSE__
#undef VK_NO_PROTOTYPES
#endif
#include <memory>

#include <nvvk/compute_pipeline.hpp>
#include "resources.hpp"
#include "scene.hpp"
#include "preloaded.hpp"
#include "streaming.hpp"
namespace lodclusters {

class RenderScene
{
public:
  const Scene*   scene        = nullptr;
  bool           useStreaming = false;
  ScenePreloaded scenePreloaded;
  SceneStreaming sceneStreaming;

  // pointers must stay valid during lifetime
  bool init(Resources* res, const Scene* scene_, const StreamingConfig& streamingConfig_, bool useStreaming_);
  void deinit();

  void streamingReset();

  const nvvk::BufferTyped<shaderio::Geometry>& getShaderGeometriesBuffer() const;

  size_t                                       getOperationsSize() const;
  size_t                                       getGeometrySize(bool reserved) const;
};

struct RendererConfig
{
  bool flipWinding               = false;
  bool forceTwoSided             = false;
  bool useForcedInvisibleCulling = false;
  bool useSorting                = false;
  bool useRenderStats            = false;
  bool useCulling                = true;
  bool useTwoPassCulling         = false;
  bool useShading                = true;
  bool useDebugVisualization     = true;
  bool useSeparateGroups         = true;
  bool useEXTmeshShader          = false;
  bool useComputeRaster          = false;
  bool useAdaptiveRasterRouting  = false;
  bool usePrimitiveCulling       = false;
  bool useDepthOnly              = false;
  // the maximum number of renderable clusters per frame in bits i.e. (1 << number)
  uint32_t numRenderClusterBits = 22;
  // the maximum number of traversal intermediate tasks
  uint32_t numTraversalTaskBits = 22;
};

class Renderer
{
public:
  virtual bool init(Resources& res, RenderScene& rscene, const RendererConfig& config) = 0;
  virtual void render(VkCommandBuffer primary, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler) = 0;
  virtual void deinit(Resources& res) = 0;
  virtual ~Renderer() {};  // Defined only so that inherited classes also have virtual destructors. Use deinit().
  virtual void updatedFrameBuffer(Resources& res, RenderScene& rscene) { updateBasicDescriptors(res, rscene); };
  struct ResourceUsageInfo
  {
    size_t operationsMemBytes{};
    size_t geometryMemBytes{};
    void add(const ResourceUsageInfo& other)
    {
      operationsMemBytes += other.operationsMemBytes;
      geometryMemBytes += other.geometryMemBytes;
    }
    size_t getTotalSum() const
    {
      return geometryMemBytes + operationsMemBytes;
    }
  };
  inline ResourceUsageInfo getResourceUsage(bool reserved) const
  {
    return reserved ? m_resourceReservedUsage : m_resourceActualUsage;
  };

  uint32_t getMaxRenderClusters() const { return m_maxRenderClusters; }
  uint32_t getMaxTraversalTasks() const { return m_maxTraversalTasks; }

protected:
  void initBasics(Resources& res, RenderScene& rscene, const RendererConfig& config);
  void deinitBasics(Resources& res);
  bool initBasicShaders(Resources& res, RenderScene& rscene, const RendererConfig& config);
  void initBasicPipelines(Resources& res, RenderScene& rscene, const RendererConfig& config);
  void updateBasicDescriptors(Resources& res, RenderScene& scene, const nvvk::Buffer* sceneBuildBuffer = nullptr);
  void writeAtomicRaster(VkCommandBuffer cmd);
  void writeBackgroundSky(VkCommandBuffer cmd);
  void renderInstanceBboxes(VkCommandBuffer cmd, uint32_t selectedInstanceID, bool selectedOnly);
  void renderClusterBboxes(VkCommandBuffer cmd, nvvk::Buffer sceneBuildBuffer);
  struct BasicShaders
  {
    shaderc::SpvCompilationResult fullScreenVertexShader;
    shaderc::SpvCompilationResult fullScreenWriteDepthFragShader;
    shaderc::SpvCompilationResult fullScreenBackgroundFragShader;
    shaderc::SpvCompilationResult fullscreenAtomicRasterFragmentShader;
    shaderc::SpvCompilationResult renderInstanceBboxesFragmentShader;
    shaderc::SpvCompilationResult renderInstanceBboxesMeshShader;
    shaderc::SpvCompilationResult renderClusterBboxesMeshShader;
    shaderc::SpvCompilationResult renderClusterBboxesFragmentShader;
  };
  struct BasicPipelines
  {
    VkPipeline writeDepth{};
    VkPipeline background{};
    VkPipeline atomicRaster{};
    VkPipeline renderInstanceBboxes{};
    VkPipeline renderClusterBboxes{};
  };

  RendererConfig m_config;
  uint32_t       m_maxRenderClusters       = 0;
  uint32_t       m_maxTraversalTasks       = 0;
  uint32_t       m_meshShaderWorkgroupSize = 0;
  uint32_t       m_meshShaderBoxes         = 0;
  uint32_t       m_frameIndex              = 0;
  BasicShaders   m_basicShaders;
  BasicPipelines m_basicPipelines;

  std::vector<shaderio::RenderInstance> m_renderInstances;
  nvvk::Buffer                          m_renderInstanceBuffer;

  ResourceUsageInfo m_resourceReservedUsage{};
  ResourceUsageInfo m_resourceActualUsage{};

  nvvk::DescriptorPack m_basicDset;
  VkShaderStageFlags   m_basicShaderFlags{};
  VkPipelineLayout     m_basicPipelineLayout{};

  nvvk::Buffer m_sortingAuxBuffer;
};

/////////////////////////////////////////////////////////////////////////
inline float clusterLodErrorOverDistance(float errorSizeInPixels, float fov, float resolution)
{
  return (tanf(fov * 0.5f) * errorSizeInPixels / resolution);
}
//////////////////////////////////////////////////////////////////////////
std::unique_ptr<Renderer> makeRendererRasterClustersLod();

}
