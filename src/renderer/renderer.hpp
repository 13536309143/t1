//==============================================================================
// 文件：src/renderer/renderer.hpp
// 模块定位：渲染器抽象声明，定义渲染配置、场景驻留访问、共享基础 管线 和调试绘制接口。
// 数据流：输入是 RenderScene、Resources 和 FrameConfig；输出是每帧命令缓冲中的绘制、后处理和统计更新。
// 方法说明：抽象层把“场景如何驻留”和“场景如何绘制”解耦，使预加载与流式加载共用渲染路径。
// 正确性约束：派生 renderer 必须遵守 init/render/deinit 顺序；共享 缓冲 和 descriptor 的生命周期由基类统一管理。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#pragma once
#if __INTELLISENSE__
#undef VK_NO_PROTOTYPES
#endif


// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <memory>

#include <nvvk/compute_pipeline.hpp>
#include "resources.hpp"
#include "scene.hpp"
#include "preloaded.hpp"
#include "streaming.hpp"


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace lodclusters {


// 类型：RenderScene。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class RenderScene
{
public:
  const Scene*   scene        = nullptr;
  bool           useStreaming = false;
  ScenePreloaded scenePreloaded;
  SceneStreaming sceneStreaming;


  // 函数：init。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  bool init(Resources* res, const Scene* scene_, const StreamingConfig& streamingConfig_, bool useStreaming_);


  // 函数：deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinit();


  // 函数：streamingReset。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void streamingReset();


  // 函数：getShaderGeometriesBuffer。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  const nvvk::BufferTyped<shaderio::Geometry>& getShaderGeometriesBuffer() const;


  // 函数：getOperationsSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  size_t                                       getOperationsSize() const;


  // 函数：getGeometrySize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  size_t                                       getGeometrySize(bool reserved) const;
};


// 结构：RendererConfig。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
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

  uint32_t numRenderClusterBits = 22;

  uint32_t numTraversalTaskBits = 22;
};


// 类型：Renderer。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class Renderer
{
public:
  virtual bool init(Resources& res, RenderScene& rscene, const RendererConfig& config) = 0;
  virtual void render(VkCommandBuffer primary, Resources& res, RenderScene& rscene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler) = 0;
  virtual void deinit(Resources& res) = 0;
  virtual ~Renderer() {};
  virtual void updatedFrameBuffer(Resources& res, RenderScene& rscene) { updateBasicDescriptors(res, rscene); };


  // 结构：ResourceUsageInfo。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct ResourceUsageInfo
  {
    size_t operationsMemBytes{};
    size_t geometryMemBytes{};


    // 函数：add。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    void add(const ResourceUsageInfo& other)
    {
      operationsMemBytes += other.operationsMemBytes;
      geometryMemBytes += other.geometryMemBytes;
    }


    // 函数：getTotalSum。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    size_t getTotalSum() const
    {
      return geometryMemBytes + operationsMemBytes;
    }
  };


  // 函数：getResourceUsage。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  inline ResourceUsageInfo getResourceUsage(bool reserved) const
  {
    return reserved ? m_resourceReservedUsage : m_resourceActualUsage;
  };

  uint32_t getMaxRenderClusters() const { return m_maxRenderClusters; }
  uint32_t getMaxTraversalTasks() const { return m_maxTraversalTasks; }

protected:


  // 函数：initBasics。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void initBasics(Resources& res, RenderScene& rscene, const RendererConfig& config);


  // 函数：deinitBasics。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinitBasics(Resources& res);


  // 函数：initBasicShaders。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  bool initBasicShaders(Resources& res, RenderScene& rscene, const RendererConfig& config);


  // 函数：initBasicPipelines。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void initBasicPipelines(Resources& res, RenderScene& rscene, const RendererConfig& config);

  void updateBasicDescriptors(Resources& res, RenderScene& scene, const nvvk::Buffer* sceneBuildBuffer = nullptr);


  // 函数：writeAtomicRaster。把当前状态写入缓存、缓冲、文件或着色器可消费的数据布局。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：写入路径应明确字节对齐、所有权和可见性，避免后续读取端解释错误。
  void writeAtomicRaster(VkCommandBuffer cmd);


  // 函数：writeBackgroundSky。把当前状态写入缓存、缓冲、文件或着色器可消费的数据布局。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：写入路径应明确字节对齐、所有权和可见性，避免后续读取端解释错误。
  void writeBackgroundSky(VkCommandBuffer cmd);


  // 函数：renderInstanceBboxes。录制或执行渲染相关工作，把准备好的数据提交到当前渲染阶段。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：渲染函数通常处于帧级关键路径，必须尊重前序计算阶段写出的计数、地址和同步屏障。
  void renderInstanceBboxes(VkCommandBuffer cmd);


  // 函数：renderClusterBboxes。录制或执行渲染相关工作，把准备好的数据提交到当前渲染阶段。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：渲染函数通常处于帧级关键路径，必须尊重前序计算阶段写出的计数、地址和同步屏障。
  void renderClusterBboxes(VkCommandBuffer cmd, nvvk::Buffer sceneBuildBuffer);


  // 结构：BasicShaders。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
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


  // 结构：BasicPipelines。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
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


// 函数：clusterLodErrorOverDistance。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
inline float clusterLodErrorOverDistance(float errorSizeInPixels, float fov, float resolution)
{
  return (tanf(fov * 0.5f) * errorSizeInPixels / resolution);
}


// 函数：makeRendererRasterClustersLod。录制或执行渲染相关工作，把准备好的数据提交到当前渲染阶段。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：渲染函数通常处于帧级关键路径，必须尊重前序计算阶段写出的计数、地址和同步屏障。
std::unique_ptr<Renderer> makeRendererRasterClustersLod();

}
