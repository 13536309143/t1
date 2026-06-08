//==============================================================================
// 文件：src/renderer/hiz.hpp
// 模块定位：Hi-Z 生成器声明，封装层级深度纹理、描述符、管线 和一次更新所需的视图信息。
// 数据流：输入是 帧缓冲 depth image；输出是可被 traversal 采样的层级深度 mip。
// 方法说明：Hi-Z 通过深度金字塔把遮挡测试从逐像素比较转化为保守的区域查询，是层次剔除的重要加速结构。
// 正确性约束：mip 尺寸、采样器和 reversed-Z 语义必须与剔除 着色器 一致；update view 生命周期依赖目标图像。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#pragma once


// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <stdint.h>
#include <string>
#include <vector>
#include <span>
#include <vulkan/vulkan_core.h>
#include <nvvkglsl/glsl.hpp>


// 类型：NVHizVK。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class NVHizVK
{
private:


  // 枚举：ProgViewMode。集中定义本模块可选模式或状态值，避免调用点使用裸整数。
  // 设计意图：把实验开关、渲染模式或阶段编号显式命名，使配置文件、UI 和代码路径可以互相对应。
  // 使用约束：新增枚举值时需要同步 UI 文本、参数解析和相关 switch 分支。
  enum ProgViewMode : uint32_t
  {
    PROG_VIEW_MONO,
    PROG_VIEW_STEREO,
    PROG_VIEW_COUNT,
  };


  // 枚举：ProgHizMode。集中定义本模块可选模式或状态值，避免调用点使用裸整数。
  // 设计意图：把实验开关、渲染模式或阶段编号显式命名，使配置文件、UI 和代码路径可以互相对应。
  // 使用约束：新增枚举值时需要同步 UI 文本、参数解析和相关 switch 分支。
  enum ProgHizMode : uint32_t
  {
    PROG_HIZ_FAR,
    PROG_HIZ_FAR_AND_NEAR,
    PROG_HIZ_FAR_REST,
    PROG_HIZ_COUNT,
  };

public:
  static const uint32_t MAX_MIP_LEVELS = 16;
  static const uint32_t SHADER_COUNT   = (uint32_t(PROG_HIZ_COUNT) * uint32_t(PROG_VIEW_COUNT));
  static const uint32_t POOLSIZE_COUNT = 2;


  // 枚举：BindingSlots。集中定义本模块可选模式或状态值，避免调用点使用裸整数。
  // 设计意图：把实验开关、渲染模式或阶段编号显式命名，使配置文件、UI 和代码路径可以互相对应。
  // 使用约束：新增枚举值时需要同步 UI 文本、参数解析和相关 switch 分支。
  enum BindingSlots
  {

    BINDING_READ_DEPTH,
    BINDING_READ_FAR,
    BINDING_WRITE_NEAR,
    BINDING_WRITE_FAR,
    BINDING_COUNT,
  };


  // 结构：TextureInfo。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct TextureInfo
  {

    uint32_t           width;
    uint32_t           height;
    uint32_t           mipLevels;
    VkFormat           format;
    VkImageAspectFlags aspect;


    uint32_t usedWidth;
    uint32_t usedHeight;


    // 函数：getShaderFactors。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    void  getShaderFactors(float factors[4]) const;


    // 函数：getSizeMax。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    float getSizeMax() const;
  };


  // 结构：Update。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct Update
  {

    VkImageView sourceImageView;
    VkImageView nearImageView;
    VkImageView farImageView;
    VkImageView farImageViews[MAX_MIP_LEVELS];

    VkDescriptorImageInfo farImageInfo;
    VkDescriptorImageInfo nearImageInfo;

    VkImage sourceImage;
    VkImage nearImage;
    VkImage farImage;

    TextureInfo sourceInfo;
    TextureInfo farInfo;
    TextureInfo nearInfo;
    bool        stereo;

    Update() { memset(this, 0, sizeof(Update)); }
  };


  // 结构：DescriptorUpdate。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct DescriptorUpdate
  {
    VkWriteDescriptorSet  writeSets[BINDING_COUNT];
    VkDescriptorImageInfo imageInfos[BINDING_COUNT + MAX_MIP_LEVELS - 1];
  };


  // 结构：Config。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct Config
  {
    int  msaaSamples             = 0;
    bool reversedZ               = false;
    bool supportsSubGroupShuffle = false;
    bool supportsMinmaxFilter    = false;
  };


  // 函数：init。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void init(VkDevice device, const Config& config, uint32_t descrSetsCount);


  // 函数：getReadFarSampler。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
  VkSampler                   getReadFarSampler() const;


  // 函数：getDescriptorPoolSizes。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  const VkDescriptorPoolSize* getDescriptorPoolSizes(uint32_t& count) const;


  // 函数：getDescriptorSetLayout。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  VkDescriptorSetLayout       getDescriptorSetLayout() const;


  // 函数：appendShaderDefines。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void                        appendShaderDefines(uint32_t shader, shaderc::CompileOptions& options) const;


  // 函数：initPipelines。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void                        initPipelines(const shaderc::SpvCompilationResult spvResults[SHADER_COUNT]);


  // 函数：deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinit();


  // 函数：setupUpdateInfos。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void setupUpdateInfos(Update& update, uint32_t width, uint32_t height, VkFormat sourceFormat, VkImageAspectFlags sourceAspect) const;


  // 函数：setupDescriptorUpdate。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void setupDescriptorUpdate(DescriptorUpdate& updateWrite, const Update& update, VkDescriptorSet set) const;


  // 函数：cmdUpdateHiz。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
  void cmdUpdateHiz(VkCommandBuffer cmd, const Update& update, VkDescriptorSet set) const;


  // 函数：initUpdateViews。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void initUpdateViews(Update& update) const;


  // 函数：deinitUpdateViews。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinitUpdateViews(Update& update) const;


  // 函数：updateDescriptorSet。根据最新状态刷新缓存数据、GPU 地址、描述符或统计信息。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：更新函数负责把“旧状态”推进到“当前状态”，因此要避免部分更新造成 CPU/GPU 视图不一致。
  void updateDescriptorSet(const Update& update, uint32_t setIdx) const;


  // 函数：cmdUpdateHiz。向命令缓冲录制 GPU 操作，并依赖外层调用者安排提交与同步。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该类函数只描述命令序列，不应假设命令已经立即执行。
  void cmdUpdateHiz(VkCommandBuffer cmd, const Update& update, uint32_t setIdx) const
  {

    cmdUpdateHiz(cmd, update, m_descrSets[setIdx]);
  }

private:


  // 结构：InternalConfig。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct InternalConfig : public Config
  {
    uint32_t hizLevels    = 1;
    uint32_t hizNearLevel = 0;
    uint32_t hizFarLevel  = 0;
  };


  // 函数：getShaderIndexConfig。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  static void getShaderIndexConfig(uint32_t index, ProgHizMode& hiz, ProgViewMode& view)
  {
    hiz  = ProgHizMode(index % uint32_t(PROG_HIZ_COUNT));
    view = ProgViewMode(index / uint32_t(PROG_HIZ_COUNT));
  }

  static uint32_t getShaderIndex(ProgHizMode hiz, ProgViewMode view) { return view * uint32_t(PROG_HIZ_COUNT) + hiz; }


  // 结构：PushConstants。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct PushConstants
  {

    int srcSize[4];
    int writeLod;
    int startLod;
    int layer;
    int _pad0;
    int levelActive[4];
  };


  // 函数：deinitPipelines。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinitPipelines();

  InternalConfig        m_config                  = {};
  VkDevice              m_device                  = {};
  VkSampler             m_readDepthSampler        = {};
  VkSampler             m_readFarSampler          = {};
  VkSampler             m_readNearSampler         = {};
  VkPipeline            m_pipelines[SHADER_COUNT] = {0};
  VkPipelineLayout      m_pipelineLayout          = {};
  VkDescriptorSetLayout m_descrLayout             = {};
  VkDescriptorPoolSize  m_poolSizes[POOLSIZE_COUNT];
  uint32_t              m_descrSetsCount = 0;
  VkDescriptorPool      m_descrPool      = {};
  VkDescriptorSet*      m_descrSets      = {};
};
