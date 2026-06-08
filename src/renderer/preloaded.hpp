//==============================================================================
// 文件：src/renderer/preloaded.hpp
// 模块定位：预加载 GPU 场景声明，描述一次性上传全量几何数据的资源结构和查询接口。
// 数据流：输入是 Scene；输出是 着色器 可通过设备地址访问的 Geometry、Group 和 簇 缓冲。
// 方法说明：预加载模式以更高显存占用换取最简单的运行时访问路径，适合作为 流式加载 模式的性能上界参照。
// 正确性约束：只有当显存预算足够时才能启用；上传后所有 缓冲 地址必须在 renderer 生命周期内稳定。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#pragma once


// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include "scene.hpp"
#include "resources.hpp"


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace lodclusters {


// 类型：ScenePreloaded。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class ScenePreloaded
{
public:


  // 结构：Config。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct Config
  {
  };


  // 函数：canPreload。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
  static bool canPreload(VkDeviceSize, const Scene* scene);


  // 函数：init。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  bool init(Resources* res, const Scene* scene, const Config& config);


  // 函数：deinit。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void deinit();


  const nvvk::BufferTyped<shaderio::Geometry>& getShaderGeometriesBuffer() const { return m_shaderGeometriesBuffer; }


  size_t getGeometrySize() const { return m_geometrySize; }
  size_t getOperationsSize() const { return m_operationsSize; }

private:


  // 结构：Geometry。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct Geometry
  {
    nvvk::BufferTyped<shaderio::LodLevel> lodLevels;
    nvvk::BufferTyped<shaderio::Node>     lodNodes;
    nvvk::BufferTyped<shaderio::BBox>     lodNodeBboxes;

    nvvk::Buffer                groupData;
    nvvk::BufferTyped<uint64_t> groupAddresses;
    nvvk::BufferTyped<uint64_t> clusterAddresses;
  };

  Config       m_config;
  Resources*   m_resources = nullptr;
  const Scene* m_scene     = nullptr;

  size_t m_geometrySize       = 0;
  size_t m_operationsSize     = 0;

  std::vector<ScenePreloaded::Geometry> m_geometries;
  std::vector<shaderio::Geometry>       m_shaderGeometries;

  nvvk::BufferTyped<shaderio::Geometry> m_shaderGeometriesBuffer;
};
}
