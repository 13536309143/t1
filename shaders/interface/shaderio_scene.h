//==============================================================================
// 文件：shaders/interface/shaderio_scene.h
// 模块定位：CPU 与 GPU 共享布局文件，定义着色器和 C++ 共同理解的数据结构、常量和访问约定。
// 数据流：CPU 侧填充这些结构，GPU 侧按完全相同的内存布局读取和写回。
// 方法说明：共享布局是异构系统的 ABI，任何字段顺序、对齐和位域变化都会影响两侧解释一致性。
// 正确性约束：结构对齐、标量布局和 缓冲 reference 类型必须与 Vulkan/GLSL 编译选项一致。
// 注释风格：使用中文解释 GPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
// 依赖说明：引入共享布局、剔除、着色或阶段间复用的着色器片段。
// 这些 include 共同决定本文件能访问的结构布局、数学辅助函数和编译期宏。
#include "shaderio_core.h"
#ifndef _SHADERIO_SCENE_H_


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define _SHADERIO_SCENE_H_
#ifdef __cplusplus


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace shaderio {
using namespace glm;


// 枚举：ClusterAttributeBits。集中定义本模块可选模式或状态值，避免调用点使用裸整数。
// 设计意图：把实验开关、渲染模式或阶段编号显式命名，使配置文件、UI 和代码路径可以互相对应。
// 使用约束：新增枚举值时需要同步 UI 文本、参数解析和相关 switch 分支。
enum ClusterAttributeBits
{
  CLUSTER_ATTRIBUTE_VERTEX_NORMAL           = 1,
  CLUSTER_ATTRIBUTE_VERTEX_TANGENT          = 2,
  CLUSTER_ATTRIBUTE_VERTEX_TEX_0            = 4,
  CLUSTER_ATTRIBUTE_VERTEX_TEX_1            = 8,
  CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_TEX_0 = 32,
  CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_TEX_1 = 64,
  CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_POS   = 128,
};

#else


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define CLUSTER_ATTRIBUTE_VERTEX_NORMAL 1


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define CLUSTER_ATTRIBUTE_VERTEX_TANGENT 2


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define CLUSTER_ATTRIBUTE_VERTEX_TEX_0 4


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define CLUSTER_ATTRIBUTE_VERTEX_TEX_1 8


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_TEX_0 32


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_TEX_1 64


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_POS 128

#ifndef CLUSTER_VERTEX_COUNT


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define CLUSTER_VERTEX_COUNT 32
#endif

#ifndef CLUSTER_TRIANGLE_COUNT


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define CLUSTER_TRIANGLE_COUNT 32
#endif

#endif


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define SHADERIO_ORIGINAL_MESH_GROUP 0xffffffffu


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define SHADERIO_MAX_LOD_LEVELS 32


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define SHADERIO_MAX_NODE_CHILDREN 32


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define SHADERIO_MAX_GROUP_CLUSTERS 128


// 结构：BBox。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct BBox
{
  vec3 lo;
  vec3 hi;

  float shortestEdge;
  float longestEdge;
};


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE(BBox_in, BBox, readonly, 16);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(BBoxes_in, BBox, readonly, 16);


// 结构：Cluster。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct Cluster
{
  uint8_t triangleCountMinusOne;
  uint8_t vertexCountMinusOne;
  uint8_t lodLevel;
  uint8_t groupChildIndex;

  uint8_t  attributeBits;
  uint8_t  localMaterialID;
  uint16_t reserved;


  uint32_t vertices;
  uint32_t indices;

};


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE(Cluster_in, Cluster, , 16);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(Clusters_inout, Cluster, , 16);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_SIZE(Cluster_size, Cluster, 16);

#ifndef __cplusplus


// 函数：Cluster_getVertexPositions。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
vec3s_in Cluster_getVertexPositions(Cluster_in cluster)
{
  return vec3s_in(uint64_t(cluster) + cluster.d.vertices);
}


// 函数：Cluster_getVertexNormals。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
uint32s_in Cluster_getVertexNormals(Cluster_in cluster)
{
  return uint32s_in(uint64_t(cluster) + (cluster.d.vertices + 4 * 3 * (cluster.d.vertexCountMinusOne + 1)));
}


// 函数：Cluster_getVertexTexCoords。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
vec2s_in Cluster_getVertexTexCoords(Cluster_in cluster)
{

  uint32_t elems = (cluster.d.attributeBits & CLUSTER_ATTRIBUTE_VERTEX_NORMAL) == 0 ? 3 : 4;
  return vec2s_in(uint64_t(cluster) + (((cluster.d.vertices + 4 * elems * (cluster.d.vertexCountMinusOne + 1)) + 7) & ~7));
}


// 函数：Cluster_getTriangleIndices。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
uint8s_in Cluster_getTriangleIndices(Cluster_in cluster)
{
  return uint8s_in(uint64_t(cluster) + cluster.d.indices);
}


// 函数：Cluster_getTriangleMaterials。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
uint8s_in Cluster_getTriangleMaterials(Cluster_in cluster)
{
  return uint8s_in(uint64_t(cluster) + (cluster.d.indices + 3 * (cluster.d.triangleCountMinusOne + 1)));
}
#endif


// 结构：TraversalMetric。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct TraversalMetric
{


  float boundingSphereX;
  float boundingSphereY;
  float boundingSphereZ;
  float boundingSphereRadius;
  float maxQuadricError;
};


// 结构：Group。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct Group
{


  uint32_t residentID;
  uint32_t clusterResidentID;

  uint16_t lodLevel;
  uint16_t clusterCount;

  TraversalMetric traversalMetric;
};


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE(Group_in, Group, , 16);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(Groups_in, Group, , 16);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_SIZE(Group_size, Group, 32);

#ifndef __cplusplus


// 函数：Group_getGeneratingGroup。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
uint Group_getGeneratingGroup(Group_in group, uint clusterIndex)
{
  return uint32s_in(uint64_t(group) + uint32_t(Group_size + Cluster_size * group.d.clusterCount)).d[clusterIndex];
}


// 函数：Group_getClusterBBox。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
BBox Group_getClusterBBox(Group_in group, uint clusterIndex)
{
  return BBoxes_in(uint64_t(group)
                   + uint32_t(Group_size + Cluster_size * group.d.clusterCount + (((4 * group.d.clusterCount) + 15) & ~15)))
      .d[clusterIndex];
}


// 函数：Cluster_getGroup。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
Group_in Cluster_getGroup(Cluster_in cluster)
{
  return Group_in(uint64_t(cluster) - uint32_t(cluster.d.groupChildIndex * Cluster_size + Group_size));
}


// 函数：Cluster_getBBox。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
BBox Cluster_getBBox(Cluster_in cluster)
{
  return Group_getClusterBBox(Cluster_getGroup(cluster), cluster.d.groupChildIndex);
}
#endif

#ifdef __cplusplus


// 结构：NodeRange。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct NodeRange
{
  uint32_t isGroup : 1;
  uint32_t childOffset : 26;
  uint32_t childCountMinusOne : 5;
};


// 结构：GroupRange。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct GroupRange
{
  uint32_t isGroup : 1;
  uint32_t groupIndex : 23;
  uint32_t groupClusterCountMinusOne : 8;
};
#endif


// 结构：Node。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct Node
{
#ifdef __cplusplus
  union
  {
    NodeRange  nodeRange;
    GroupRange groupRange;
  };
#else
  uint32_t packed;

#define Node_packed_isGroup 0 : 1

#define Node_packed_nodeChildOffset 1 : 26
#define Node_packed_nodeChildCountMinusOne 27 : 5

#define Node_packed_groupIndex 1 : 23
#define Node_packed_groupClusterCountMinusOne 24 : 8

#endif

  TraversalMetric traversalMetric;
};


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(Nodes_in, Node, readonly, 8);


// 结构：LodLevel。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct LodLevel
{
  float    minBoundingSphereRadius;
  float    minMaxQuadricError;
  uint32_t groupOffset;
  uint32_t groupCount;
  uint32_t clusterOffset;
  uint32_t clusterCount;
};


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(LodLevels_inout, LodLevel, , 8);


// 结构：Geometry。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct Geometry
{
  uint32_t instancesOffset;
  uint32_t instancesCount;
  uint8_t  lodLevelsCount;


  uint16_t lowDetailTriangles;
  uint32_t lowDetailClusterID;
  uint64_t lowDetailBlasAddress;


  BBox bbox;

  BUFFER_REF(LodLevels_inout) lodLevels;


  BUFFER_REF(Nodes_in) nodes;
  BUFFER_REF(BBoxes_in) nodeBboxes;


  BUFFER_REF(uint64s_inout) streamingGroupAddresses;


  BUFFER_REF(uint64s_in) preloadedGroups;
  BUFFER_REF(uint64s_in) preloadedClusters;
  BUFFER_REF(uint64s_in) preloadedClusterClasAddresses;
  BUFFER_REF(uint32s_in) preloadedClusterClasSizes;
};


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE(Geometry_in, Geometry, readonly, 16);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE(Geometry_inout, Geometry, , 16);


// 结构：RenderInstance。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct RenderInstance
{
  mat4x3 worldMatrix;
  mat4x3 worldMatrixI;
  uint32_t geometryID;
  uint16_t materialID;
  uint8_t  flipWinding;
  uint8_t  twoSided;
  float    maxLodLevelRcp;
  uint32_t packedColor;
  vec4 _pad;
};


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(RenderInstances_in, RenderInstance, readonly, 16);

#ifdef __cplusplus
}
#endif
#endif
