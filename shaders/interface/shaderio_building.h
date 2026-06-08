//==============================================================================
// 文件：shaders/interface/shaderio_building.h
// 模块定位：CPU 与 GPU 共享布局文件，定义着色器和 C++ 共同理解的数据结构、常量和访问约定。
// 数据流：CPU 侧填充这些结构，GPU 侧按完全相同的内存布局读取和写回。
// 方法说明：共享布局是异构系统的 ABI，任何字段顺序、对齐和位域变化都会影响两侧解释一致性。
// 正确性约束：结构对齐、标量布局和 缓冲 reference 类型必须与 Vulkan/GLSL 编译选项一致。
// 注释风格：使用中文解释 GPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
// 依赖说明：引入共享布局、剔除、着色或阶段间复用的着色器片段。
// 这些 include 共同决定本文件能访问的结构布局、数学辅助函数和编译期宏。
#include "shaderio_streaming.h"
#ifndef _SHADERIO_BUILDING_H_


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define _SHADERIO_BUILDING_H_


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define TRAVERSAL_INVALID_LOD_LEVEL 0xFF
#ifdef __cplusplus


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace shaderio {
using namespace glm;
#else


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define INSTANCE_VISIBLE_BIT 1


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define INSTANCE_USES_MERGED_BIT 2


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BLAS_BUILD_INDEX_LOWDETAIL (uint(~0))


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BLAS_BUILD_INDEX_SHARE_BIT (uint(1 << 31))


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BLAS_BUILD_INDEX_CACHE_BIT (uint(1 << 30))


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define TRAVERSAL_ALLOW_LOW_DETAIL_BLAS true

#endif


// 结构：TraversalInfo。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct TraversalInfo
{
  uint32_t instanceID;
  uint32_t packedNode;
};


// 结构：ClusterInfo。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct ClusterInfo
{
  uint32_t instanceID;
  uint32_t clusterID;
};


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(ClusterInfos_inout, ClusterInfo, , 8);


// 结构：BlasBuildInfo。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct BlasBuildInfo
{

  uint32_t clusterReferencesCount;

  uint32_t clusterReferencesStride;

  uint64_t clusterReferences;
};


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(BlasBuildInfo_inout, BlasBuildInfo, , 16);


// 结构：TlasInstance。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct TlasInstance
{
  mat3x4   worldMatrix;
  uint32_t instanceCustomIndex24_mask8;
  uint32_t instanceShaderBindingTableRecordOffset24_flags8;
  uint64_t blasReference;
};


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(TlasInstances_inout, TlasInstance, , 16);


// 结构：GeometryBuildHistogram。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct GeometryBuildHistogram
{

  uint32_t lodLevelMinHistogram[SHADERIO_MAX_LOD_LEVELS];

  uint32_t lodLevelMaxHistogram[SHADERIO_MAX_LOD_LEVELS];


  uint32_t lodLevelMaxPackedInstance[SHADERIO_MAX_LOD_LEVELS];
};


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(GeometryBuildHistogram_inout, GeometryBuildHistogram, , 16);


// 结构：GeometryBuildInfo。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct GeometryBuildInfo
{

  uint32_t cachedBuildIndex;
  uint16_t cachedLevel;

  uint8_t  shareLevelMin;
  uint8_t  shareLevelMax;
  uint32_t shareInstanceID;

  uint32_t mergedInstanceID;
};


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(GeometryBuildInfos_inout, GeometryBuildInfo, , 16);


// 结构：InstanceBuildInfo。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct InstanceBuildInfo
{

  uint32_t clusterReferencesCount;


  uint32_t blasBuildIndex;

  uint8_t  lodLevelMin;
  uint8_t  lodLevelMax;
  uint16_t geometryLodLevelMax;
  uint32_t geometryID;
};


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(InstanceBuildInfos_inout, InstanceBuildInfo, , 16);


// 结构：SceneBuilding。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct SceneBuilding
{
  mat4 traversalViewMatrix;
  mat4 cullViewProjMatrix;
  mat4 cullViewProjMatrixLast;


  uint pass;
  uint frameIndex;

  uint numGeometries;
  uint numRenderInstances;
  uint maxRenderClusters;
  uint maxTraversalInfos;

  float errorOverDistanceThreshold;
  float culledErrorScale;
  float swRasterThreshold;
  float swRasterTriangleDensityThreshold;

  uint sharingEnabledLevels;
  uint sharingTolerantLevels;
  uint sharingPushCulled;

  uint renderClusterCounter;
  uint renderClusterCounterSW;

  int  traversalTaskCounter;
  uint traversalInfoReadCounter;
  uint traversalInfoWriteCounter;

  uint traversalGroupCounter;


  BUFFER_REF(uint64s_coh_volatile) traversalNodeInfos;


  BUFFER_REF(uint64s_coh_volatile) traversalGroupInfos;


  BUFFER_REF(ClusterInfos_inout) renderClusterInfos;


  DispatchIndirectCommand indirectDispatchGroups;


  DrawMeshTasksIndirectCommandNV indirectDrawClustersNV;
  DrawMeshTasksIndirectCommandNV indirectDrawClusterBoxesNV;


  DrawMeshTasksIndirectCommandEXT indirectDrawClustersEXT;
  DrawMeshTasksIndirectCommandEXT indirectDrawClusterBoxesEXT;
  uint                            numRenderedClusters;


  DispatchIndirectCommand indirectDrawClustersSW;
  uint                    numRenderedClustersSW;


  BUFFER_REF(ClusterInfos_inout) renderClusterInfosSW;


  DispatchIndirectCommand indirectDispatchBlasInsertion;


  uint blasClasCounter;


  uint blasBuildCounter;


  BUFFER_REF(uint8s_inout) instanceVisibility;


  BUFFER_REF(uint32s_inout) instanceSortValues;
  BUFFER_REF(uint32s_inout) instanceSortKeys;


  BUFFER_REF(InstanceBuildInfos_inout) instanceBuildInfos;
  BUFFER_REF(TlasInstances_inout) tlasInstances;


  BUFFER_REF(BlasBuildInfo_inout) blasBuildInfos;


  BUFFER_REF(uint32s_inout) blasBuildSizes;
  BUFFER_REF(uint64s_inout) blasBuildAddresses;


  BUFFER_REF(uint64s_inout) blasClusterAddresses;


  BUFFER_REF(GeometryBuildInfos_inout) geometryBuildInfos;
  BUFFER_REF(GeometryBuildHistogram_inout) geometryHistograms;

  uint32_t cachedBlasCopyCounter;
};


#ifdef __cplusplus
}
#endif
#endif
