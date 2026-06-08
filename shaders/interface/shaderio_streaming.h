//==============================================================================
// 文件：shaders/interface/shaderio_streaming.h
// 模块定位：CPU 与 GPU 共享布局文件，定义着色器和 C++ 共同理解的数据结构、常量和访问约定。
// 数据流：CPU 侧填充这些结构，GPU 侧按完全相同的内存布局读取和写回。
// 方法说明：共享布局是异构系统的 ABI，任何字段顺序、对齐和位域变化都会影响两侧解释一致性。
// 正确性约束：结构对齐、标量布局和 缓冲 reference 类型必须与 Vulkan/GLSL 编译选项一致。
// 注释风格：使用中文解释 GPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
// 依赖说明：引入共享布局、剔除、着色或阶段间复用的着色器片段。
// 这些 include 共同决定本文件能访问的结构布局、数学辅助函数和编译期宏。
#include "shaderio_scene.h"
#ifndef _SHADERIO_STREAMING_H_


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define _SHADERIO_STREAMING_H_
#ifdef __cplusplus


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace shaderio {
using namespace glm;
#else


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAMING_DEBUG_USEDBITS_COUNT 1


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAMING_DEBUG_FREEGAPS_OVERLAP 0

#endif


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAMING_INVALID_ADDRESS_START (uint64_t(1) << 63)


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAMING_DEBUG_ADDRESSES 0


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAMING_DEBUG_ALWAYS_BUILD_FREEGAPS 0


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAMING_DEBUG_WITHOUT_RT 0


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAMING_DEBUG_MANUAL_MOVE 0


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAMING_ALLOCATOR_MIN_SIZE 32


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAMING_CACHED_BLAS_MAX_CLUSTERS 0xFFFF


// 结构：StreamingRequest。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct StreamingRequest
{
  uint maxLoads;
  uint maxUnloads;
  uint loadCounter;
  uint unloadCounter;
#ifdef __cplusplus
  union
  {
    uint64_t frameIndex;
    uint32_t frameIndexU32[2];
  };
#else
  uint64_t frameIndex;
#endif


  uint64_t clasCompactionUsedSize;
  uint     clasCompactionCount;

  uint     clasAllocatedMaxSizedLeft;
  uint64_t clasAllocatedUsedSize;
  uint64_t clasAllocatedWastedSize;

  uint taskIndex;
  uint errorUpdate;
  uint errorAgeFilter;
  uint errorClasNotFound;
  uint errorClasList;
  uint errorClasAlloc;
  uint errorClasDealloc;
  int  errorClasUsedVsAlloc;

  BUFFER_REF(uvec2s_inout) loadGeometryGroups;
  BUFFER_REF(uvec2s_inout) unloadGeometryGroups;
};


// 结构：ClasBuildInfo。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct ClasBuildInfo
{
  uint32_t clusterID;
  uint32_t clusterFlags;

#define ClasBuildInfo_packed_triangleCount 0 : 9
#define ClasBuildInfo_packed_vertexCount 9 : 9
#define ClasBuildInfo_packed_positionTruncateBitCount 18 : 6
#define ClasBuildInfo_packed_indexType 24 : 4
#define ClasBuildInfo_packed_opacityMicromapIndexType 28 : 4
  uint32_t packed;
#define ClasGeometryFlag_OPAQUE_BIT_NV (4 << 29)
  uint32_t baseGeometryIndexAndFlags;

  uint16_t indexBufferStride;
  uint16_t vertexBufferStride;
  uint16_t geometryIndexAndFlagsBufferStride;
  uint16_t opacityMicromapIndexBufferStride;
  uint64_t indexBuffer;
  uint64_t vertexBuffer;
  uint64_t geometryIndexAndFlagsBuffer;
  uint64_t opacityMicromapArray;
  uint64_t opacityMicromapIndexBuffer;
};


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(ClasBuildInfos_inout, ClasBuildInfo, , 16);


// 结构：StreamingPatch。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct StreamingPatch
{

  uint32_t geometryID;
  uint32_t groupID;
  uint64_t groupAddress;


  uint16_t clusterCount;
  uint16_t lodLevel;
  uint32_t groupResidentID;
  uint32_t clusterResidentID;
  uint32_t clasBuildOffset;
};


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(StreamingPatchs_in, StreamingPatch, , 16);


// 结构：StreamingGeometryPatch。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct StreamingGeometryPatch
{
  uint32_t geometryID;
  uint16_t cachedBlasClustersCount;
};


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(StreamingGeometryPatchs_in, StreamingGeometryPatch, , 16);


// 结构：StreamingUpdate。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct StreamingUpdate
{

  uint patchUnloadGroupsCount;

  uint patchGroupsCount;


  uint patchCachedBlasCount;
  uint patchCachedClustersCount;


  uint loadActiveGroupsOffset;
  uint loadActiveClustersOffset;

  uint taskIndex;
  uint frameIndex;


  BUFFER_REF(StreamingPatchs_in) patches;


  BUFFER_REF(ClasBuildInfos_inout) newClasBuilds;
  BUFFER_REF(uint32s_inout) newClasResidentIDs;
  BUFFER_REF(uint32s_inout) newClasSizes;
  BUFFER_REF(uint64s_inout) newClasAddresses;
  uint32_t newClasCount;


  BUFFER_REF(StreamingGeometryPatchs_in) geometryPatches;


  uint32_t moveClasCounter;
  uint64_t moveClasSize;
  BUFFER_REF(uint64s_inout) moveClasSrcAddresses;
  BUFFER_REF(uint64s_inout) moveClasDstAddresses;
};


// 结构：StreamingGroup。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct StreamingGroup
{
  uint32_t geometryID;
  uint16_t lodLevel;
  uint16_t age;
  BUFFER_REF(Group_in) group;
};


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(StreamingGroup_inout, StreamingGroup, , 16);


// 结构：StreamingResident。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct StreamingResident
{


  uint activeGroupsCount;
  uint activeClustersCount;

  BUFFER_REF(uint32s_in) activeGroups;


  BUFFER_REF(StreamingGroup_inout) groups;
  BUFFER_REF(uint32s_inout) groupIDs;


  BUFFER_REF(uvec2s_inout) groupClasSizes;

  BUFFER_REF(uint64s_inout) clusters;
  BUFFER_REF(uint64s_inout) clasAddresses;
  BUFFER_REF(uint32s_inout) clasSizes;

  uint64_t clasBaseAddress;
  uint64_t clasMaxSize;


  BUFFER_REF(uint64s_inout) clasCompactionUsedSize;

  BUFFER_REF(uint32s_inout) clasAllocatedMaxSizedLeft;


  uint taskIndex;
  uint frameIndex;
};


// 结构：AllocatorRange。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct AllocatorRange
{
  int32_t  count;
  uint32_t offset;
};


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(AllocatorRange_inout, AllocatorRange, , 8);


// 结构：AllocatorStats。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct AllocatorStats
{
  int64_t allocatedSize;
  int64_t wastedSize;
};


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE(AllocatorStats_inout, AllocatorStats, , 8);


// 结构：StreamingAllocator。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct StreamingAllocator
{
  uint freeGapsCounter;
  uint granularityByteShift;
  uint maxAllocationSize;
  uint sectorCount;
  uint sectorMaxAllocationSized;
  uint sectorSizeShift;
  uint baseWastedSize;
  uint usedBitsCount;

  DispatchIndirectCommand dispatchFreeGapsInsert;

  BUFFER_REF(uint32s_inout) freeGapsPos;
  BUFFER_REF(uint16s_inout) freeGapsSize;
  BUFFER_REF(uint32s_inout) freeGapsPosBinned;
  BUFFER_REF(AllocatorRange_inout) freeSizeRanges;
  BUFFER_REF(uint32s_inout) usedBits;
  BUFFER_REF(uint32s_inout) usedSectorBits;
  BUFFER_REF(AllocatorStats_inout) stats;
};


// 结构：SceneStreaming。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct SceneStreaming
{
  int32_t ageThreshold;
  uint    frameIndex;
  uint    useBlasCaching;
  uint    clasPositionTruncateBits;

  StreamingResident  resident;
  StreamingUpdate    update;
  StreamingRequest   request;
  StreamingAllocator clasAllocator;
};


#ifdef __cplusplus
}
#endif
#endif
