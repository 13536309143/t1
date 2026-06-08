//==============================================================================
// 文件：shaders/interface/shaderio.h
// 模块定位：CPU 与 GPU 共享布局文件，定义着色器和 C++ 共同理解的数据结构、常量和访问约定。
// 数据流：CPU 侧填充这些结构，GPU 侧按完全相同的内存布局读取和写回。
// 方法说明：共享布局是异构系统的 ABI，任何字段顺序、对齐和位域变化都会影响两侧解释一致性。
// 正确性约束：结构对齐、标量布局和 缓冲 reference 类型必须与 Vulkan/GLSL 编译选项一致。
// 注释风格：使用中文解释 GPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#ifndef _SHADERIO_H_


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define _SHADERIO_H_


// 依赖说明：引入共享布局、剔除、着色或阶段间复用的着色器片段。
// 这些 include 共同决定本文件能访问的结构布局、数学辅助函数和编译期宏。
#include "shaderio_core.h"
#include "shaderio_scene.h"
#include "shaderio_streaming.h"
#include "shaderio_building.h"
#include "nvshaders/sky_io.h.slang"


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define VISUALIZE_MATERIAL 0


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define VISUALIZE_GREY 1


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define VISUALIZE_VIS_BUFFER 2


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define VISUALIZE_CLUSTER 3


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define VISUALIZE_GROUP 4


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define VISUALIZE_LOD 5


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define VISUALIZE_TRIANGLE 6


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define VISUALIZE_DEPTH_ONLY 7


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define MESHSHADER_BBOX_VERTICES 8


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define MESHSHADER_BBOX_LINES 12


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define MESHSHADER_BBOX_THREADS 4


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BINDINGS_FRAME_UBO 0


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BINDINGS_READBACK_SSBO 1


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BINDINGS_GEOMETRIES_SSBO 2


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BINDINGS_RENDERINSTANCES_SSBO 3


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BINDINGS_SCENEBUILDING_SSBO 4


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BINDINGS_SCENEBUILDING_UBO 5


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BINDINGS_HIZ_TEX 6


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BINDINGS_STREAMING_UBO 7


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BINDINGS_STREAMING_SSBO 8


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BINDINGS_RASTER_ATOMIC 9


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BINDINGS_RENDER_TARGET 10


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BUILD_SETUP_TRAVERSAL_RUN 1


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BUILD_SETUP_DRAW 2


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BUILD_SETUP_BLAS_INSERTION 3


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAM_SETUP_COMPACTION_OLD_NO_UNLOADS 0


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAM_SETUP_COMPACTION_STATUS 1


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAM_SETUP_ALLOCATOR_FREEINSERT 2


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAM_SETUP_ALLOCATOR_STATUS 3


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define TRAVERSAL_PRESORT_WORKGROUP 128


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define TRAVERSAL_INIT_WORKGROUP 64


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define TRAVERSAL_RUN_WORKGROUP 64


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define TRAVERSAL_GROUPS_WORKGROUP 64


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define TRAVERSAL_BLAS_MERGING_WORKGROUP 64


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BLAS_SETUP_INSERTION_WORKGROUP 64


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BLAS_INSERT_CLUSTERS_WORKGROUP 64


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define INSTANCES_ASSIGN_BLAS_WORKGROUP 64


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define INSTANCES_CLASSIFY_LOD_WORKGROUP 64


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BLAS_CACHING_SETUP_BUILD_WORKGROUP 64


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BLAS_CACHING_SETUP_COPY_WORKGROUP 64


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAM_UPDATE_SCENE_WORKGROUP 64


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAM_AGEFILTER_GROUPS_WORKGROUP 96


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAM_COMPACTION_NEW_CLAS_WORKGROUP 96


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAM_COMPACTION_OLD_CLAS_WORKGROUP 64


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAM_ALLOCATOR_LOAD_GROUPS_WORKGROUP 64


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAM_ALLOCATOR_UNLOAD_GROUPS_WORKGROUP 64


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAM_ALLOCATOR_BUILD_FREEGAPS_WORKGROUP 64


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAM_ALLOCATOR_FREEGAPS_INSERT_WORKGROUP 64


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define STREAM_ALLOCATOR_SETUP_INSERTION_WORKGROUP 64


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define FORCE_INVISIBLE_CULLED_REMOVES_INSTANCE 1

#ifdef __cplusplus


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace shaderio {
using namespace glm;

#else

#ifndef ALLOW_SHADING


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ALLOW_SHADING 1
#endif

#ifndef ALLOW_VERTEX_NORMALS


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ALLOW_VERTEX_NORMALS 1
#endif

#ifndef ALLOW_VERTEX_TEXCOORDS


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ALLOW_VERTEX_TEXCOORDS 1
#endif

#ifndef ALLOW_VERTEX_TEXCOORD_1


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ALLOW_VERTEX_TEXCOORD_1 1
#endif

#ifndef ALLOW_VERTEX_TANGENTS


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ALLOW_VERTEX_TANGENTS 1
#endif

#ifndef USE_SW_RASTER


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define USE_SW_RASTER 0
#endif

#ifndef USE_RENDER_STATS


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define USE_RENDER_STATS 1
#endif

#ifndef USE_MEMORY_STATS


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define USE_MEMORY_STATS 1
#endif

#ifndef USE_CULLING


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define USE_CULLING 1
#endif

#ifndef USE_FORCED_INVISIBLE_CULLING


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define USE_FORCED_INVISIBLE_CULLING 1
#endif

#ifndef USE_TWO_PASS_CULLING


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define USE_TWO_PASS_CULLING 1
#endif


#ifndef USE_PRIMITIVE_CULLING


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define USE_PRIMITIVE_CULLING 1
#endif

#ifndef USE_INSTANCE_SORTING


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define USE_INSTANCE_SORTING 1
#endif


#ifndef USE_STREAMING


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define USE_STREAMING 0
#endif

#ifndef USE_TWO_SIDED


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define USE_TWO_SIDED 1
#endif

#ifndef USE_FORCED_TWO_SIDED


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define USE_FORCED_TWO_SIDED 0
#endif

#ifndef MAX_VISIBLE_CLUSTERS


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define MAX_VISIBLE_CLUSTERS 1024
#endif

#ifndef TARGETS_RASTERIZATION


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define TARGETS_RASTERIZATION 1
#endif


#endif


// 结构：FrameConstants。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct FrameConstants
{
  mat4 projMatrix;
  mat4 projMatrixI;

  mat4 viewProjMatrix;
  mat4 viewProjMatrixI;
  mat4 viewMatrix;
  mat4 viewMatrixI;
  vec4 viewPos;
  vec4 viewDir;
  vec4 viewPlane;

  mat4 skyProjMatrixI;


  mat4 viewProjMatrixPrev;

  ivec2 viewport;
  vec2  viewportf;

  vec2 viewPixelSize;
  vec2 viewClipSize;

  vec3  wLightPos;
  float lightMixer;

  vec3  wUpDir;
  float sceneSize;


  uint  colorXor;
  uint  visualize;
  float fov;

  float   nearPlane;
  float   farPlane;
  float   ambientOcclusionRadius;
  int32_t ambientOcclusionSamples;

  vec4 hizSizeFactors;
  vec4 nearSizeFactors;

  float hizSizeMax;
  int   facetShading;
  vec2  jitter;

  uint  dbgUint;
  float dbgFloat;
  uint  frame;
  uint  doShadow;

  vec4 bgColor;

  uvec2 mousePosition;
  float wireThickness;
  float wireSmoothing;

  vec3 wireColor;
  uint wireStipple;

  vec3  wireBackfaceColor;
  float wireStippleRepeats;

  float wireStippleLength;
  uint  doWireframe;
  uint  visFilterInstanceID;
  uint  visFilterClusterID;


  float time;
  float deltaTime;
  float lodTransitionSpeed;

  SkySimpleParameters skyParams;
};


// 结构：Readback。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct Readback
{
  uint     numRenderClusters;

  uint     numRenderClustersSW;

  uint     numTraversalTasks;

  uint     numTraversedTasks;
  uint     numBlasBuilds;
  uint     numRenderedClusters;
  uint     numRenderedClustersSW;
  uint64_t numRenderedTriangles;
  uint64_t numRenderedTrianglesSW;
  uint64_t numRasteredTriangles;
  uint64_t numRasteredTrianglesSW;


#ifdef __cplusplus
  uint32_t clusterTriangleId;
  uint32_t _packedDepth0;

  uint32_t instanceId;
  uint32_t _packedDepth1;
#else
  uint64_t clusterTriangleId;
  uint64_t instanceId;
#endif

  uint64_t debugU64;

  int  debugI;
  uint debugUI;
  uint debugF;

  uint debugA[64];
  uint debugB[64];
  uint debugC[64];
};


#ifdef __cplusplus
}
#endif
#endif
