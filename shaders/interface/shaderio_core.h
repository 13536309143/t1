//==============================================================================
// 文件：shaders/interface/shaderio_core.h
// 模块定位：CPU 与 GPU 共享布局文件，定义着色器和 C++ 共同理解的数据结构、常量和访问约定。
// 数据流：CPU 侧填充这些结构，GPU 侧按完全相同的内存布局读取和写回。
// 方法说明：共享布局是异构系统的 ABI，任何字段顺序、对齐和位域变化都会影响两侧解释一致性。
// 正确性约束：结构对齐、标量布局和 缓冲 reference 类型必须与 Vulkan/GLSL 编译选项一致。
// 注释风格：使用中文解释 GPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#ifndef _SHADERIO_CORE_H_


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define _SHADERIO_CORE_H_
#ifndef SUBGROUP_SIZE


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define SUBGROUP_SIZE 32
#endif
#ifndef USE_16BIT_DISPATCH


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define USE_16BIT_DISPATCH 0
#endif
#ifdef __cplusplus


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace shaderio {
using namespace glm;


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BUFFER_REF(refname) uint64_t


// 函数：adjustClusterProperty。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
static uint32_t inline adjustClusterProperty(uint32_t in)
{
  return (in + 31) & ~31;
}


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BUFFER_REF_DECLARE(refname, typ, keywords, alignment)                                                          \
  static_assert(alignof(typ) == alignment || (alignment > alignof(typ) && ((alignment % alignof(typ)) == 0)),          \
                "Alignment incompatible: " #refname)


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BUFFER_REF_DECLARE_ARRAY(refname, typ, keywords, alignment)                                                    \
  static_assert(alignof(typ) == alignment || (alignment > alignof(typ) && ((alignment % alignof(typ)) == 0)),          \
                "Alignment incompatible: " #refname)


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BUFFER_REF_DECLARE_SIZE(sizename, typ, size)                                                                   \
  static_assert(sizeof(typ) == size_t(size), "GLSL vs C++ size mismatch: " #typ)

#else


#if USE_16BIT_DISPATCH
#define getGlobalInvocationIndex getGlobalInvocationIndexLinearized
#define getWorkGroupIndex getWorkGroupIndexLinearized
#else
#define getGlobalInvocationIndex(globalInvocationID) (globalInvocationID.x)
#define getWorkGroupIndex(workGroupID) (workGroupID.x)
#endif

#define getGlobalInvocationIndexLinearized(globalInvocationID)                                                         \
  (globalInvocationID.x + (globalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x))
#define getWorkGroupIndexLinearized(workGroupID) (workGroupID.x + (workGroupID.y * gl_NumWorkGroups.x))


// 函数：murmurHash。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
uint murmurHash(uint idx)
{
  uint m = 0x5bd1e995;
  uint r = 24;

  uint h = 64684;
  uint k = idx;

  k *= m;
  k ^= (k >> r);
  k *= m;
  h *= m;
  h ^= k;

  return h;
}

#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_int64 : enable


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define PACKED_GET(flag, cfg)   (((flag) >> (true ? cfg)) & ((1 << (false ? cfg))-1))


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define PACKED_FLAG(cfg, val)   ((val) << (true ? cfg))


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define PACKED_MASK(cfg)        (((1 << (false ? cfg))-1) << (true ? cfg))


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BUFFER_REF(refname) refname


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BUFFER_REF_DECLARE(refname, typ, keywords, alignment)                                                          \
  layout(buffer_reference, buffer_reference_align = alignment, scalar) keywords buffer refname                         \
  {                                                                                                                    \
    typ d;                                                                                                             \
  };


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BUFFER_REF_DECLARE_ARRAY(refname, typ, keywords, alignment)                                                    \
  layout(buffer_reference, buffer_reference_align = alignment, scalar) keywords buffer refname                         \
  {                                                                                                                    \
    typ d[];                                                                                                           \
  };


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define BUFFER_REF_DECLARE_SIZE(sizename, typ, size) const uint32_t sizename = size


// 函数：toMat4。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
mat4 toMat4(mat4x3 m)
{
  return mat4(vec4(m[0], 0.0), vec4(m[1], 0.0), vec4(m[2], 0.0), vec4(m[3], 1.0));
}

#endif


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(uint8s_in, uint8_t, readonly, 1);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(uint8s_inout, uint8_t, , 1);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(uint16s_in, uint16_t, readonly, 2);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(uint16s_inout, uint16_t, , 2);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(uint32s_in, uint32_t, readonly, 4);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(uint32s_inout, uint32_t, , 4);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(int32s_inout, int32_t, , 4);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(uvec2s_in, uvec2, , 8);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(uvec2s_inout, uvec2, , 8);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(vec2s_in, vec2, , 8);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(vec2s_inout, vec2, , 8);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(uint64s_in, uint64_t, readonly, 8);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(uint64s_inout, uint64_t, , 8);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(uint64s_coh, uint64_t, coherent, 8);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(uint64s_coh_volatile, uint64_t, coherent volatile, 8);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(vec3s_in, vec3, readonly, 4);


// GPU 指针声明：为设备地址访问建立结构化缓冲引用类型。
// 该机制允许着色器通过 64 位地址访问 group、cluster、node 等运行时数据。
BUFFER_REF_DECLARE_ARRAY(vec4s_in, vec4, readonly, 16);


// 结构：DispatchIndirectCommand。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct DispatchIndirectCommand
{
  uint gridX;
  uint gridY;
  uint gridZ;
};


// 结构：DrawMeshTasksIndirectCommandNV。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct DrawMeshTasksIndirectCommandNV
{
  uint count;
  uint first;
};


// 结构：DrawMeshTasksIndirectCommandEXT。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct DrawMeshTasksIndirectCommandEXT
{
  uint gridX;
  uint gridY;
  uint gridZ;
};

#ifdef __cplusplus
static inline
#endif
    uvec3

    fit16bitLaunchGrid(uint count)
{


  if(count <= 0xFFFF)
    return uvec3(count, 1, 1);


#if 0
  uint side = uint(ceil(sqrt(float(count))));

  return uvec3(side, side, 1);
#else


  float countF = float(count);
  uint  n      = uint(ceil(uintBitsToFloat(floatBitsToUint(sqrt(countF)) + 1)));


  uint m = uint(sqrt(float(n * n - count)));

  return uvec3(n - m, n + m, 1);
#endif
}

#ifdef __cplusplus
}
#endif
#endif
