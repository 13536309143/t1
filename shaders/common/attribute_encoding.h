//==============================================================================
// 文件：shaders/common/attribute_encoding.h
// 模块定位：着色器公共函数片段，集中提供属性编码、剔除、屏幕空间估计和着色辅助逻辑。
// 数据流：多个计算、网格和片元阶段通过 include 复用这些函数，避免同一数学逻辑在不同阶段分叉。
// 方法说明：公共函数将几何、可见性和材质计算标准化，使 traversal 与 render 对同一对象得到一致判断。
// 正确性约束：公共函数不能依赖某个单独 阶段 的私有状态；所有宏开关都应有明确默认值。
// 注释风格：使用中文解释 GPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#ifndef _ATTRIBUTE_ENCODING_H_


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define _ATTRIBUTE_ENCODING_H_


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ATTRENC_PI           3.14159265358979323846f


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ATTRENC_NORMAL_BITS  22


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ATTRENC_TANGENT_BITS 10


#ifdef __cplusplus


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace shaderio {


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ATTRENC_INLINE inline


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ATTRENC_OUT(a) a&


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ATTRENC_ATAN2F atan2f


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ATTRENC_FLOOR  glm::floor


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ATTRENC_CLAMP  glm::clamp


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ATTRENC_ABS    glm::abs


static_assert(ATTRENC_NORMAL_BITS % 2 == 0, "Normal bits must be an even number");
#else


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ATTRENC_INLINE


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ATTRENC_OUT(a) out a


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ATTRENC_ATAN2F atan


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ATTRENC_FLOOR  floor


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ATTRENC_CLAMP  clamp


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define ATTRENC_ABS    abs
#endif


// 函数：oct_signNotZero。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
ATTRENC_INLINE vec2 oct_signNotZero(vec2 v) {
    return vec2((v.x >= 0.0f) ? 1.0f : -1.0f, (v.y >= 0.0f) ? 1.0f : -1.0f);
}


// 函数：oct_to_vec。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
ATTRENC_INLINE vec3 oct_to_vec(vec2 e) {

    vec3 v = vec3(e.x, e.y, 1.0f - ATTRENC_ABS(e.x) - ATTRENC_ABS(e.y));

    if (v.z < 0.0f) {

        vec2 os = oct_signNotZero(e);
        v.x = (1.0f - ATTRENC_ABS(e.y)) * os.x;
        v.y = (1.0f - ATTRENC_ABS(e.x)) * os.y;
    }
    return normalize(v);
}


// 函数：vec_to_oct。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
ATTRENC_INLINE vec2 vec_to_oct(vec3 v) {

    vec2 p = vec2(v.x, v.y) * (1.0f / (ATTRENC_ABS(v.x) + ATTRENC_ABS(v.y) + ATTRENC_ABS(v.z)));

    return (v.z <= 0.0f) ? (vec2(1.0f - ATTRENC_ABS(p.y), 1.0f - ATTRENC_ABS(p.x)) * oct_signNotZero(p)) : p;
}


// 函数：vec_to_oct_precise。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
ATTRENC_INLINE vec2 vec_to_oct_precise(vec3 v, int bits) {

    vec2 s = vec_to_oct(v);
    float M = float(1 << (bits - 1)) - 1.0f;

    s = ATTRENC_FLOOR(ATTRENC_CLAMP(s, -1.0f, 1.0f) * M) * (1.0f / M);
    vec2  bestRepresentation = s;

    float highestCosine = dot(oct_to_vec(s), v);

    for (int i = 0; i <= 1; ++i) {
        for (int j = 0; j <= 1; ++j) {
            if (i != 0 || j != 0) {

                vec2  candidate = s + vec2(i, j) * (1.0f / M);
                float cosine = dot(oct_to_vec(candidate), v);

                if (cosine > highestCosine) {
                    bestRepresentation = candidate;
                    highestCosine = cosine;
                }
            }
        }
    }
    return bestRepresentation;
}


// 函数：normal_pack。在紧凑编码和逻辑结构之间转换，减少带宽或便于着色器访问。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：编码位宽、符号位和特殊值必须与写入端/读取端完全一致，否则会产生难以定位的跨阶段错误。
ATTRENC_INLINE uint32_t normal_pack(vec3 normal) {
    const int      halfBits = ATTRENC_NORMAL_BITS / 2;
    const uint32_t mask = (1 << halfBits) - 1;


    vec2 v = vec_to_oct_precise(normal, halfBits);

    v = (v + 1.0f) * 0.5f * float(mask) + 0.5f;

    return (uint32_t(v.x) & mask) | ((uint32_t(v.y) & mask) << halfBits);
}


// 函数：normal_unpack。在紧凑编码和逻辑结构之间转换，减少带宽或便于着色器访问。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：编码位宽、符号位和特殊值必须与写入端/读取端完全一致，否则会产生难以定位的跨阶段错误。
ATTRENC_INLINE vec3 normal_unpack(uint32_t packed) {
    const int      halfBits = ATTRENC_NORMAL_BITS / 2;
    const uint32_t mask = (1 << halfBits) - 1;


    uvec2 pv = uvec2(packed, (packed >> halfBits)) & uvec2(mask);

    vec2  v = (vec2(pv) / float(mask)) * 2.0f - 1.0f;

    return oct_to_vec(v);
}


// 函数：tangent_orthonormalBasis。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
ATTRENC_INLINE void tangent_orthonormalBasis(vec3 normal, ATTRENC_OUT(vec3) tangent, ATTRENC_OUT(vec3) bitangent) {

    if (normal.z < -0.99998796f) {

        tangent = vec3(0.0f, -1.0f, 0.0f);

        bitangent = vec3(-1.0f, 0.0f, 0.0f);
        return;
    }

    float a = 1.0f / (1.0f + normal.z);
    float b = -normal.x * normal.y * a;

    tangent = vec3(1.0f - normal.x * normal.x * a, b, -normal.x);

    bitangent = vec3(b, 1.0f - normal.y * normal.y * a, -normal.y);
}


// 函数：tangent_pack。在紧凑编码和逻辑结构之间转换，减少带宽或便于着色器访问。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：编码位宽、符号位和特殊值必须与写入端/读取端完全一致，否则会产生难以定位的跨阶段错误。
ATTRENC_INLINE uint32_t tangent_pack(vec3 normal, vec4 tangent) {

    const uint32_t mask = (1 << (ATTRENC_TANGENT_BITS - 1)) - 1;
    vec3 autoTangent, autoBitangent;


    tangent_orthonormalBasis(normal, autoTangent, autoBitangent);


    float angle = ATTRENC_ATAN2F(dot(autoTangent, vec3(tangent)), dot(autoBitangent, vec3(tangent))) / ATTRENC_PI;

    float angleUnorm = ATTRENC_CLAMP((angle + 1.0f) * 0.5f, 0.0f, 1.0f);

    uint32_t angleBits = uint32_t(angleUnorm * float(mask) + 0.5f);

    return (angleBits << 1) | (tangent.w > 0.0f ? 1u : 0u);
}


// 函数：tangent_unpack。在紧凑编码和逻辑结构之间转换，减少带宽或便于着色器访问。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：编码位宽、符号位和特殊值必须与写入端/读取端完全一致，否则会产生难以定位的跨阶段错误。
ATTRENC_INLINE vec4 tangent_unpack(vec3 normal, uint32_t encoded) {
    const uint32_t mask = (1 << (ATTRENC_TANGENT_BITS - 1)) - 1;

    uint32_t signBit = encoded & 1;


    float    angleUnorm = float((encoded >> 1) & mask) / float(mask);

    float    angle = (angleUnorm * 2.0f - 1.0f) * ATTRENC_PI;

    vec3 autoTangent, autoBitangent;

    tangent_orthonormalBasis(normal, autoTangent, autoBitangent);

    vec3 tangent = cos(angle) * autoBitangent + sin(angle) * autoTangent;

    return vec4(tangent, (signBit == 1) ? 1.0f : -1.0f);
}


#undef ATTRENC_ABS
#undef ATTRENC_FLOOR
#undef ATTRENC_CLAMP
#undef ATTRENC_INLINE
#undef ATTRENC_ATAN2F
#undef ATTRENC_PI
#undef ATTRENC_OUT
#ifdef __cplusplus
}
#endif
#endif
