////压缩和解压算法
#ifndef _ATTRIBUTE_ENCODING_H_
#define _ATTRIBUTE_ENCODING_H_
// 定义圆周率 PI，提供高精度的浮点常量
#define ATTRENC_PI           3.14159265358979323846f
// 定义法线压缩后占用的总位数（22位，X和Y各占11位）
#define ATTRENC_NORMAL_BITS  22
// 定义切线压缩后占用的总位数（10位，包含1位符号位和9位角度位）
#define ATTRENC_TANGENT_BITS 10
// --- 跨平台宏定义设定 (C++ / GLSL) ---
// 这里的宏定义是为了抹平 C++ (GLM库) 和 GLSL (着色器语言) 之间的语法差异，
// 使得下面的数学函数不仅可以在 CPU 端编译打包数据，也可以在 GPU 端编译解包数据。
#ifdef __cplusplus
// ==================== 如果是在 C++ 环境中编译 ====================
namespace shaderio {
#define ATTRENC_INLINE inline           // C++ 的内联函数关键字
#define ATTRENC_OUT(a) a&               // C++ 用引用作为输出参数 (类似于 GLSL 的 out)
#define ATTRENC_ATAN2F atan2f           // C++ 标准库的反正切函数
#define ATTRENC_FLOOR  glm::floor       // 使用 GLM 数学库的向下取整
#define ATTRENC_CLAMP  glm::clamp       // 使用 GLM 数学库的范围限制函数
#define ATTRENC_ABS    glm::abs         // 使用 GLM 数学库的绝对值函数
    // 编译期断言：法线的位数必须是偶数，因为要平分给 X 和 Y 两个通道
static_assert(ATTRENC_NORMAL_BITS % 2 == 0, "Normal bits must be an even number");
#else
// ==================== 如果是在 GLSL 着色器环境中编译 ====================
#define ATTRENC_INLINE                  // GLSL 不需要 inline 关键字
#define ATTRENC_OUT(a) out a            // GLSL 使用 out 关键字定义输出参数
#define ATTRENC_ATAN2F atan             // GLSL 内置的反正切函数
#define ATTRENC_FLOOR  floor            // GLSL 内置的向下取整
#define ATTRENC_CLAMP  clamp            // GLSL 内置的范围限制
#define ATTRENC_ABS    abs              // GLSL 内置的绝对值
#endif
// --- 八面体映射 (Octahedral Mapping) 辅助函数 ---
// 八面体映射能将 3D 单位球面上的一点（3维向量）完美映射到 2D 的正方形中。
// 辅助函数：返回二维向量各分量的符号（且不返回0，保证即使是0也当做正数处理）
ATTRENC_INLINE vec2 oct_signNotZero(vec2 v) {
    return vec2((v.x >= 0.0f) ? 1.0f : -1.0f, (v.y >= 0.0f) ? 1.0f : -1.0f);
}
// 恢复顶点法线：将 2D 八面体坐标还原回 3D 单位向量 
ATTRENC_INLINE vec3 oct_to_vec(vec2 e) {
    // 根据 2D 坐标计算初步的 3D 向量 (利用特性: |x| + |y| + |z| = 1)
    vec3 v = vec3(e.x, e.y, 1.0f - ATTRENC_ABS(e.x) - ATTRENC_ABS(e.y));
    // 如果 z 小于 0，说明这个点在下半球面，需要根据八面体展开的规则进行对角线翻折
    if (v.z < 0.0f) {
        vec2 os = oct_signNotZero(e);// 获取符号
        v.x = (1.0f - ATTRENC_ABS(e.y)) * os.x;// 交叉翻折
        v.y = (1.0f - ATTRENC_ABS(e.x)) * os.y;
    }
    return normalize(v);// 最后归一化，得到标准的 3D 法线
}
// 将 3D 法线向量压缩成 2D 八面体坐标
ATTRENC_INLINE vec2 vec_to_oct(vec3 v) {
    // 投影到 |x| + |y| + |z| = 1 的八面体面上
    vec2 p = vec2(v.x, v.y) * (1.0f / (ATTRENC_ABS(v.x) + ATTRENC_ABS(v.y) + ATTRENC_ABS(v.z)));
    // 如果在下半球面 (v.z <= 0)，需要映射到 2D 平面的四个角落区域
    return (v.z <= 0.0f) ? (vec2(1.0f - ATTRENC_ABS(p.y), 1.0f - ATTRENC_ABS(p.x)) * oct_signNotZero(p)) : p;
}
// 高精度映射：因为浮点数最终要转成有限的整数位数(bits)，这会产生量化误差。
// 这个函数会在量化后的点周围 2x2 的邻域内搜索，找到还原后和原始法线最接近（点乘余弦值最大）的值。
ATTRENC_INLINE vec2 vec_to_oct_precise(vec3 v, int bits) {
    vec2 s = vec_to_oct(v);// 先获取常规的 2D 坐标
    float M = float(1 << (bits - 1)) - 1.0f;// 计算给定位数下的最大整数值 (例如8位对应127)
    // 模拟量化截断：先放大 M 倍取整，再缩小回来，这是直接量化后的结果
    s = ATTRENC_FLOOR(ATTRENC_CLAMP(s, -1.0f, 1.0f) * M) * (1.0f / M);
    vec2  bestRepresentation = s;
    // 计算直接量化后的结果与原法线的相似度 (点乘)
    float highestCosine = dot(oct_to_vec(s), v);
    // 搜索周围 2x2 邻域寻找最佳精度匹配
    for (int i = 0; i <= 1; ++i) {
        for (int j = 0; j <= 1; ++j) {
            if (i != 0 || j != 0) {// 跳过自身(0,0)
                // 偏移一个小网格单位
                vec2  candidate = s + vec2(i, j) * (1.0f / M);
                float cosine = dot(oct_to_vec(candidate), v);
                // 如果偏移后的结果比之前更好，就记录下来
                if (cosine > highestCosine) {
                    bestRepresentation = candidate;
                    highestCosine = cosine;
                }
            }
        }
    }
    return bestRepresentation;
}
// --- 法线压缩与解压 ---
// 将 3D 法线打包成一个 32位无符号整数 (只使用其中 22 位)
ATTRENC_INLINE uint32_t normal_pack(vec3 normal) {
    const int      halfBits = ATTRENC_NORMAL_BITS / 2; // X和Y各占一半，即 11 位
    const uint32_t mask = (1 << halfBits) - 1;         // 11位的掩码：0x7FF (十进制2047)
        // 获取高精度八面体映射的 2D 坐标 [-1, 1]
    vec2 v = vec_to_oct_precise(normal, halfBits);
        // 将 [-1, 1] 映射到 [0, mask] 的整数区间
    v = (v + 1.0f) * 0.5f * float(mask) + 0.5f;
        // 把 X 放在低 11 位，Y 放在高 11 位，合并返回
    return (uint32_t(v.x) & mask) | ((uint32_t(v.y) & mask) << halfBits);
}
// 将 32位 整数解包还原回 3D 法线
ATTRENC_INLINE vec3 normal_unpack(uint32_t packed) {
    const int      halfBits = ATTRENC_NORMAL_BITS / 2;
    const uint32_t mask = (1 << halfBits) - 1;
    // 拆解出 X 和 Y 的整数值 (uvec2 是 uint组成的2维向量)
    uvec2 pv = uvec2(packed, (packed >> halfBits)) & uvec2(mask);
    // 将 [0, mask] 的整数映射回 [-1.0, 1.0] 的浮点数
    vec2  v = (vec2(pv) / float(mask)) * 2.0f - 1.0f;
    // 使用八面体映射还原函数转回 3D 向量
    return oct_to_vec(v);
}
// --- 切线空间正交基及切线压缩 ---
// 核心数学基础：基于已知的一个法线向量，自动生成互相垂直的切线(tangent)和副切线(bitangent)，构成局部坐标系。
ATTRENC_INLINE void tangent_orthonormalBasis(vec3 normal, ATTRENC_OUT(vec3) tangent, ATTRENC_OUT(vec3) bitangent) {
    // 处理法线朝向正下方的奇异点 (-Z)，防止数学公式出现除零错误
    if (normal.z < -0.99998796f) {
        tangent = vec3(0.0f, -1.0f, 0.0f);
        bitangent = vec3(-1.0f, 0.0f, 0.0f);
        return;
    }
    // 使用无奇异点的稳健正交基生成算法 (Duff et al. "Building an Orthonormal Basis, Revisited")
    float a = 1.0f / (1.0f + normal.z);
    float b = -normal.x * normal.y * a;
    tangent = vec3(1.0f - normal.x * normal.x * a, b, -normal.x);
    bitangent = vec3(b, 1.0f - normal.y * normal.y * a, -normal.y);
}
// 将 4D 切线向量压缩打包：因为副切线可以由法线和切线叉乘得出，
// 我们只需要知道真实切线在"自动生成的局部坐标系"中的旋转角度，就能压缩它。
ATTRENC_INLINE uint32_t tangent_pack(vec3 normal, vec4 tangent) {
    // 取 10 位中的 9 位存角度 (最大值 511)
    const uint32_t mask = (1 << (ATTRENC_TANGENT_BITS - 1)) - 1;
    vec3 autoTangent, autoBitangent;
    // 根据法线计算出自动正交基
    tangent_orthonormalBasis(normal, autoTangent, autoBitangent);
    // 计算真实切线和自动生成的切线之间的夹角。
    // 使用 atan2f 算出 [-PI, PI]，除以 PI 得到 [-1.0, 1.0] 的范围
    float angle = ATTRENC_ATAN2F(dot(autoTangent, vec3(tangent)), dot(autoBitangent, vec3(tangent))) / ATTRENC_PI;
    // 映射到 [0, 1] 范围
    float angleUnorm = ATTRENC_CLAMP((angle + 1.0f) * 0.5f, 0.0f, 1.0f);
    // 映射成 9 位的整数 (0 ~ 511)
    uint32_t angleBits = uint32_t(angleUnorm * float(mask) + 0.5f);
    // 修复：确保位运算的右侧明确是 uint 类型 (1u : 0u)
    return (angleBits << 1) | (tangent.w > 0.0f ? 1u : 0u);
}
// 从压缩数据中解压出 4D 切线
ATTRENC_INLINE vec4 tangent_unpack(vec3 normal, uint32_t encoded) {
    const uint32_t mask = (1 << (ATTRENC_TANGENT_BITS - 1)) - 1;
    // 最低 1 位是符号位
    uint32_t signBit = encoded & 1;
    // 高 9 位映射回 [0, 1] 的角度浮点数
    float    angleUnorm = float((encoded >> 1) & mask) / float(mask);
    // 还原回真实弧度角 [-PI, PI]
    float    angle = (angleUnorm * 2.0f - 1.0f) * ATTRENC_PI;
    // 再次通过法线生成自动的正交基
    vec3 autoTangent, autoBitangent;
    tangent_orthonormalBasis(normal, autoTangent, autoBitangent);
    // 利用角度，将自动基底旋转回原始的切线方向
    vec3 tangent = cos(angle) * autoBitangent + sin(angle) * autoTangent;
    // 还原 4D 切线的 w 分量 (用于计算副切线方向)
    return vec4(tangent, (signBit == 1) ? 1.0f : -1.0f);
}
// --- 清理内部作用域宏 ---
// 防止这些局部的数学宏污染了外部包含此头文件的其他代码
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
//#ifndef _ATTRIBUTE_ENCODING_H_
//#define _ATTRIBUTE_ENCODING_H_
//#define ATTRENC_PI float(3.14159265358979323846264338327950288)
//#define ATTRENC_NORMAL_BITS 22
//#define ATTRENC_TANGENT_BITS 10
//#ifdef __cplusplus
//namespace shaderio {
//#define ATTRENC_INLINE inline
//#define ATTRENC_OUT(a) a&
//#define ATTRENC_ATAN2F atan2f
//#define ATTRENC_INLINE inline
//#define ATTRENC_FLOOR glm::floor
//#define ATTRENC_CLAMP glm::clamp
//#define ATTRENC_ABS glm::abs
//static_assert(ATTRENC_NORMAL_BITS % 2 == 0);
//#else
//#define ATTRENC_INLINE
//#define ATTRENC_FLOOR floor
//#define ATTRENC_CLAMP clamp
//#define ATTRENC_ABS abs
//#define ATTRENC_INLINE
//#define ATTRENC_OUT(a) out a
//#define ATTRENC_ATAN2F atan
//#endif
//ATTRENC_INLINE vec2 oct_signNotZero(vec2 v)
//{
//  return vec2((v.x >= 0.0f) ? +1.0f : -1.0f, (v.y >= 0.0f) ? +1.0 : -1.0f);
//}
//ATTRENC_INLINE vec3 oct_to_vec(vec2 e)
//{
//  vec3 v = vec3(e.x, e.y, 1.0f - ATTRENC_ABS(e.x) - ATTRENC_ABS(e.y));
//  if(v.z < 0.0f)
//  {
//    vec2 os = oct_signNotZero(e);
//    v.x     = (1.0f - ATTRENC_ABS(e.y)) * os.x;
//    v.y     = (1.0f - ATTRENC_ABS(e.x)) * os.y;
//  }
//  return normalize(v);
//}
//
//ATTRENC_INLINE vec2 vec_to_oct(vec3 v)
//{
//  vec2 p = vec2(v.x, v.y) * (1.0f / (ATTRENC_ABS(v.x) + ATTRENC_ABS(v.y) + ATTRENC_ABS(v.z)));
//  return (v.z <= 0.0f) ? (vec2(1.0f - ATTRENC_ABS(p.y), 1.0f - ATTRENC_ABS(p.x)) * oct_signNotZero(p)) : p;
//}
//
//ATTRENC_INLINE vec2 vec_to_oct_precise(vec3 v, int bits)
//{
//  vec2 s = vec_to_oct(v);  
//  float M = float(1 << ((bits)-1)) - 1.0f;
//  s                        = ATTRENC_FLOOR(ATTRENC_CLAMP(s, -1.0f, +1.0f) * M) * (1.0f / M);
//  vec2  bestRepresentation = s;
//  float highestCosine      = dot(oct_to_vec(s), v);
//           for(int i = 0; i <= 1; ++i)
//  {
//    for(int j = 0; j <= 1; ++j)
//    {
//             if((i != 0) || (j != 0))
//      {
//                                   vec2  candidate = vec2(i, j) * (1 / M) + s;
//        float cosine    = dot(oct_to_vec(candidate), v);
//        if(cosine > highestCosine)
//        {
//          bestRepresentation = candidate;
//          highestCosine      = cosine;
//        }
//      }
//    }
//  }
//  return bestRepresentation;
//}
//
//ATTRENC_INLINE vec3 normal_unpack(uint32_t packed)
//{
//  const uint32_t mask = (1 << (ATTRENC_NORMAL_BITS / 2)) - 1;
//
//  uvec2 pv = uvec2(packed, (packed >> 11)) & uvec2(mask);
//  vec2  v  = (vec2(pv) / float(mask)) * 2.0f - 1.0f;
//
//  return oct_to_vec(v);
//}
//
//ATTRENC_INLINE uint32_t normal_pack(vec3 normal)
//{
//  vec2           v    = vec_to_oct_precise(normal, (ATTRENC_NORMAL_BITS / 2));
//  const uint32_t mask = (1 << (ATTRENC_NORMAL_BITS / 2)) - 1;
//
//  v = (v + 1.0f) * 0.5f * float(mask) + 0.5f;
//
//  uint32_t packed = uint32_t(v.x) & mask;
//  packed |= (uint32_t(v.y) & mask) << 11;
//
//  return packed;
//}
//
//  
//          ATTRENC_INLINE void tangent_orthonormalBasis(vec3 normal, ATTRENC_OUT(vec3) tangent, ATTRENC_OUT(vec3) bitangent)
//{
//  if(normal.z < -0.99998796F)     {
//    tangent   = vec3(0.0F, -1.0F, 0.0F);
//    bitangent = vec3(-1.0F, 0.0F, 0.0F);
//    return;
//  }
//  float a   = 1.0F / (1.0F + normal.z);
//  float b   = -normal.x * normal.y * a;
//  tangent   = vec3(1.0F - normal.x * normal.x * a, b, -normal.x);
//  bitangent = vec3(b, 1.0f - normal.y * normal.y * a, -normal.y);
//}
//
//ATTRENC_INLINE uint32_t tangent_pack(vec3 normal, vec4 tangent)
//{
//  const uint32_t mask = (1 << (ATTRENC_TANGENT_BITS - 1)) - 1;
//
//  vec3 autoTangent;
//  vec3 autoBitangent;
//
//  tangent_orthonormalBasis(normal, autoTangent, autoBitangent);
//
//  float angle = ATTRENC_ATAN2F(dot(autoTangent, vec3(tangent)), dot(autoBitangent, vec3(tangent))) / ATTRENC_PI;
//
//  float    angleUnorm = min(max((angle + 1.0f) * 0.5f, 0.0f), 1.0f);
//  uint32_t angleBits  = uint32_t(angleUnorm * float(mask) + 0.5f);
//  uint32_t encoded    = uint32_t((angleBits << 1) | ((tangent.w > 0.0f ? 1 : 0)));
//  return encoded;
//}
//
//ATTRENC_INLINE vec4 tangent_unpack(vec3 normal, uint32_t encoded)
//{
//  const uint32_t mask = (1 << (ATTRENC_TANGENT_BITS - 1)) - 1;
//
//  uint32_t signBit   = encoded & 1;
//  uint32_t angleBits = (encoded >> 1) & mask;
//
//  float angleUnorm = float(angleBits) / float(mask);
//
//  float angle = ((angleUnorm * 2.0f) - 1.0f) * ATTRENC_PI;
//
//  vec3 autoTangent;
//  vec3 autoBitangent;
//  tangent_orthonormalBasis(normal, autoTangent, autoBitangent);
//
//  vec3  tangent = cos(angle) * autoBitangent + sin(angle) * autoTangent;
//  float w       = signBit == 1 ? 1.0f : -1.0f;
//
//  return vec4(tangent, w);
//}
//
//#undef ATTRENC_ABS
//#undef ATTRENC_FLOOR
//#undef ATTRENC_CLAMP
//#undef ATTRENC_INLINE
//#undef ATTRENC_PI
//#undef ATTRENC_OUT
//
//#ifdef __cplusplus
//}
//#endif
//
//#endif