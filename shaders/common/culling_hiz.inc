//==============================================================================
// 文件：shaders/common/culling_hiz.inc
// 模块定位：着色器公共函数片段，集中提供属性编码、剔除、屏幕空间估计和着色辅助逻辑。
// 数据流：多个计算、网格和片元阶段通过 include 复用这些函数，避免同一数学逻辑在不同阶段分叉。
// 方法说明：公共函数将几何、可见性和材质计算标准化，使 traversal 与 render 对同一对象得到一致判断。
// 正确性约束：公共函数不能依赖某个单独 阶段 的私有状态；所有宏开关都应有明确默认值。
// 注释风格：使用中文解释 GPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#ifndef CULLING_NO_HIZ


// 函数：fastMipmapLevel。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
int fastMipmapLevel(float maxsize)
{

    if (maxsize <= 1.0) return 0;


    int miplevel = 0;
    float val = maxsize;


    while (val > 1.0) {
        val *= 0.5;
        miplevel++;
    }

    return miplevel;
}


// 函数：intersectHiz。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
bool intersectHiz(vec4 clipMin, vec4 clipMax, uint idx)
{

    if (clipMax.w <= 0.0) return false;


    vec2 minUV = clipMin.xy * 0.5 + 0.5;
    vec2 maxUV = clipMax.xy * 0.5 + 0.5;


    minUV *= view.hizSizeFactors.xy;
    maxUV *= view.hizSizeFactors.xy;


    minUV = min(minUV, view.hizSizeFactors.zw);

    maxUV = min(maxUV, view.hizSizeFactors.zw);


    vec2 size = maxUV - minUV;


    if (size.x <= 0.0 || size.y <= 0.0) return false;


    float maxsize = max(size.x, size.y) * view.hizSizeMax;


    if (maxsize < 0.5) return true;


    int miplevel = fastMipmapLevel(maxsize);


    vec2 centerUV = (minUV + maxUV) * 0.5;


    float depth;
#if USE_TWO_PASS_CULLING

    depth = textureLod(texHizFar[idx], centerUV, float(miplevel)).r;
#else

    depth = textureLod(texHizFar, centerUV, float(miplevel)).r;
#endif


    bool visible = clipMax.z >= depth - c_depthNudge;

    return visible;
}

#endif
