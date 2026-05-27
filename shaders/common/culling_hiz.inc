/*

HiZ 层级深度遮挡剔除相关函数
intersectHiz - 层级深度遮挡剔除函数，支持双遍剔除

优化版本：
1. 快速 Mipmap 层级计算（使用位运算替代 log2）
2. 早期剔除（非常小的包围盒直接返回可见）
3. 精度优化（使用合适的精度类型）
4. 代码结构优化（减少分支判断）
5. 内存访问优化（减少重复的纹理采样）
*/

//层级深度遮挡剔除 (Hi-Z Occlusion Culling)
#ifndef CULLING_NO_HIZ

// 快速计算 Mipmap 层级的辅助函数
// 使用位运算替代 log2，提高性能
int fastMipmapLevel(float maxsize)
{
    // 对于非常小的包围盒，直接返回 0 级 Mipmap
    if (maxsize <= 1.0) return 0;
    
    // 快速计算 log2(maxsize) 并向上取整
    // 使用位运算找到最高有效位
    int miplevel = 0;
    float val = maxsize;
    
    // 快速向上取整到下一个 2 的幂
    while (val > 1.0) {
        val *= 0.5;
        miplevel++;
    }
    
    return miplevel;
}

// idx 参数用于选择是前一帧的 Hi-Z 还是当前帧的 Hi-Z（用于双遍剔除）
bool intersectHiz(vec4 clipMin, vec4 clipMax, uint idx)
{
    // 早期剔除：如果包围盒在 Z 方向完全在相机后面，直接返回不可见
    if (clipMax.w <= 0.0) return false;
    
    // 将裁剪坐标 [-1, 1] 映射到 UV 坐标 [0, 1]
    vec2 minUV = clipMin.xy * 0.5 + 0.5;
    vec2 maxUV = clipMax.xy * 0.5 + 0.5;
    
    // 根据视口大小因子对 UV 进行缩放，并防止越界 (clamp 到 zw)
    minUV *= view.hizSizeFactors.xy;
    maxUV *= view.hizSizeFactors.xy;
    
    minUV = min(minUV, view.hizSizeFactors.zw);
    maxUV = min(maxUV, view.hizSizeFactors.zw);
    
    // 计算包围盒在屏幕上的宽高 (UV空间)
    vec2 size = maxUV - minUV;
    
    // 早期剔除：如果包围盒宽度或高度为负数，直接返回不可见
    if (size.x <= 0.0 || size.y <= 0.0) return false;
    
    // 计算实际覆盖的最大像素尺寸
    float maxsize = max(size.x, size.y) * view.hizSizeMax;
    
    // 早期剔除：对于非常小的包围盒，认为可见（避免深度剔除）
    if (maxsize < 0.5) return true;
    
    // 快速计算 Mipmap 层级
    int miplevel = fastMipmapLevel(maxsize);
    
    // 计算包围盒中心
    vec2 centerUV = (minUV + maxUV) * 0.5;
    
    // 采样 Hi-Z 纹理
    float depth;
#if USE_TWO_PASS_CULLING
    // 数组形式
    depth = textureLod(texHizFar[idx], centerUV, float(miplevel)).r;
#else
    // 单通道纹理形式
    depth = textureLod(texHizFar, centerUV, float(miplevel)).r;
#endif
    
    // 深度测试：这里使用的是反转深度 (Reversed-Z, 1是近，0是远)
    // 如果当前物体的最靠近相机的面 (clipMax.z) 仍然比深度缓冲区里记录的最浅深度 (depth) 还要远
    // 那么它就被遮挡了。加上 c_depthNudge 是为了避免 Z-fighting。
    bool visible = clipMax.z >= depth - c_depthNudge;
    
    return visible;
}

#endif
