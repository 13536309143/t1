#version 460
// ------------------- 宏定义预设 -------------------
// 这些宏通常由 C++ 端在编译 Shader 时动态注入，如果没有注入则使用这里的默认值
#ifndef NV_HIZ_MAX_LEVELS
#define NV_HIZ_MAX_LEVELS   16 // Hi-Z 纹理支持的最大 Mipmap 层级数
#endif
#ifndef NV_HIZ_MSAA_SAMPLES
#define NV_HIZ_MSAA_SAMPLES 0  // MSAA 采样数，0 表示非多重采样
#endif
#ifndef NV_HIZ_IS_FIRST
#define NV_HIZ_IS_FIRST 1      // 标记这是否是生成 Hi-Z 的第一遍（通常第一遍从全尺寸 Depth 读取）
#endif
#ifndef NV_HIZ_FORMAT
#define NV_HIZ_FORMAT r32f     // 深度图格式，通常是 32 位浮点数
#endif
#ifndef NV_HIZ_OUTPUT_NEAR
#define NV_HIZ_OUTPUT_NEAR 1   // 是否同时输出 Near-Z（最小深度）。Far-Z 用于遮挡剔除，Near-Z 有时用于优化光照或其他计算
#endif
#ifndef NV_HIZ_LEVELS 
#define NV_HIZ_LEVELS 3        // 本次 Dispatch 计划在一个 Pass 中计算并生成的 Mipmap 层级数量（默认一口气生成3层）
#endif
#ifndef NV_HIZ_NEAR_LEVEL
#define NV_HIZ_NEAR_LEVEL 0    // 指定写入 Near-Z 的目标层级偏移
#endif
#ifndef NV_HIZ_FAR_LEVEL
#define NV_HIZ_FAR_LEVEL 0     // 指定写入 Far-Z 的目标层级偏移
#endif
#ifndef NV_HIZ_REVERSED_Z
#define NV_HIZ_REVERSED_Z 0    // 是否使用了反转深度（Reversed-Z, 即 1.0 是近平面，0.0 是远平面，极大提升深度精度）
#endif
#ifndef NV_HIZ_USE_STEREO 
#define NV_HIZ_USE_STEREO 0    // 是否用于 VR 双目立体渲染
#endif
// ------------------- 扩展启用 -------------------
// 如果需要一口气生成超过 1 级的 Mipmap，就必须启用 Subgroup 扩展
#if NV_HIZ_LEVELS > 1
  #extension GL_KHR_shader_subgroup_basic : require
  #extension GL_KHR_shader_subgroup_shuffle : require
#endif
// ------------------- 深度比较逻辑适配 -------------------
// 根据是否使用了反转深度 (Reversed-Z) 动态定义 minOp 和 maxOp
// 构建遮挡剔除用的 Far-Z 缓冲时，我们需要取 2x2 像素中最远的那个。对于标准Z是最远（最大），对于反转Z是最远（最小）
#if NV_HIZ_REVERSED_Z
  #define minOp max // 反转深度下，找最“近”的深度其实是找最大的值
  #define maxOp min // 反转深度下，找最“远”的深度其实是找最小的值
#else
  #define minOp min // 正常深度下的近
  #define maxOp max // 正常深度下的远
#endif
// ==========================================
// 工作组与资源绑定
// ==========================================
// 定义工作组大小：32 x 2 = 64 个线程。
// 在 NVIDIA 架构上，这正好等于 2 个 Warp（每个 Warp 32 线程）。
layout(local_size_x=32,local_size_y=2) in;
// Push Constants：轻量级的常量传递，用于告知当前 Pass 的范围和状态
layout(push_constant) uniform passUniforms {
  ivec4 srcSize;      // 源纹理的尺寸（xy: 宽/高, zw: 减一后的边界用于 clamp）
  int   writeLod;     // 当前写入目标的起始 Mipmap Level
  int   startLod;     // 读取源的 Mipmap Level
  int   layer;        // 目标纹理数组的层 (如果是 Stereo VR 渲染)
  int   _pad0;        // 内存对齐填充
  bvec4 levelActive;  // 标记哪些较低级别的 Mipmap 是需要/可以被生成的（防止在小分辨率时越界）
};
// 宏定义：处理立体渲染(Stereo)时的 Texture Array 访问差异
#if NV_HIZ_USE_STEREO
  #define samplerTypeMS sampler2DMSArray
  #define samplerType   sampler2DArray
  #define imageType     image2DArray
  #define IACCESS(v,l)  ivec3(v,l) // 3D 坐标访问
#else
  #define samplerTypeMS sampler2DMS
  #define samplerType   sampler2D
  #define imageType     image2D
  #define IACCESS(v,l)  v          // 2D 坐标访问
#endif

// 纹理绑定
#if NV_HIZ_IS_FIRST && NV_HIZ_MSAA_SAMPLES
  layout(binding=0) uniform samplerTypeMS texDepth; // 源深度（MSAA 版）
#else
  layout(binding=0) uniform samplerType   texDepth; // 源深度（普通版）
#endif
  layout(binding=1) uniform samplerType   texNear;  // 上一级的 Near 深度纹理
  layout(binding=2,NV_HIZ_FORMAT) uniform imageType imgNear; // 写入目标的 Near 深度纹理（UAV）
  layout(binding=3,NV_HIZ_FORMAT) uniform imageType imgLevels[NV_HIZ_MAX_LEVELS]; // 写入目标的 Mipmap 层级数组（UAV）

// ==========================================
// 主函数逻辑
// ==========================================
void main()
{
  // --- 线程空间映射魔法 ---
  // 这部分极其巧妙。64个线程被重排映射到一个 8x8 的像素块输出上。
  // base: 当前工作组对应的左上角基础像素坐标 (乘以8是因为 64个线程输出 8x8)
  ivec2 base = ivec2(gl_WorkGroupID.xy) * 8;
  // subset: 根据线程在一维/二维上的 ID，重算它在 8x8 块中的(x,y)相对偏移。
  // 这种特定的排列(Morton 码变种)是为了让相邻的 4 个线程正好处理相邻的 2x2 像素区域，
  // 这对后续的 Subgroup Shuffle 规约至关重要。
  ivec2 subset = ivec2(int(gl_LocalInvocationID.x) & 1, int(gl_LocalInvocationID.x) / 2);
  subset += gl_LocalInvocationID.x >= 16 ? ivec2(2,-8) : ivec2(0,0);
  subset += ivec2(gl_LocalInvocationID.y * 4,0);
  
#if NV_HIZ_LEVELS > 1
  uint laneID = gl_SubgroupInvocationID; // 获取当前线程在 32 线程 Warp 中的编号 (0~31)
#endif
  // outcoord 是当前线程负责写入的目标 Mipmap (Level 0) 的像素坐标
  ivec2 outcoord = base + subset;
  // coord 是该线程去读取上一级源图像的像素坐标 (乘以 2 因为是在做降采样，1个目标像素对应 2x2 源像素)
  ivec2 coord = outcoord * 2;
  // ==========================================
  // 第一阶段：从显存中读取深度，并执行第一次降采样 (生成 Level 0)
  // ==========================================
#if NV_HIZ_IS_FIRST && NV_HIZ_MSAA_SAMPLES
// 如果是开启了抗锯齿的第一遍，需要处理 MSAA 多个子采样点
  #if NV_HIZ_REVERSED_Z
  float zMin = 0;
  float zMax = 1;// 反转Z的初始值
  #else
  float zMin = 1;
  float zMax = 0;// 标准Z的初始值
  #endif
  // 向量化处理，减少循环开销
  for (int i = 0; i < NV_HIZ_MSAA_SAMPLES; i++){
  // 读取 2x2 块中每个像素的第 i 个采样点
    vec4 zRead = vec4(texelFetch(texDepth, IACCESS(min(coord + ivec2(0,0), srcSize.zw), layer), i).r,
                      texelFetch(texDepth, IACCESS(min(coord + ivec2(1,0), srcSize.zw), layer), i).r,
                      texelFetch(texDepth, IACCESS(min(coord + ivec2(0,1), srcSize.zw), layer), i).r,
                      texelFetch(texDepth, IACCESS(min(coord + ivec2(1,1), srcSize.zw), layer), i).r);
    // 对这 4 个值进行归约
    zMin = minOp(zMin, minOp(zRead.x, minOp(zRead.y, minOp(zRead.z, zRead.w))));
    zMax = maxOp(zMax, maxOp(zRead.x, maxOp(zRead.y, maxOp(zRead.z, zRead.w))));
  }
#else
// 非 MSAA 的标准处理
  #if NV_HIZ_IS_FIRST
    #define texRead texDepth// 第一遍读完整深度
  #else
    #define texRead texNear// 后续遍读上一遍生成的深度
  #endif

  // 初始化深度值变量
  #if NV_HIZ_REVERSED_Z
  float zMin = 0;
  float zMax = 1;// 反转Z的初始值
  #else
  float zMin = 1;
  float zMax = 0;// 标准Z的初始值
  #endif

  coord = min(coord, srcSize.zw);// 防止在边缘处越界访问
  // 利用 texelFetchOffset 高效读取 2x2 像素块到 zRead 的 xyzw 分量中
  // 合并内存访问，提高缓存命中率
  vec4 zRead;
  zRead.x = texelFetchOffset(texRead, IACCESS(coord, layer), startLod, ivec2(0,0)).r;
  zRead.y = texelFetchOffset(texRead, IACCESS(coord, layer), startLod, ivec2(1,0)).r;
  zRead.z = texelFetchOffset(texRead, IACCESS(coord, layer), startLod, ivec2(0,1)).r;
  zRead.w = texelFetchOffset(texRead, IACCESS(coord, layer), startLod, ivec2(1,1)).r;
  // 在当前线程的寄存器内部，对这 2x2 = 4 个深度值求最大值（Far-Z）和最小值（Near-Z）
  zMax = maxOp(zRead.x, maxOp(zRead.y, maxOp(zRead.z, zRead.w)));
  zMin = minOp(zRead.x, minOp(zRead.y, minOp(zRead.z, zRead.w)));
#endif
// --- 写入 Level 0 ---
  // 将规约得到的第一个结果直接写入目标 Mipmap
  //zMax = float(gl_ThreadInWarpNV) / 32.0;
#if !(NV_HIZ_IS_FIRST && NV_HIZ_FAR_LEVEL > 0)
  imageStore(imgLevels[writeLod + 0], IACCESS(outcoord,layer), vec4(zMax));
#endif
#if NV_HIZ_IS_FIRST && NV_HIZ_OUTPUT_NEAR && NV_HIZ_NEAR_LEVEL == 0
  imageStore(imgNear, IACCESS(outcoord,layer), vec4(zMin));
#endif
  // ==========================================
  // 第二阶段：Subgroup Shuffle 规约 (生成 Level 1)
  // ==========================================
#if NV_HIZ_LEVELS > 1
  // 此时，每个线程的 zMax 变量都持有一个 2x2 块的最大值。
  // 通过 subgroupShuffle，一个线程可以直接向相邻的 3 个线程索要它们的 zMax！
  // 结合之前的 subset 布局安排，laneID, laneID+1, laneID+2, laneID+3 这四个线程的数据
  // 恰好构成了下一级 Mipmap 的一个 2x2 块！ (等同于原图的 4x4)
  // 安全的子群 shuffle，确保不会越界
  vec4 zRead0 = vec4( zMax,
                      subgroupShuffle(zMax, min(laneID + 1, 31u)),
                      subgroupShuffle(zMax, min(laneID + 2, 31u)),
                      subgroupShuffle(zMax, min(laneID + 3, 31u)));
  

#if NV_HIZ_IS_FIRST && NV_HIZ_OUTPUT_NEAR && NV_HIZ_NEAR_LEVEL >= 1
  vec4 zRead1 = vec4( zMin,
                      subgroupShuffle(zMin, min(laneID + 1, 31u)),
                      subgroupShuffle(zMin, min(laneID + 2, 31u)),
                      subgroupShuffle(zMin, min(laneID + 3, 31u)));
#endif
// 因为 4 个线程互相共享了数据，算出的结果是一样的，所以我们只需要让每组的第 1 个线程 (laneID & 3 == 0) 去写入显存即可。
  if ((levelActive.y || levelActive.z) && (laneID & 3) == 0)
  {
    outcoord /= 2;// 目标坐标再次减半，准备写入更小的一级 Mipmap
    // 边界检查，确保坐标不越界
    if (outcoord.x >= 0 && outcoord.y >= 0) {
      zMax = maxOp(maxOp(maxOp(zRead0.x, zRead0.y),zRead0.z),zRead0.w);
      // --- 写入 Level 1 ---
    #if !(NV_HIZ_IS_FIRST && NV_HIZ_FAR_LEVEL > 1)
      imageStore(imgLevels[writeLod + 1], IACCESS(outcoord, layer), vec4(zMax));
    #endif
    #if NV_HIZ_IS_FIRST && NV_HIZ_OUTPUT_NEAR && NV_HIZ_NEAR_LEVEL >= 1
      zMin = minOp(minOp(minOp(zRead1.x, zRead1.y),zRead1.z),zRead1.w);
      #if NV_HIZ_NEAR_LEVEL == 1
      imageStore(imgNear, IACCESS(outcoord, layer), vec4(zMin));
      #endif
    #endif
      // ==========================================
      // 第三阶段：终极规约 (生成 Level 2)
      // ==========================================
    #if NV_HIZ_LEVELS > 2
      if (levelActive.z) {
        outcoord /= 2;// 坐标第三次减半
        // 边界检查，确保坐标不越界
        if (outcoord.x >= 0 && outcoord.y >= 0) {
          // 继续扩大搜索范围！
          // 现在少数几个活跃的线程（每组的头号线程）手中已经掌握了 4x4 块的极值。
          // 为了拼凑出 8x8 块（对应源图像 8x8），我们需要跨度更大的 Shuffle：跨 4，跨 16，跨 20。
          // 这些数字不是随便写的，它是基于一开始那个精密的 subset 座标映射公式推导出的邻域线程间隔。
          // 安全的子群 shuffle，确保不会越界
          zRead0 = vec4(  zMax,
                          subgroupShuffle(zMax, min(laneID + 4, 31u)),
                          subgroupShuffle(zMax, min(laneID + 16, 31u)),
                          subgroupShuffle(zMax, min(laneID + 20, 31u)));
        #if NV_HIZ_IS_FIRST && NV_HIZ_OUTPUT_NEAR && NV_HIZ_NEAR_LEVEL == 2
          zRead1 = vec4(  zMin,
                          subgroupShuffle(zMin, min(laneID + 4, 31u)),
                          subgroupShuffle(zMin, min(laneID + 16, 31u)),
                          subgroupShuffle(zMin, min(laneID + 20, 31u)));
        #endif
          // 在这 64 个线程（2 个 Warp）中，最终只有 laneID 0 和 laneID 8 的线程凑齐了完整的数据。
          // 由这两个“天选之子”完成最后一次 min/max 规约并写入显存。
          if ((laneID == 0) || (laneID == 8)) {
            zMax = maxOp(maxOp(maxOp(zRead0.x, zRead0.y),zRead0.z),zRead0.w);
            // --- 写入 Level 2 ---
            imageStore(imgLevels[writeLod + 2], IACCESS(outcoord, layer), vec4(zMax));
          #if NV_HIZ_IS_FIRST && NV_HIZ_OUTPUT_NEAR && NV_HIZ_NEAR_LEVEL == 2
            zMin = minOp(minOp(minOp(zRead1.x, zRead1.y),zRead1.z),zRead1.w);
            imageStore(imgNear, IACCESS(outcoord, layer), vec4(zMin));
          #endif
          }
        }
      }
    #endif
    }
  }
#endif
}
