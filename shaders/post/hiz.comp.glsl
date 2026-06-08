//==============================================================================
// 文件：shaders/post/hiz.comp.glsl
// 模块定位：后处理着色器，负责全屏背景、软件光栅结果合并和 Hi-Z 层级深度生成。
// 数据流：读取 帧缓冲、atomic raster image 或 depth texture，输出最终颜色或下一帧剔除使用的层级深度。
// 方法说明：后处理阶段把异步生成的中间结果规约为统一图像，并为下一帧可见性判断建立反馈。
// 正确性约束：atomic resolve 的深度比较语义要与写入端一致；Hi-Z 归约必须保守，不能错误剔除可见对象。
// 注释风格：使用中文解释 GPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#version 460


#ifndef NV_HIZ_MAX_LEVELS


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define NV_HIZ_MAX_LEVELS   16
#endif
#ifndef NV_HIZ_MSAA_SAMPLES


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define NV_HIZ_MSAA_SAMPLES 0
#endif
#ifndef NV_HIZ_IS_FIRST


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define NV_HIZ_IS_FIRST 1
#endif
#ifndef NV_HIZ_FORMAT


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define NV_HIZ_FORMAT r32f
#endif
#ifndef NV_HIZ_OUTPUT_NEAR


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define NV_HIZ_OUTPUT_NEAR 1
#endif
#ifndef NV_HIZ_LEVELS


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define NV_HIZ_LEVELS 3
#endif
#ifndef NV_HIZ_NEAR_LEVEL


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define NV_HIZ_NEAR_LEVEL 0
#endif
#ifndef NV_HIZ_FAR_LEVEL


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define NV_HIZ_FAR_LEVEL 0
#endif
#ifndef NV_HIZ_REVERSED_Z


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define NV_HIZ_REVERSED_Z 0
#endif
#ifndef NV_HIZ_USE_STEREO


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define NV_HIZ_USE_STEREO 0
#endif


#if NV_HIZ_LEVELS > 1
  #extension GL_KHR_shader_subgroup_basic : require
  #extension GL_KHR_shader_subgroup_shuffle : require
#endif


#if NV_HIZ_REVERSED_Z
  #define minOp max
  #define maxOp min
#else
  #define minOp min
  #define maxOp max
#endif


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(local_size_x=32,local_size_y=2) in;


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(push_constant) uniform passUniforms {
  ivec4 srcSize;
  int   writeLod;
  int   startLod;
  int   layer;
  int   _pad0;
  bvec4 levelActive;
};

#if NV_HIZ_USE_STEREO
  #define samplerTypeMS sampler2DMSArray
  #define samplerType   sampler2DArray
  #define imageType     image2DArray


  // 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
  // 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
  #define IACCESS(v,l)  ivec3(v,l)
#else
  #define samplerTypeMS sampler2DMS
  #define samplerType   sampler2D
  #define imageType     image2D


  // 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
  // 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
  #define IACCESS(v,l)  v
#endif


#if NV_HIZ_IS_FIRST && NV_HIZ_MSAA_SAMPLES


  // 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
  // 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
  layout(binding=0) uniform samplerTypeMS texDepth;
#else


  // 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
  // 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
  layout(binding=0) uniform samplerType   texDepth;
#endif


  // 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
  // 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
  layout(binding=1) uniform samplerType   texNear;


  // 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
  // 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
  layout(binding=2,NV_HIZ_FORMAT) uniform imageType imgNear;


  // 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
  // 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
  layout(binding=3,NV_HIZ_FORMAT) uniform imageType imgLevels[NV_HIZ_MAX_LEVELS];


// 函数：main。作为本着色器阶段入口，按绑定资源执行当前 GPU 工作。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该入口位于控制流根部，调用顺序决定后续资源生命周期和数据依赖。
void main()
{


  ivec2 base = ivec2(gl_WorkGroupID.xy) * 8;


  ivec2 subset = ivec2(int(gl_LocalInvocationID.x) & 1, int(gl_LocalInvocationID.x) / 2);

  subset += gl_LocalInvocationID.x >= 16 ? ivec2(2,-8) : ivec2(0,0);

  subset += ivec2(gl_LocalInvocationID.y * 4,0);

#if NV_HIZ_LEVELS > 1
  uint laneID = gl_SubgroupInvocationID;
#endif

  ivec2 outcoord = base + subset;

  ivec2 coord = outcoord * 2;


#if NV_HIZ_IS_FIRST && NV_HIZ_MSAA_SAMPLES

  #if NV_HIZ_REVERSED_Z
  float zMin = 0;
  float zMax = 1;
  #else
  float zMin = 1;
  float zMax = 0;
  #endif

  for (int i = 0; i < NV_HIZ_MSAA_SAMPLES; i++){

    vec4 zRead = vec4(texelFetch(texDepth, IACCESS(min(coord + ivec2(0,0), srcSize.zw), layer), i).r,
                      texelFetch(texDepth, IACCESS(min(coord + ivec2(1,0), srcSize.zw), layer), i).r,
                      texelFetch(texDepth, IACCESS(min(coord + ivec2(0,1), srcSize.zw), layer), i).r,
                      texelFetch(texDepth, IACCESS(min(coord + ivec2(1,1), srcSize.zw), layer), i).r);

    zMin = minOp(zMin, minOp(zRead.x, minOp(zRead.y, minOp(zRead.z, zRead.w))));
    zMax = maxOp(zMax, maxOp(zRead.x, maxOp(zRead.y, maxOp(zRead.z, zRead.w))));
  }
#else

  #if NV_HIZ_IS_FIRST
    #define texRead texDepth
  #else
    #define texRead texNear
  #endif


  #if NV_HIZ_REVERSED_Z
  float zMin = 0;
  float zMax = 1;
  #else
  float zMin = 1;
  float zMax = 0;
  #endif


  coord = min(coord, srcSize.zw);


  vec4 zRead;
  zRead.x = texelFetchOffset(texRead, IACCESS(coord, layer), startLod, ivec2(0,0)).r;
  zRead.y = texelFetchOffset(texRead, IACCESS(coord, layer), startLod, ivec2(1,0)).r;
  zRead.z = texelFetchOffset(texRead, IACCESS(coord, layer), startLod, ivec2(0,1)).r;
  zRead.w = texelFetchOffset(texRead, IACCESS(coord, layer), startLod, ivec2(1,1)).r;

  zMax = maxOp(zRead.x, maxOp(zRead.y, maxOp(zRead.z, zRead.w)));
  zMin = minOp(zRead.x, minOp(zRead.y, minOp(zRead.z, zRead.w)));
#endif


#if !(NV_HIZ_IS_FIRST && NV_HIZ_FAR_LEVEL > 0)
  imageStore(imgLevels[writeLod + 0], IACCESS(outcoord,layer), vec4(zMax));
#endif
#if NV_HIZ_IS_FIRST && NV_HIZ_OUTPUT_NEAR && NV_HIZ_NEAR_LEVEL == 0
  imageStore(imgNear, IACCESS(outcoord,layer), vec4(zMin));
#endif


#if NV_HIZ_LEVELS > 1


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

  if ((levelActive.y || levelActive.z) && (laneID & 3) == 0)
  {
    outcoord /= 2;

    if (outcoord.x >= 0 && outcoord.y >= 0) {
      zMax = maxOp(maxOp(maxOp(zRead0.x, zRead0.y),zRead0.z),zRead0.w);

    #if !(NV_HIZ_IS_FIRST && NV_HIZ_FAR_LEVEL > 1)
      imageStore(imgLevels[writeLod + 1], IACCESS(outcoord, layer), vec4(zMax));
    #endif
    #if NV_HIZ_IS_FIRST && NV_HIZ_OUTPUT_NEAR && NV_HIZ_NEAR_LEVEL >= 1
      zMin = minOp(minOp(minOp(zRead1.x, zRead1.y),zRead1.z),zRead1.w);
      #if NV_HIZ_NEAR_LEVEL == 1
      imageStore(imgNear, IACCESS(outcoord, layer), vec4(zMin));
      #endif
    #endif


    #if NV_HIZ_LEVELS > 2
      if (levelActive.z) {
        outcoord /= 2;

        if (outcoord.x >= 0 && outcoord.y >= 0) {


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


          if ((laneID == 0) || (laneID == 8)) {
            zMax = maxOp(maxOp(maxOp(zRead0.x, zRead0.y),zRead0.z),zRead0.w);

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
