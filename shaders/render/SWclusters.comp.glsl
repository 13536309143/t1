//==============================================================================
// 文件：shaders/render/SWclusters.comp.glsl
// 模块定位：渲染阶段着色器，把遍历输出的 簇 转换为 帧缓冲 中的颜色、深度和调试信息。
// 数据流：读取 render 簇 list 和 组/簇 payload，输出硬件网格着色器 primitive 或计算软件光栅结果。
// 方法说明：系统同时支持硬件网格着色器和软件光栅路径，用不同执行模型覆盖不同大小和密度的 簇。
// 正确性约束：硬件与软件路径必须共享同一深度编码和材质解释；簇 payload 地址必须来自有效驻留数据。
// 注释风格：使用中文解释 GPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_image_int64 : enable
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_ballot : require


// 依赖说明：引入共享布局、剔除、着色或阶段间复用的着色器片段。
// 这些 include 共同决定本文件能访问的结构布局、数学辅助函数和编译期宏。
#include "shaderio.h"


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(push_constant) uniform pushData
{
  uint instanceID;
}
push;


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar,binding=BINDINGS_READBACK_SSBO,set=0) buffer readbackBuffer
{
  Readback readback;
};


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer
{
  RenderInstance instances[];
};


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];
};


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;
};


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) buffer buildBufferRW
{
  SceneBuilding buildRW;
};

#if USE_STREAMING


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer
{
  SceneStreaming streaming;
};


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) buffer streamingBufferRW
{
  SceneStreaming streamingRW;
};
#endif


#if USE_TWO_PASS_CULLING


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar[2];
#else


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;
#endif


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(set = 0, binding = BINDINGS_RASTER_ATOMIC, r64ui) uniform u64image2D imgRasterAtomic;

#if CLUSTER_TRIANGLE_COUNT > 64


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define COMPUTE_WORKGROUP_SIZE 64
#else


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define COMPUTE_WORKGROUP_SIZE CLUSTER_TRIANGLE_COUNT
#endif


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(local_size_x = COMPUTE_WORKGROUP_SIZE) in;

const uint MESHLET_VERTEX_ITERATIONS = ((CLUSTER_VERTEX_COUNT + COMPUTE_WORKGROUP_SIZE - 1) / COMPUTE_WORKGROUP_SIZE);
const uint MESHLET_TRIANGLE_ITERATIONS = ((CLUSTER_TRIANGLE_COUNT + COMPUTE_WORKGROUP_SIZE - 1) / COMPUTE_WORKGROUP_SIZE);

#include "culling.glsl"
#include "render_shading.glsl"


// 函数：edgeFunction。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
float edgeFunction(vec2 a, vec2 b, vec2 c, float winding)
{
  float edge = ((c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x));
  return edge * winding;
}


// 函数：rasterTriangle。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void rasterTriangle(vec2 pixel, uint packedColor, uvec3 indices, RasterVertex a, RasterVertex b, RasterVertex c, float triArea, float winding)
{

  float baryA = edgeFunction(b.xy, c.xy, pixel, winding);

  float baryB = edgeFunction(c.xy, a.xy, pixel, winding);

  float baryC = edgeFunction(a.xy, b.xy, pixel, winding);

  if (baryA >= 0 && baryB >= 0 && baryC >= 0){
    baryA /= triArea;
    baryB /= triArea;
    baryC /= triArea;

    float depth = a.z * baryA +
                  b.z * baryB +
                  c.z * baryC;

    uint64_t u64 = packUint2x32(uvec2(packedColor, floatBitsToUint(depth)));
    imageAtomicMax(imgRasterAtomic, ivec2(pixel.xy), u64);
  }
}

shared RasterVertex s_vertices[CLUSTER_VERTEX_COUNT];


// 函数：main。作为本着色器阶段入口，按绑定资源执行当前 GPU 工作。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该入口位于控制流根部，调用顺序决定后续资源生命周期和数据依赖。
void main()
{

  uint workGroupID = getWorkGroupIndex(gl_WorkGroupID);
#if USE_16BIT_DISPATCH
  bool isValid      = workGroupID < build.numRenderedClustersSW;
  ClusterInfo cinfo = build.renderClusterInfosSW.d[min(workGroupID, build.numRenderedClustersSW-1)];
  if (!isValid) return;
#else
  ClusterInfo cinfo = build.renderClusterInfosSW.d[workGroupID];
#endif

  uint instanceID = cinfo.instanceID;
  uint clusterID  = cinfo.clusterID;

  RenderInstance instance = instances[instanceID];
  Geometry geometry       = geometries[instance.geometryID];

#if USE_STREAMING

  Cluster_in clusterRef = Cluster_in(streaming.resident.clusters.d[clusterID]);
#else

  Cluster_in clusterRef = Cluster_in(geometry.preloadedClusters.d[clusterID]);
#endif
  Cluster cluster = clusterRef.d;

  vec3s_in  oVertices    = vec3s_in(Cluster_getVertexPositions(clusterRef));
  uint8s_in localIndices = uint8s_in(Cluster_getTriangleIndices(clusterRef));

  uint vertMax = cluster.vertexCountMinusOne;
  uint triMax  = cluster.triangleCountMinusOne;

#if USE_RENDER_STATS
  if (gl_LocalInvocationID.x == 0) {
    atomicAdd(readback.numRenderedTrianglesSW, uint(triMax + 1));
  }
#endif

  [[unroll]] for(uint i = 0; i < uint(MESHLET_VERTEX_ITERATIONS); i++)
  {
    uint vert        = gl_LocalInvocationID.x + i * COMPUTE_WORKGROUP_SIZE;

    uint vertLoad    = min(vert, vertMax);

    vec3 oPos = oVertices.d[vertLoad];

    vec3 wPos = instance.worldMatrix * vec4(oPos, 1.0f);

    if(vert <= vertMax)
    {
      s_vertices[vert] = getRasterVertex(view.viewProjMatrix * vec4(wPos,1));
    }
  }


  memoryBarrierShared();

  barrier();

  uint numRasteredTriangles = 0;

  for(uint tri = gl_LocalInvocationID.x; tri <= triMax; tri += COMPUTE_WORKGROUP_SIZE )
  {
    uint triLoad  = tri;
    uvec3 indices = uvec3(localIndices.d[triLoad * 3 + 0],
                          localIndices.d[triLoad * 3 + 1],
                          localIndices.d[triLoad * 3 + 2]);
#if !USE_FORCED_TWO_SIDED
    if (instance.flipWinding != 0
#if USE_TWO_SIDED
      || (instance.twoSided != 0 && !isFrontFacingSW(s_vertices[indices.x],
                                                     s_vertices[indices.y],
                                                     s_vertices[indices.z]))
#endif
    )
    {
      indices.xy = indices.yx;
    }
#endif

    RasterVertex a = s_vertices[indices.x];
    RasterVertex b = s_vertices[indices.y];
    RasterVertex c = s_vertices[indices.z];

    vec2 pixelMin;
    vec2 pixelMax;
    float triArea;


    bool visible  = testTriangleSW(a,b,c, pixelMin, pixelMax, triArea);
    float winding = 1.0;
  #if USE_FORCED_TWO_SIDED
    if (triArea < 0)
    {
      triArea = -triArea;
      winding = -winding;
    }
  #endif

    if (visible)
    {

      uint triangleCountMinusOne = CLUSTER_TRIANGLE_COUNT-1;
      float relative   = (float(tri) / float(triangleCountMinusOne)) * 0.25 + 0.75;


    #if USE_DEPTH_ONLY
      uint packedColor = packUnorm4x8(vec4(0,0,0,1));
    #else

      vec4 color       = vec4(colorizeID(clusterID) * relative, 1.0);

      uint packedColor = packUnorm4x8(color);
    #endif


      uvec2 pixelDim  = uvec2(pixelMax - pixelMin);

      for (uint p = 0; p < pixelDim.x * pixelDim.y; p++)
      {
        uint px = p % pixelDim.x;
        uint py = p / pixelDim.x;


        vec2 pixel = pixelMin + vec2(0.5) + vec2(px,py);


        rasterTriangle(pixel, packedColor, indices, a, b, c, triArea, winding);
      }
#if USE_RENDER_STATS
      numRasteredTriangles += subgroupBallotBitCount(subgroupBallot(true));
#endif
    }
  }
#if USE_RENDER_STATS
  if (subgroupElect()) {

    atomicAdd(readback.numRasteredTrianglesSW, numRasteredTriangles);
  }
#endif
}
