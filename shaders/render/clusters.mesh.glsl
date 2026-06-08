//==============================================================================
// 文件：shaders/render/clusters.mesh.glsl
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
#extension GL_KHR_shader_subgroup_ballot : require

#if USE_EXT_MESH_SHADER
#extension GL_EXT_mesh_shader : require
#else
#extension GL_NV_mesh_shader : require
#endif
#extension GL_EXT_control_flow_attributes : require


// 依赖说明：引入共享布局、剔除、着色或阶段间复用的着色器片段。
// 这些 include 共同决定本文件能访问的结构布局、数学辅助函数和编译期宏。
#include "shaderio.h"

#if USE_PRIMITIVE_CULLING && (USE_EXT_MESH_SHADER || !USE_CULLING)
#undef USE_PRIMITIVE_CULLING


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define USE_PRIMITIVE_CULLING 0
#endif


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

#if !USE_DEPTH_ONLY


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(location = 0) out Interpolants
{
  flat uint clusterID;
  flat uint instanceID;
#if ALLOW_SHADING
  vec3      wPos;
#endif
}
OUT[];
#if ALLOW_SHADING && (ALLOW_VERTEX_NORMALS || ALLOW_VERTEX_TEXCOORDS)


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(location = 3) out Interpolants2
{
  flat uint vertexID;
}
OUTBARY[];
#endif
#endif

#ifndef MESHSHADER_WORKGROUP_SIZE


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define MESHSHADER_WORKGROUP_SIZE 32
#endif


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(local_size_x = MESHSHADER_WORKGROUP_SIZE) in;


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(max_vertices = CLUSTER_VERTEX_COUNT, max_primitives = CLUSTER_TRIANGLE_COUNT) out;


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(triangles) out;

const uint MESHLET_VERTEX_ITERATIONS = ((CLUSTER_VERTEX_COUNT + MESHSHADER_WORKGROUP_SIZE - 1) / MESHSHADER_WORKGROUP_SIZE);
const uint MESHLET_TRIANGLE_ITERATIONS = ((CLUSTER_TRIANGLE_COUNT + MESHSHADER_WORKGROUP_SIZE - 1) / MESHSHADER_WORKGROUP_SIZE);

#if USE_PRIMITIVE_CULLING || USE_TWO_SIDED


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define CULLING_NO_HIZ
#include "culling.glsl"
#endif

#if USE_EXT_MESH_SHADER && USE_TWO_SIDED
shared vec4 s_vertices[CLUSTER_VERTEX_COUNT];
#endif


// 函数：main。作为本着色器阶段入口，按绑定资源执行当前 GPU 工作。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该入口位于控制流根部，调用顺序决定后续资源生命周期和数据依赖。
void main()
{
#if USE_EXT_MESH_SHADER


uint workGroupID  = getWorkGroupIndexLinearized(gl_WorkGroupID);
  bool isValid      = workGroupID < build.numRenderedClusters;

  ClusterInfo cinfo = build.renderClusterInfos.d[min(workGroupID, build.numRenderedClusters-1)];
#else

  uint workGroupID  = gl_WorkGroupID.x;
  ClusterInfo cinfo = build.renderClusterInfos.d[workGroupID];
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
  uint vertMax = cluster.vertexCountMinusOne;
  uint triMax  = cluster.triangleCountMinusOne;

#if USE_EXT_MESH_SHADER

  uint vertCount = isValid ? vertMax + 1 : 0;
  uint triCount  = isValid ? triMax + 1 : 0;

  SetMeshOutputsEXT(vertCount, triCount);
  if (triCount == 0)
    return;
#elif !USE_PRIMITIVE_CULLING

  if (gl_LocalInvocationID.x == 0) {
    gl_PrimitiveCountNV = triMax + 1;
  }
#endif

#if USE_RENDER_STATS
  if (gl_LocalInvocationID.x == 0) {
    atomicAdd(readback.numRenderedTriangles, uint(triMax + 1));
  #if !USE_PRIMITIVE_CULLING
    atomicAdd(readback.numRasteredTriangles, uint(triMax + 1));
  #endif
  }
#endif

  vec3s_in  oVertices    = vec3s_in(Cluster_getVertexPositions(clusterRef));
  uint8s_in localIndices = uint8s_in(Cluster_getTriangleIndices(clusterRef));


  uint currentLodLevel = cluster.lodLevel;


  float lodTransitionFactor = 0.0;


  [[unroll]] for(uint i = 0; i < uint(MESHLET_VERTEX_ITERATIONS); i++)
  {

    uint vert        = gl_LocalInvocationID.x + i * MESHSHADER_WORKGROUP_SIZE;

    uint vertLoad    = min(vert, vertMax);

    vec3 oPos = oVertices.d[vertLoad];

    vec3 wPos = instance.worldMatrix * vec4(oPos, 1.0f);


    if(vert <= vertMax)
    {


      vec4 hPos = view.viewProjMatrix * vec4(wPos,1);

    #if USE_EXT_MESH_SHADER
      gl_MeshVerticesEXT[vert].gl_Position = hPos;
    #else
      gl_MeshVerticesNV[vert].gl_Position = hPos;
    #endif

    #if USE_EXT_MESH_SHADER && USE_TWO_SIDED
      s_vertices[vert] = hPos;
    #endif

    #if !USE_DEPTH_ONLY
    #if ALLOW_SHADING
      OUT[vert].wPos                      = wPos.xyz;
    #endif
    #if ALLOW_SHADING && (ALLOW_VERTEX_NORMALS || ALLOW_VERTEX_TEXCOORDS)
      OUTBARY[vert].vertexID              = vert;
    #endif
      OUT[vert].clusterID                 = clusterID;
      OUT[vert].instanceID                = instanceID;
    #endif
    }
  }

  uint triOutCount = 0;
  [[unroll]] for(uint i = 0; i < uint(MESHLET_TRIANGLE_ITERATIONS); i++)
  {

    uint tri     = gl_LocalInvocationID.x + i * MESHSHADER_WORKGROUP_SIZE;

    uint triLoad = min(tri, triMax);

    uvec3 indices = uvec3(localIndices.d[triLoad * 3 + 0],
                          localIndices.d[triLoad * 3 + 1],
                          localIndices.d[triLoad * 3 + 2]);
#if !USE_FORCED_TWO_SIDED

    if (instance.flipWinding != 0
#if USE_TWO_SIDED && !USE_EXT_MESH_SHADER

      || (instance.twoSided != 0 && !isFrontFacingHW(gl_MeshVerticesNV[indices.x].gl_Position,
                                                     gl_MeshVerticesNV[indices.y].gl_Position,
                                                     gl_MeshVerticesNV[indices.z].gl_Position))
#elif USE_TWO_SIDED && USE_EXT_MESH_SHADER

      || (instance.twoSided != 0 && !isFrontFacingHW(s_vertices[indices.x],s_vertices[indices.y],s_vertices[indices.z]))
#endif
    )
    {

      indices.xy = indices.yx;
    }
#endif
#if USE_PRIMITIVE_CULLING

    bool isRendered = tri <= triMax

       && testTriangleHW( gl_MeshVerticesNV[indices.x].gl_Position,gl_MeshVerticesNV[indices.y].gl_Position,gl_MeshVerticesNV[indices.z].gl_Position);


    uvec4 voteRendered = subgroupBallot(isRendered);


    uint triOut = subgroupBallotExclusiveBitCount(voteRendered) + triOutCount;


    triOutCount += subgroupBallotBitCount(voteRendered);
#else

    bool isRendered = tri <= triMax;
    uint triOut     = tri;
#endif

    if(isRendered)
    {


    #if USE_EXT_MESH_SHADER
      gl_PrimitiveTriangleIndicesEXT[triOut] = indices;
      #if !USE_DEPTH_ONLY

      gl_MeshPrimitivesEXT[triOut].gl_PrimitiveID = int(tri);
      #endif
    #else

      gl_PrimitiveIndicesNV[triOut * 3 + 0] = indices.x;
      gl_PrimitiveIndicesNV[triOut * 3 + 1] = indices.y;
      gl_PrimitiveIndicesNV[triOut * 3 + 2] = indices.z;
      #if !USE_DEPTH_ONLY

      gl_MeshPrimitivesNV[triOut].gl_PrimitiveID = int(tri);
      #endif
    #endif
    }
  }

#if USE_PRIMITIVE_CULLING

  if (gl_LocalInvocationID.x == 0) {
    gl_PrimitiveCountNV = triOutCount;
  }

#if USE_RENDER_STATS
  if (gl_LocalInvocationID.x == 0) {

    atomicAdd(readback.numRasteredTriangles, triOutCount);
  }
#endif
#endif
}
