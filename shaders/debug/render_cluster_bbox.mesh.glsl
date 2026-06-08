//==============================================================================
// 文件：shaders/debug/render_cluster_bbox.mesh.glsl
// 模块定位：调试可视化着色器，用线框方式显示实例或 簇 的空间范围。
// 数据流：读取实例、几何或当前渲染 簇 列表，输出包围盒线段和固定调试颜色。
// 方法说明：空间范围可视化用于验证 LOD 层次、剔除判断和实例变换是否符合预期。
// 正确性约束：调试绘制不应改变主渲染数据；线框生成必须兼容 EXT/NV 网格着色器路径。
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
#if USE_EXT_MESH_SHADER
#extension GL_EXT_mesh_shader : require
#else
#extension GL_NV_mesh_shader : require
#endif
#extension GL_EXT_control_flow_attributes: require


// 依赖说明：引入共享布局、剔除、着色或阶段间复用的着色器片段。
// 这些 include 共同决定本文件能访问的结构布局、数学辅助函数和编译期宏。
#include "shaderio.h"


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(push_constant) uniform pushData
{
  uint numRenderInstances;
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

#if USE_STREAMING


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer
{
  SceneStreaming streaming;
};

#endif


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(location=0) out Interpolants {
  flat uint clusterID;
} OUT[];

#ifndef MESHSHADER_WORKGROUP_SIZE


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define MESHSHADER_WORKGROUP_SIZE  32
#endif

#ifndef MESHSHADER_BBOX_COUNT


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define MESHSHADER_BBOX_COUNT 8
#endif


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(local_size_x=MESHSHADER_WORKGROUP_SIZE) in;


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(max_vertices=MESHSHADER_BBOX_COUNT * MESHSHADER_BBOX_VERTICES, max_primitives=MESHSHADER_BBOX_COUNT * MESHSHADER_BBOX_LINES) out;


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(lines) out;


// 函数：writePrimitiveLineIndices。把当前状态写入缓存、缓冲、文件或着色器可消费的数据布局。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：写入路径应明确字节对齐、所有权和可见性，避免后续读取端解释错误。
void writePrimitiveLineIndices(uint idx, uvec2 vertexIndices)
{
#if USE_EXT_MESH_SHADER
  gl_PrimitiveLineIndicesEXT[idx] = vertexIndices;
#else
  gl_PrimitiveIndicesNV[idx * 2 + 0] = vertexIndices.x;
  gl_PrimitiveIndicesNV[idx * 2 + 1] = vertexIndices.y;
#endif
}


// 函数：main。作为本着色器阶段入口，按绑定资源执行当前 GPU 工作。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该入口位于控制流根部，调用顺序决定后续资源生命周期和数据依赖。
void main()
{
#if USE_EXT_MESH_SHADER

  uint workGroupID = getWorkGroupIndexLinearized(gl_WorkGroupID);
#else
  uint workGroupID = gl_WorkGroupID.x;
#endif
  uint numRenderedClusters = build.numRenderedClusters;
  uint baseID   = workGroupID * MESHSHADER_BBOX_COUNT;
  uint numBoxes = min(numRenderedClusters, baseID + MESHSHADER_BBOX_COUNT) - baseID;

#if USE_EXT_MESH_SHADER

  SetMeshOutputsEXT(numBoxes * MESHSHADER_BBOX_VERTICES, numBoxes * MESHSHADER_BBOX_LINES);
  if (numBoxes == 0) return;
#else
  if (gl_LocalInvocationID.x == 0)
  {
    gl_PrimitiveCountNV = numBoxes * MESHSHADER_BBOX_LINES;
  }
#endif

  const uint vertexRuns = ((MESHSHADER_BBOX_COUNT * MESHSHADER_BBOX_VERTICES) + MESHSHADER_WORKGROUP_SIZE-1) / MESHSHADER_WORKGROUP_SIZE;

  [[unroll]]
  for (uint32_t run = 0; run < vertexRuns; run++)
  {
    uint vert   = gl_LocalInvocationID.x + run * MESHSHADER_WORKGROUP_SIZE;
    uint box    = vert / MESHSHADER_BBOX_VERTICES;
    uint corner = vert % MESHSHADER_BBOX_VERTICES;


    uint boxLoad = min(box,numBoxes-1);

    ClusterInfo cinfo = build.renderClusterInfos.d[boxLoad + baseID];
    uint clusterID = cinfo.clusterID;
    RenderInstance instance = instances[cinfo.instanceID];

  #if USE_STREAMING

    Cluster_in clusterRef = Cluster_in(streaming.resident.clusters.d[clusterID]);
  #else
    Geometry geometry = geometries[instance.geometryID];

    Cluster_in clusterRef = Cluster_in(geometry.preloadedClusters.d[clusterID]);
  #endif


    BBox bbox = Cluster_getBBox(clusterRef);

    bvec3 weight   = bvec3((corner & 1) != 0, (corner & 2) != 0, (corner & 4) != 0);

    vec3 cornerPos = mix(bbox.lo, bbox.hi, weight);

    if (box < numBoxes)
    {
    #if USE_EXT_MESH_SHADER
      gl_MeshVerticesEXT[vert].gl_Position =
    #else
      gl_MeshVerticesNV[vert].gl_Position =
    #endif
        view.viewProjMatrix * vec4(instance.worldMatrix * vec4(cornerPos,1), 1);
      OUT[vert].clusterID = clusterID;
    }
  }

  {
    uvec2 boxIndices[4] = uvec2[4](

      uvec2(0,1),uvec2(1,3),uvec2(3,2),uvec2(2,0)
    );

    uint subID = gl_LocalInvocationID.x & (MESHSHADER_BBOX_THREADS-1);
    uint box   = gl_LocalInvocationID.x / MESHSHADER_BBOX_THREADS;

    uvec2 circle = boxIndices[subID];

    if (box < numBoxes)
    {

             writePrimitiveLineIndices(box * 12 + subID + 0, circle + box * MESHSHADER_BBOX_VERTICES);

             writePrimitiveLineIndices(box * 12 + subID + 4, circle + 4 + box * MESHSHADER_BBOX_VERTICES);
             writePrimitiveLineIndices(box * 12 + subID + 8, uvec2(subID, subID + 4) + box * MESHSHADER_BBOX_VERTICES);
    }
  }
}
