//==============================================================================
// 文件：shaders/build/build_setup.comp.glsl
// 模块定位：构建阶段计算着色器，依据本帧统计计数生成后续间接执行参数。
// 数据流：读取 SceneBuilding 中的 traversal/render 计数，写出 调度 和 绘制 indirect 命令。
// 方法说明：间接命令让 GPU 自主决定后续工作量，是 GPU driven rendering 的关键环节。
// 正确性约束：写 indirect 参数前必须完成对应计数；计数需要按资源上限 clamp，避免越界执行。
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
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require


// 依赖说明：引入共享布局、剔除、着色或阶段间复用的着色器片段。
// 这些 include 共同决定本文件能访问的结构布局、数学辅助函数和编译期宏。
#include "shaderio.h"


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(push_constant) uniform pushData
{
  uint setup;
} push;


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
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
layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;
};


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) coherent buffer buildBufferRW
{
  SceneBuilding buildRW;
};


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(local_size_x=1) in;


#ifndef MESHSHADER_BBOX_COUNT


// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define MESHSHADER_BBOX_COUNT 8
#endif


#if USE_TWO_PASS_CULLING


// 函数：setupSecondPass。初始化本模块所需状态、资源或 GPU 侧绑定。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
void setupSecondPass()
{

  buildRW.pass = 1;
  buildRW.traversalTaskCounter = 0;
  buildRW.traversalGroupCounter = 0;
  buildRW.renderClusterCounter = 0;
  buildRW.renderClusterCounterSW = 0;
  buildRW.traversalInfoReadCounter = 0;
}
#endif


// 函数：main。作为本着色器阶段入口，按绑定资源执行当前 GPU 工作。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该入口位于控制流根部，调用顺序决定后续资源生命周期和数据依赖。
void main()
{


  if (push.setup == BUILD_SETUP_TRAVERSAL_RUN)
  {

    int traversalTaskCounter = min(buildRW.traversalTaskCounter, int(build.maxTraversalInfos));
    buildRW.traversalTaskCounter = traversalTaskCounter;


    buildRW.traversalInfoWriteCounter = uint(traversalTaskCounter);
  }
#if TARGETS_RASTERIZATION && USE_SW_RASTER
  else if (push.setup == BUILD_SETUP_DRAW)
  {

    uint renderClusterCounter   = buildRW.renderClusterCounter;
    uint renderClusterCounterSW = buildRW.renderClusterCounterSW;


    uint numRenderedClusters   = min(renderClusterCounter,   build.maxRenderClusters);

    uint numRenderedClustersSW = min(renderClusterCounterSW, build.maxRenderClusters);

  #if USE_EXT_MESH_SHADER

    uvec3 grid = fit16bitLaunchGrid(numRenderedClusters);
    buildRW.indirectDrawClustersEXT.gridX = grid.x;
    buildRW.indirectDrawClustersEXT.gridY = grid.y;
    buildRW.indirectDrawClustersEXT.gridZ = grid.z;

    grid = fit16bitLaunchGrid((numRenderedClusters + MESHSHADER_BBOX_COUNT - 1) / MESHSHADER_BBOX_COUNT);
    buildRW.indirectDrawClusterBoxesEXT.gridX = grid.x;
    buildRW.indirectDrawClusterBoxesEXT.gridY = grid.y;
    buildRW.indirectDrawClusterBoxesEXT.gridZ = grid.z;
  #else
    buildRW.indirectDrawClustersNV.count = numRenderedClusters;
    buildRW.indirectDrawClustersNV.first = 0;

    buildRW.indirectDrawClusterBoxesNV.count = (numRenderedClusters + MESHSHADER_BBOX_COUNT - 1) / MESHSHADER_BBOX_COUNT;
    buildRW.indirectDrawClusterBoxesNV.first = 0;
  #endif
    buildRW.numRenderedClusters = numRenderedClusters;

  #if USE_16BIT_DISPATCH

    uvec3 grid = fit16bitLaunchGrid(numRenderedClustersSW);
    buildRW.indirectDrawClustersSW.gridX = grid.x;
    buildRW.indirectDrawClustersSW.gridY = grid.y;
    buildRW.indirectDrawClustersSW.gridZ = grid.z;
  #else
    buildRW.indirectDrawClustersSW.gridX = numRenderedClustersSW;
  #endif
    buildRW.numRenderedClustersSW = numRenderedClustersSW;


    atomicMax(readback.numRenderClusters, renderClusterCounter);

    atomicMax(readback.numRenderClustersSW, renderClusterCounterSW);
  #if USE_SEPARATE_GROUPS

    atomicMax(readback.numTraversalTasks, max(buildRW.traversalInfoWriteCounter, buildRW.traversalGroupCounter));
  #else


    atomicMax(readback.numTraversalTasks, buildRW.traversalInfoWriteCounter);
  #endif

  #if USE_RENDER_STATS


    readback.numRenderedClusters   += numRenderedClusters;
    readback.numRenderedClustersSW += numRenderedClustersSW;
    readback.numTraversedTasks     += buildRW.traversalInfoWriteCounter;
  #endif
  #if USE_TWO_PASS_CULLING

    setupSecondPass();
  #endif
  }
#elif TARGETS_RASTERIZATION
  else if (push.setup == BUILD_SETUP_DRAW)
  {

    uint renderClusterCounter = buildRW.renderClusterCounter;


    uint numRenderedClusters = min(renderClusterCounter, build.maxRenderClusters);
  #if USE_EXT_MESH_SHADER

    uvec3 grid = fit16bitLaunchGrid(numRenderedClusters);
    buildRW.indirectDrawClustersEXT.gridX = grid.x;
    buildRW.indirectDrawClustersEXT.gridY = grid.y;
    buildRW.indirectDrawClustersEXT.gridZ = grid.z;

    grid = fit16bitLaunchGrid((numRenderedClusters + MESHSHADER_BBOX_COUNT - 1) / MESHSHADER_BBOX_COUNT);
    buildRW.indirectDrawClusterBoxesEXT.gridX = grid.x;
    buildRW.indirectDrawClusterBoxesEXT.gridY = grid.y;
    buildRW.indirectDrawClusterBoxesEXT.gridZ = grid.z;
  #else
    buildRW.indirectDrawClustersNV.count = numRenderedClusters;
    buildRW.indirectDrawClustersNV.first = 0;

    buildRW.indirectDrawClusterBoxesNV.count = (numRenderedClusters + MESHSHADER_BBOX_COUNT - 1) / MESHSHADER_BBOX_COUNT;
    buildRW.indirectDrawClusterBoxesNV.first = 0;
  #endif
    buildRW.numRenderedClusters = numRenderedClusters;


    atomicMax(readback.numRenderClusters, renderClusterCounter);
  #if USE_SEPARATE_GROUPS

    atomicMax(readback.numTraversalTasks, max(buildRW.traversalInfoWriteCounter, buildRW.traversalGroupCounter));
  #else


    atomicMax(readback.numTraversalTasks, buildRW.traversalInfoWriteCounter);
  #endif

  #if USE_RENDER_STATS

    readback.numRenderedClusters += numRenderedClusters;
    readback.numTraversedTasks   += buildRW.traversalInfoWriteCounter;
  #endif

  #if USE_TWO_PASS_CULLING

    setupSecondPass();
  #endif
  }
#endif
}
