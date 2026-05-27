/*
第一遍 Pass (pass = 0)
  ↓
收集初步的culling结果
  ↓
setupSecondPass() 被调用
  ↓ (buildRW.pass = 1，重置计数器)
第二遍 Pass (pass = 1)
  ↓
基于第一遍的结果继续culling
  ↓
atomicMax 和 += 确保两遍结果都被正确合并
*/

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

#include "shaderio.h"

layout(push_constant) uniform pushData
{
  uint setup;
} push;

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer
{
  RenderInstance instances[];
};

layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];
};

layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;  
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) coherent buffer buildBufferRW
{
  SceneBuilding buildRW;  
};

////////////////////////////////////////////

layout(local_size_x=1) in;

////////////////////////////////////////////

#ifndef MESHSHADER_BBOX_COUNT
#define MESHSHADER_BBOX_COUNT 8
#endif

//2
//重置所有计数器，为第二遍pass做准备
#if USE_TWO_PASS_CULLING
void setupSecondPass()
{
  // setup second pass  
  buildRW.pass = 1;
  buildRW.traversalTaskCounter = 0;
  buildRW.traversalGroupCounter = 0;
  buildRW.renderClusterCounter = 0;
  buildRW.renderClusterCounterSW = 0;
  buildRW.traversalInfoReadCounter = 0;
}
#endif
//8
void main()
{  
  // special operations for setting up indirect dispatches
  // or clamping other operations to actual limits
  
  if (push.setup == BUILD_SETUP_TRAVERSAL_RUN)
  {
    // during traversal_init we might overshoot the traversalTaskCounter  
    int traversalTaskCounter = min(buildRW.traversalTaskCounter, int(build.maxTraversalInfos));
    buildRW.traversalTaskCounter = traversalTaskCounter;
    // also set up the initial writeCounter to be equal, so that new jobs are enqueued after it
    buildRW.traversalInfoWriteCounter = uint(traversalTaskCounter);
  }
#if TARGETS_RASTERIZATION && USE_SW_RASTER
  else if (push.setup == BUILD_SETUP_DRAW)
  {
    // during traversal_run we might overshoot visibleClusterCounter  
    uint renderClusterCounter   = buildRW.renderClusterCounter;
    uint renderClusterCounterSW = buildRW.renderClusterCounterSW;
    
    // set drawindirect for actual rendered clusters
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
    //2
    // keep originals for array size warnings
    // use max if there is two passes
    // 之前：直接赋值（会被覆盖）
    //readback.numRenderClusters    = renderClusterCounter;
    //readback.numRenderClustersSW  = renderClusterCounterSW;
    // 现在：原子操作取最大值（两遍都能保留结果）
    atomicMax(readback.numRenderClusters, renderClusterCounter);
    atomicMax(readback.numRenderClustersSW, renderClusterCounterSW);
  #if USE_SEPARATE_GROUPS
  //    readback.numTraversalTasks    = max(buildRW.traversalInfoWriteCounter, buildRW.traversalGroupCounter);
    atomicMax(readback.numTraversalTasks, max(buildRW.traversalInfoWriteCounter, buildRW.traversalGroupCounter));
  #else
  //    readback.numTraversalTasks    = buildRW.traversalInfoWriteCounter;
    atomicMax(readback.numTraversalTasks, buildRW.traversalInfoWriteCounter);
  #endif
  //8
  #if USE_RENDER_STATS
    //之前：覆盖
    //readback.numRenderedClusters   = numRenderedClusters;
    //readback.numRenderedClustersSW = numRenderedClustersSW;
    //现在：累积
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
    // during traversal_run we might overshoot visibleClusterCounter  
    uint renderClusterCounter = buildRW.renderClusterCounter;
    
    // set drawindirect for actual rendered clusters
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

    // keep originals for array size warnings 
    // use max if there is two passes
    //    readback.numRenderClusters  = renderClusterCounter;
    atomicMax(readback.numRenderClusters, renderClusterCounter);
  #if USE_SEPARATE_GROUPS
  //    readback.numTraversalTasks  = max(buildRW.traversalInfoWriteCounter, buildRW.traversalGroupCounter);
    atomicMax(readback.numTraversalTasks, max(buildRW.traversalInfoWriteCounter, buildRW.traversalGroupCounter));
  #else
  //    readback.numTraversalTasks  = buildRW.traversalInfoWriteCounter;
    atomicMax(readback.numTraversalTasks, buildRW.traversalInfoWriteCounter);
  #endif

  #if USE_RENDER_STATS
  //    readback.numRenderedClusters = numRenderedClusters;
    readback.numRenderedClusters += numRenderedClusters;
    readback.numTraversedTasks   += buildRW.traversalInfoWriteCounter;
  #endif
  
  #if USE_TWO_PASS_CULLING
    setupSecondPass();
  #endif
  }
#endif
}