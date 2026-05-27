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
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#include "shaderio.h"

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
//layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;
/*
texHizFar[0]：前一帧的 HiZ
texHizFar[1]：当前帧建立的新 HiZ
*/
#if USE_TWO_PASS_CULLING
layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar[2];
#else
layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;
#endif

layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;  
};

layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) buffer buildBufferRW
{
  SceneBuilding buildRW;  
};

layout(local_size_x=TRAVERSAL_INIT_WORKGROUP) in;

#include "culling.glsl"
#include "traversal.glsl"

void main()
{
  uint instanceID   = getGlobalInvocationIndex(gl_GlobalInvocationID);
  uint instanceLoad = min(build.numRenderInstances-1, instanceID);
  bool isValid      = instanceID == instanceLoad;

#if USE_SORTING
  instanceLoad = build.instanceSortValues.d[instanceLoad];
  instanceID   = instanceLoad;
#endif

  RenderInstance instance = instances[instanceLoad];
  uint geometryID = instance.geometryID;
  Geometry geometry = geometries[geometryID];
  
  uint blasBuildIndex = BLAS_BUILD_INDEX_LOWDETAIL;
  
  vec4 clipMin;
  vec4 clipMax;
  bool clipValid;
  
#if USE_TWO_PASS_CULLING && TARGETS_RASTERIZATION
  //bool inFrustum = intersectFrustum(geometry.bbox.lo, geometry.bbox.hi, instance.worldMatrix, clipMin, clipMax, clipValid);
  //bool isVisible = inFrustum && (!clipValid || (intersectSize(clipMin, clipMax, 1.0) && intersectHiz(clipMin, clipMax)));
  // 第一遍 (pass == 0)：使用前一帧的 HiZ
  // 第二遍 (pass == 1)：使用当前帧的 HiZ
  bool inFrustum = intersectFrustum( build.pass == 0 ? build.cullViewProjMatrixLast : build.cullViewProjMatrix, geometry.bbox.lo, geometry.bbox.hi, instance.worldMatrix, clipMin, clipMax, clipValid);
  bool isVisible = inFrustum && (!clipValid || (intersectSize(clipMin, clipMax, 1.0) && intersectHiz(clipMin, clipMax, build.pass)));
    // 如果在第二遍且该实例已在第一遍渲染过，跳过它
  // if smallish and was already drawn, don't process again
  if (build.pass == 1 && isVisible && clipValid && !intersectSize(clipMin, clipMax, 8.0) && ((uint(build.instanceVisibility.d[instanceLoad]) & INSTANCE_VISIBLE_BIT) != 0)) {
    isVisible = false;
  }

// 单遍模式：总是使用前一帧数据
/*
第一遍 (pass == 0)
build.pass == 0
// 使用前一帧数据
bool inFrustum = intersectFrustum(viewLast.viewProjMatrix, ...);
bool isVisible = inFrustum && (!clipValid || 
                 (intersectSize(...) && intersectHiz(..., 0)));
// 第一遍无过滤，所有可见实例都会被渲染
if (inFrustum && isLargeEnough && notOccluded)
{
  isVisible = true;  // 渲染
}
第一遍的作用：
使用前一帧的视图矩阵（viewLast）
使用前一帧的 HiZ（texHizFar[0]）
渲染所有通过前一帧遮挡测试的实例
记录哪些实例被渲染（保存到 instanceVisibility）
第二遍 (pass == 1)
build.pass == 1
// 使用当前帧数据
bool inFrustum = intersectFrustum(view.viewProjMatrix, ...);
bool isVisible = inFrustum && (!clipValid || 
                 (intersectSize(...) && intersectHiz(..., 1)));
// 额外过滤：跳过已在第一遍渲染过的小实例
if (build.pass == 1 && isVisible && clipValid && 
    !intersectSize(clipMin, clipMax, 8.0) &&  // 大小 < 8 像素
    (instanceVisibility[instanceID] & VISIBLE_BIT) != 0)  // 已在第一遍渲染
{
  isVisible = false;  // 跳过，不再渲染
}
第二遍的作用：
使用当前帧的视图矩阵（view）
使用当前帧的新 HiZ（texHizFar[1]）
渲染第一遍未渲染的实例
但跳过小实例（已被第一遍处理）以避免重复
关键的过滤条件
if (build.pass == 1 && isVisible && clipValid && 
    !intersectSize(clipMin, clipMax, 8.0) && 
    ((uint(build.instanceVisibility.d[instanceLoad]) & INSTANCE_VISIBLE_BIT) != 0))
{
  isVisible = false;
}
    */

#else
  bool inFrustum = intersectFrustum(build.cullViewProjMatrixLast, geometry.bbox.lo, geometry.bbox.hi, instance.worldMatrix, clipMin, clipMax, clipValid);
  bool isVisible = inFrustum && (!clipValid || (intersectSize(clipMin, clipMax, 1.0) && intersectHiz(clipMin, clipMax, 0)));
#endif
  
  uint visibilityState = isVisible ? INSTANCE_VISIBLE_BIT : 0;
  
  bool isRenderable = isValid
  #if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)
    && isVisible
  #endif
    ;
    
  bool traverseRootNode = isRenderable;

  if (isRenderable)
  {
    // We test if we are only using the furthest lod.
    // If that is true, then we can skip lod traversal completely and
    // straight enqueue the lowest detail cluster directly.    
    
    uint rootNodePacked = geometry.nodes.d[0].packed;
    
    uint childOffset        = PACKED_GET(rootNodePacked, Node_packed_nodeChildOffset);
    uint childCountMinusOne = PACKED_GET(rootNodePacked, Node_packed_nodeChildCountMinusOne);
    
    // test if the second to last lod needs to be traversed
    uint childNodeIndex     = (childCountMinusOne > 1 ? (childCountMinusOne - 1) : 0);
    Node childNode          = geometry.nodes.d[childOffset + childNodeIndex];
    TraversalMetric traversalMetric = childNode.traversalMetric;
  
    mat4x3 worldMatrix = instances[instanceID].worldMatrix;
    float uniformScale = computeUniformScale(worldMatrix);
    float errorScale   = 1.0;
  
    mat4 transform = build.traversalViewMatrix * toMat4(worldMatrix);
  
    // if there is no need to traverse the pen ultimate lod level,
    // then just insert the last lod level node's cluster directly
    if (!testForTraversal(mat4x3(transform), uniformScale, traversalMetric, errorScale))
    {
    
    #if TARGETS_RASTERIZATION
      // lowest detail lod is guaranteed to have only one cluster
      
      uvec4 voteClusters = subgroupBallot(true); 
      
      uint offsetClusters = 0;
      if (subgroupElect())
      {
        offsetClusters = atomicAdd(buildRW.renderClusterCounter, int(subgroupBallotBitCount(voteClusters)));
      }
  
      offsetClusters = subgroupBroadcastFirst(offsetClusters);  
      offsetClusters += subgroupBallotExclusiveBitCount(voteClusters);
      
      if (offsetClusters < build.maxRenderClusters)
      {
        ClusterInfo clusterInfo;
        clusterInfo.instanceID = instanceID;
        clusterInfo.clusterID  = geometry.lowDetailClusterID;
        build.renderClusterInfos.d[offsetClusters] = clusterInfo;
      }
    #endif
      
      // we can skip adding the node for traversal
      traverseRootNode = false;
    }
  }

  uvec4 voteNodes = subgroupBallot(traverseRootNode);  
  
  uint offsetNodes = 0;
  if (subgroupElect())
  {
    offsetNodes = atomicAdd(buildRW.traversalTaskCounter, int(subgroupBallotBitCount(voteNodes)));
  }
  
  offsetNodes = subgroupBroadcastFirst(offsetNodes);  
  offsetNodes += subgroupBallotExclusiveBitCount(voteNodes);
      
  if (traverseRootNode && offsetNodes < build.maxTraversalInfos)
  {
    uint rootNodePacked = geometry.nodes.d[0].packed;

    TraversalInfo traversalInfo;
    traversalInfo.instanceID = instanceID;
    traversalInfo.packedNode = rootNodePacked;

    build.traversalNodeInfos.d[offsetNodes] = packTraversalInfo(traversalInfo);
  }
#if USE_TWO_PASS_CULLING && TARGETS_RASTERIZATION
  if (build.pass == 0 && isValid) {
    build.instanceVisibility.d[instanceID]                        = uint8_t(visibilityState);
  }
#endif
}
// 检查这个实例是否在第一遍已经渲染过
/*
输入：
  - viewLast（前一帧视图）
  - texHizFar[0]（前一帧 HiZ）

处理：
  1. 视锥剔除：使用 viewLast.viewProjMatrix
  2. 大小检查：投影 > 1 像素
  3. 遮挡剔除：使用 texHizFar[0]（前一帧深度）
  4. 渲染可见实例
  5. 保存可见性标志到 instanceVisibility[]

输出：
  - 渲染的实例
  - instanceVisibility[]（记录哪些实例被渲染）
  - 新的 HiZ（texHizFar[1]）

  输入：
  - view（当前帧视图）
  - texHizFar[1]（第一遍后建立的新 HiZ）
  - instanceVisibility[]（第一遍的结果）

处理：
  1. 视锥剔除：使用 view.viewProjMatrix
  2. 大小检查：投影 > 1 像素
  3. 遮挡剔除：使用 texHizFar[1]（第一遍后的深度）
  4. 过滤小实例：
     - 如果屏幕投影 < 8 像素
     - 且已在第一遍渲染
     - 则跳过（避免重复）
  5. 渲染未被渲染的实例

输出：
  - 补充渲染的实例（大实例或新可见实例）
  - 更新的 HiZ（texHizFar[1]）用于下一帧
  */