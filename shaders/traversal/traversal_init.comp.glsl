//==============================================================================
// 文件：shaders/traversal/traversal_init.comp.glsl
// 模块定位：LOD 遍历着色器，负责从实例层次中选择本帧需要渲染或请求加载的 簇。
// 数据流：读取实例、几何层次、Hi-Z 和 流式加载 地址，输出 traversal queue、组 queue、render 簇 list 和 request。
// 方法说明：遍历阶段把屏幕空间误差、视锥剔除和遮挡剔除合并为并行剪枝问题，以减少后续光栅工作量。
// 正确性约束：队列计数必须原子更新；流式加载 地址无效时只能发请求，不能解引用；two-阶段 状态必须区分上一帧和当前帧 Hi-Z。
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
#extension GL_KHR_shader_subgroup_vote : require
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


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(local_size_x=TRAVERSAL_INIT_WORKGROUP) in;

#include "culling.glsl"
#include "traversal.glsl"


// 函数：main。作为本着色器阶段入口，按绑定资源执行当前 GPU 工作。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该入口位于控制流根部，调用顺序决定后续资源生命周期和数据依赖。
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


  bool inFrustum = intersectFrustum( build.pass == 0 ? build.cullViewProjMatrixLast : build.cullViewProjMatrix, geometry.bbox.lo, geometry.bbox.hi, instance.worldMatrix, clipMin, clipMax, clipValid);
  bool isVisible = inFrustum && (!clipValid || (intersectSize(clipMin, clipMax, 1.0) && intersectHiz(clipMin, clipMax, build.pass)));


  if (build.pass == 1 && isVisible && clipValid && !intersectSize(clipMin, clipMax, 8.0) && ((uint(build.instanceVisibility.d[instanceLoad]) & INSTANCE_VISIBLE_BIT) != 0)) {
    isVisible = false;
  }


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


    uint rootNodePacked = geometry.nodes.d[0].packed;


    uint childOffset        = PACKED_GET(rootNodePacked, Node_packed_nodeChildOffset);

    uint childCountMinusOne = PACKED_GET(rootNodePacked, Node_packed_nodeChildCountMinusOne);


    uint childNodeIndex     = (childCountMinusOne > 1 ? (childCountMinusOne - 1) : 0);
    Node childNode          = geometry.nodes.d[childOffset + childNodeIndex];
    TraversalMetric traversalMetric = childNode.traversalMetric;

    mat4x3 worldMatrix = instances[instanceID].worldMatrix;

    float uniformScale = computeUniformScale(worldMatrix);
    float errorScale   = 1.0;


    mat4 transform = build.traversalViewMatrix * toMat4(worldMatrix);


    if (!testForTraversal(mat4x3(transform), uniformScale, traversalMetric, errorScale))
    {

    #if TARGETS_RASTERIZATION


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
