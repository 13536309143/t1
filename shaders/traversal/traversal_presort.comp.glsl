//==============================================================================
// 文件：shaders/traversal/traversal_presort.comp.glsl
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
layout(local_size_x=TRAVERSAL_PRESORT_WORKGROUP) in;


// 函数：main。作为本着色器阶段入口，按绑定资源执行当前 GPU 工作。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该入口位于控制流根部，调用顺序决定后续资源生命周期和数据依赖。
void main()
{

  uint instanceID   = getGlobalInvocationIndex(gl_GlobalInvocationID);

  uint instanceLoad = min(build.numRenderInstances-1, instanceID);

  RenderInstance instance = instances[instanceLoad];
  Geometry geometry = geometries[instance.geometryID];


  vec3 oPos = instance.worldMatrixI * vec4(view.viewPos.xyz,1);

  bool isInside = all(equal(greaterThanEqual(oPos, geometry.bbox.lo),lessThanEqual(oPos, geometry.bbox.hi)));

  vec3 oPosClamp = isInside ? (geometry.bbox.lo + geometry.bbox.hi) * 0.5 :

    clamp(oPos, geometry.bbox.lo, geometry.bbox.hi);


  vec3 wPos = instance.worldMatrix * vec4(oPosClamp, 1);

  if (instanceID == instanceLoad) {
    build.instanceSortValues.d[instanceID] = instanceID;
    build.instanceSortKeys.d[instanceID]   = floatBitsToUint(distance(wPos.xyz, view.viewPos.xyz));
  }
}
