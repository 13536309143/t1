//==============================================================================
// 文件：shaders/streaming/stream_update_scene.comp.glsl
// 模块定位：流式加载着色器，维护 GPU 侧请求、驻留年龄、地址修补和任务压缩。
// 数据流：遍历产生请求后，这些 阶段 整理 request、筛选 unload 候选并更新 Geometry 组 地址。
// 方法说明：GPU 侧 流式加载 逻辑把可见性反馈转化为驻留集管理信号，从而实现按需几何加载。
// 正确性约束：最低细节层通常保持常驻；地址更新必须先于后续 traversal 解引用；load/unload 顺序要与 CPU 任务一致。
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
layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];
};


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


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(local_size_x=STREAM_UPDATE_SCENE_WORKGROUP) in;


// 函数：main。作为本着色器阶段入口，按绑定资源执行当前 GPU 工作。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该入口位于控制流根部，调用顺序决定后续资源生命周期和数据依赖。
void main()
{


  uint threadID = getGlobalInvocationIndex(gl_GlobalInvocationID);


  StreamingPatch spatch = streaming.update.patches.d[threadID];

  if (threadID < streaming.update.patchGroupsCount)
  {
    uint oldResidentID = 0;
    if (threadID < streaming.update.patchUnloadGroupsCount)
    {
      Group group = Group_in(geometries[spatch.geometryID].streamingGroupAddresses.d[spatch.groupID]).d;
      oldResidentID = group.residentID;
    }

    geometries[spatch.geometryID].streamingGroupAddresses.d[spatch.groupID] = spatch.groupAddress;

    if (threadID < streaming.update.patchUnloadGroupsCount)
    {
    #if STREAMING_DEBUG_ADDRESSES

      streaming.resident.groups.d[oldResidentID].group = Group_in(STREAMING_INVALID_ADDRESS_START);
    #endif
    }
    else
    {
      uint loadGroupIndex = threadID - streaming.update.patchUnloadGroupsCount;

      Group_in groupRef = Group_in(spatch.groupAddress);
      Group group = Group_in(groupRef).d;

      uint groupResidentID  = spatch.groupResidentID;
      groupRef.d.residentID = spatch.groupResidentID;
      groupRef.d.clusterResidentID = spatch.clusterResidentID;

      StreamingGroup residentGroup;
      residentGroup.geometryID   = spatch.geometryID;
      residentGroup.lodLevel     = spatch.lodLevel;

      residentGroup.age          = uint16_t(0);
      residentGroup.group        = groupRef;
    #if STREAMING_DEBUG_ADDRESSES
      if (uint64_t(streaming.resident.groups.d[groupResidentID].group) < STREAMING_INVALID_ADDRESS_START)
        streamingRW.request.errorUpdate = groupResidentID;
    #endif


      streaming.resident.groups.d[groupResidentID] = residentGroup;


      streaming.resident.groupIDs.d[groupResidentID] = spatch.groupID;


      streaming.resident.activeGroups.d[streaming.update.loadActiveGroupsOffset + loadGroupIndex] = groupResidentID;


      uint newBuildOffset = spatch.clasBuildOffset;
      for (uint c = 0; c < spatch.clusterCount; c++)
      {
        uint clusterResidentID = spatch.clusterResidentID + c;


        Cluster_in clusterRef = Cluster_in(spatch.groupAddress + Group_size + Cluster_size * c);

        streaming.resident.clusters.d[clusterResidentID] = uint64_t(clusterRef);
      }
    }
  }

}
