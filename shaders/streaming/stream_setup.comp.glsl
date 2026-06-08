//==============================================================================
// 文件：shaders/streaming/stream_setup.comp.glsl
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
layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) coherent buffer streamingBufferRW
{
  SceneStreaming streamingRW;
};


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(local_size_x=1) in;


// 函数：main。作为本着色器阶段入口，按绑定资源执行当前 GPU 工作。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该入口位于控制流根部，调用顺序决定后续资源生命周期和数据依赖。
void main()
{
  if (push.setup == STREAM_SETUP_COMPACTION_OLD_NO_UNLOADS)
  {


    if (streaming.frameIndex == 1)
    {

      streaming.resident.clasCompactionUsedSize.d[0] = 0;
      streamingRW.update.moveClasSize = 0;
    }
    else {
      streamingRW.update.moveClasSize = streaming.resident.clasCompactionUsedSize.d[0];
    }
  }
  else if (push.setup == STREAM_SETUP_COMPACTION_STATUS)
  {

    if (streaming.update.patchGroupsCount > 0) {

      streaming.resident.clasCompactionUsedSize.d[0] = streamingRW.update.moveClasSize;

      streamingRW.request.clasCompactionUsedSize = streamingRW.update.moveClasSize;
      streamingRW.request.clasCompactionCount    = streamingRW.update.moveClasCounter;
    }
    else {

      streamingRW.request.clasCompactionUsedSize = streaming.resident.clasCompactionUsedSize.d[0];
      streamingRW.request.clasCompactionCount    = 0;
    }
  }
  else if (push.setup == STREAM_SETUP_ALLOCATOR_FREEINSERT)
  {
    uint freeGaps = streamingRW.clasAllocator.freeGapsCounter;
    uint maxFreeGaps = (streaming.clasAllocator.sectorCount << streaming.clasAllocator.sectorSizeShift);


    streamingRW.clasAllocator.freeGapsCounter = 0;


    uint workGroupCount = (min(freeGaps,maxFreeGaps) + STREAM_ALLOCATOR_FREEGAPS_INSERT_WORKGROUP -1) / STREAM_ALLOCATOR_FREEGAPS_INSERT_WORKGROUP;
  #if USE_16BIT_DISPATCH

    uvec3 grid = fit16bitLaunchGrid(workGroupCount);
    streamingRW.clasAllocator.dispatchFreeGapsInsert.gridX = grid.x;
    streamingRW.clasAllocator.dispatchFreeGapsInsert.gridY = grid.y;
    streamingRW.clasAllocator.dispatchFreeGapsInsert.gridZ = grid.z;
  #else
    streamingRW.clasAllocator.dispatchFreeGapsInsert.gridX = workGroupCount;
  #endif
  #if STREAMING_DEBUG_USEDBITS_COUNT

    uint64_t allocatedSize = streaming.clasAllocator.stats.d.allocatedSize;
    if (streaming.clasAllocator.usedBitsCount > 0 &&
        allocatedSize != uint64_t(streaming.clasAllocator.usedBitsCount) << streaming.clasAllocator.granularityByteShift)
    {

      streamingRW.request.errorClasUsedVsAlloc = int(allocatedSize >> streaming.clasAllocator.granularityByteShift) - int(streaming.clasAllocator.usedBitsCount);
    }
  #endif
  }
  else if (push.setup == STREAM_SETUP_ALLOCATOR_STATUS)
  {
    if (streaming.frameIndex == 1)
    {

      uint clasAllocatedMaxSizedLeft = streaming.clasAllocator.sectorMaxAllocationSized * streaming.clasAllocator.sectorCount;
      streaming.resident.clasAllocatedMaxSizedLeft.d[0] = clasAllocatedMaxSizedLeft;
      streamingRW.request.clasAllocatedMaxSizedLeft     = clasAllocatedMaxSizedLeft;
    #if USE_MEMORY_STATS
      streaming.clasAllocator.stats.d.allocatedSize = 0;
      streaming.clasAllocator.stats.d.wastedSize    = streaming.clasAllocator.baseWastedSize << streaming.clasAllocator.granularityByteShift;
    #endif
    }
    else {

      if (streaming.update.patchGroupsCount > 0) {

        uint clasAllocatedMaxSizedLeft = uint(max(0,streaming.clasAllocator.freeSizeRanges.d[streaming.clasAllocator.maxAllocationSize-1].count));
        streaming.resident.clasAllocatedMaxSizedLeft.d[0] = clasAllocatedMaxSizedLeft;
        streamingRW.request.clasAllocatedMaxSizedLeft     = clasAllocatedMaxSizedLeft;
      }
      else {

        streamingRW.request.clasAllocatedMaxSizedLeft = streaming.resident.clasAllocatedMaxSizedLeft.d[0];
      }
    }
  #if USE_MEMORY_STATS
    streamingRW.request.clasAllocatedUsedSize   = streaming.clasAllocator.stats.d.allocatedSize;
    streamingRW.request.clasAllocatedWastedSize = streaming.clasAllocator.stats.d.wastedSize;
  #endif
  }
}
