//==============================================================================
// 文件：shaders/streaming/streaming.glsl
// 模块定位：流式加载着色器，维护 GPU 侧请求、驻留年龄、地址修补和任务压缩。
// 数据流：遍历产生请求后，这些 阶段 整理 request、筛选 unload 候选并更新 Geometry 组 地址。
// 方法说明：GPU 侧 流式加载 逻辑把可见性反馈转化为驻留集管理信号，从而实现按需几何加载。
// 正确性约束：最低细节层通常保持常驻；地址更新必须先于后续 traversal 解引用；load/unload 顺序要与 CPU 任务一致。
// 注释风格：使用中文解释 GPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
// 函数：streamingAgeFilter。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void streamingAgeFilter(uint residentID, uint geometryID, Group_in groupRef, bool useBlasCaching)
{
#if STREAMING_DEBUG_ADDRESSES
  if (uint64_t(groupRef) >= STREAMING_INVALID_ADDRESS_START)
  {
    streamingRW.request.errorAgeFilter = residentID;
    return;
  }
#endif


  uint age = streaming.resident.groups.d[residentID].age;

  if (useBlasCaching)
  {
    uint lodLevel    = streaming.resident.groups.d[residentID].lodLevel;
    uint cachedLevel = build.geometryBuildInfos.d[geometryID].cachedLevel;


    if (lodLevel >= cachedLevel) {
      age = 0;
    }
  }

  if (age < 0xFFFF)
  {
    age++;

    streaming.resident.groups.d[residentID].age = uint16_t(age);
  }


  if (age > streaming.ageThreshold)
  {

    uint unloadOffset = atomicAdd(streamingRW.request.unloadCounter, 1);
    if (unloadOffset <= streaming.request.maxUnloads) {

      streaming.request.unloadGeometryGroups.d[unloadOffset] = uvec2(geometryID, streaming.resident.groupIDs.d[residentID]);
    }
  }
}
