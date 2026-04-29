//当 USE_SEPARATE_GROUPS = false 时，它不仅处理内部节点，连最底层的微网格簇（Clusters）也一并处理；但当 USE_SEPARATE_GROUPS = true 时，它被限制为仅处理树的内部节点。
//负责遍历庞大的 BVH（层次包围盒）树结构
//主要做出“粗粒度决策”，通过视锥体和遮挡剔除快速剔除掉不可见的大片区域（Nodes）。
#version 460
// 启用各种扩展，支持 8/16/32/64 位整数运算、物理指针(Buffer Reference)以及 64 位原子操作
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
// 核心扩展：Subgroup（子组 / Warp / Wavefront）级别的内置指令，用于极速的线程间通信和规约
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_memory_scope_semantics : require

#include "shaderio.h"
// 绑定各种全局状态缓冲 (UBO/SSBO)，用于读取摄像机矩阵、实例数据和几何体数据
layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer{FrameConstants view;};
layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer{Readback readback;};
layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer{RenderInstance instances[];};
layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer{Geometry geometries[];};
//两遍剔除需要访问两个不同的 HiZ 纹理// 绑定 Hi-Z 层级深度纹理。如果是两遍剔除（Two-Pass），则绑定一个数组（0存上一帧，1存当前帧新深度）
#if USE_TWO_PASS_CULLING && TARGETS_RASTERIZATION
layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar[2];// 0 存上一帧深度，1 存当前帧新深度
#else
layout(binding = BINDINGS_HIZ_TEX)  uniform sampler2D texHizFar;// 单遍只用一个深度图
#endif
// 绑定遍历过程的输入/输出队列 (SSBO)，包括只读和可读写缓冲
layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer{SceneBuilding build; };
layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) coherent buffer buildBufferRW{volatile SceneBuilding buildRW;  };
#if USE_STREAMING
// 绑定流式加载队列，用于向系统请求缺失的 Mesh
layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer{SceneStreaming streaming;};
layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) buffer streamingBufferRW{SceneStreaming streamingRW;};
#endif
// 定义 Compute Shader 的每个 Workgroup 的线程数
layout(local_size_x=TRAVERSAL_RUN_WORKGROUP) in;
#include "culling.glsl"
#include "traversal.glsl"
// 修复老旧驱动对 volatile 和 coherent 支持不佳的问题，强制使用原子级加载/存储
#define USE_ATOMIC_LOAD_STORE 1
////////////////////////////////////////////
// Computes the number of children for an incoming node / group task.
// These children are then processed within `processSubTask` a few lines down
// 计算当前从队列中取出的任务（节点或组）包含了多少个子元素
uint setupTask(inout TraversalInfo traversalInfo, uint readIndex, uint pass)
{
  uint subCount = 0;
// 解析压缩的 64-bit 节点数据，判断当前是个内部节点(Node)还是叶子组(Group)
  bool isNode = PACKED_GET(traversalInfo.packedNode, Node_packed_isGroup) == 0;
  if (isNode) {
  // 如果是节点，读取它有多少个子节点
    subCount = PACKED_GET(traversalInfo.packedNode, Node_packed_nodeChildCountMinusOne);
  }
  else {
  // 如果是组，读取它包含多少个需要被渲染的微网格集群
    subCount = PACKED_GET(traversalInfo.packedNode, Node_packed_groupClusterCountMinusOne);
  }
  // 加 1 是因为为了省 bit，存储时存的是 "数量减一" (如存 0 代表 1 个)
  return subCount + 1;
}

#if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)
// simplified occlusion culling based on last frame's depth buffer
// bool queryWasVisible(mat4x3 instanceTransform, BBox bbox)
// 函数：queryWasVisible - 查询当前物体的包围盒是否在屏幕内并且没有被挡住
bool queryWasVisible(mat4x3 instanceTransform, BBox bbox, bool isNode)
{
  vec3 bboxMin = bbox.lo;
  vec3 bboxMax = bbox.hi;
  vec4 clipMin;
  vec4 clipMax;
  bool clipValid;
  //  bool useOcclusion = true;
  //  bool inFrustum = intersectFrustum(bboxMin, bboxMax, instanceTransform, clipMin, clipMax, clipValid);
  //  硬编码使用 viewLast 和单个 texHizFar，无法支持两遍剔除
#if USE_TWO_PASS_CULLING
  #if USE_SEPARATE_GROUPS
    isNode = true;
  #endif
    // --- 双遍剔除策略核心 ---
  // 策略：如果是真正要画的 Cluster 簇，总是对比前一帧的 HiZ（保障帧间连续性）。
    //      如果是内部节点 Node，第一遍对前一帧，第二遍对最新帧。
    bool useLast = !isNode || build.pass == 0;
    // 视锥剔除（Frustum Culling）：看看包围盒在不在摄像机视角内
    bool inFrustum = intersectFrustum(useLast ? build.cullViewProjMatrixLast : build.cullViewProjMatrix, bboxMin, bboxMax, instanceTransform, clipMin, clipMax, clipValid);
    // 遮挡剔除（Occlusion Culling）：查询硬件层级深度缓存(Hi-Z)
    bool isVisible = inFrustum && 
      //(!useOcclusion || !clipValid || (intersectSize(clipMin, clipMax, 1.0) && intersectHiz(clipMin, clipMax)));
      (!clipValid || (intersectSize(clipMin, clipMax, 1.0) && intersectHiz(clipMin, clipMax, useLast ? 0 : 1)));
    
  #if !USE_SEPARATE_GROUPS
    // 簇在两遍中都要测试
    // 针对 Cluster 在第二遍 (pass == 1) 时的特殊逻辑：
    if (!isNode && build.pass == 1) 
    {
      if (isVisible) {
        // 如果上一帧它是可见的，说明第一遍已经渲染过了，这里直接返回 false 避免重复画
        isVisible = false;
      }
      else {
        // 如果第一遍（上一帧）没看到它，可能它这一帧因为相机移动露出来了。重新用最新视锥矩阵测试。
        inFrustum = intersectFrustum(build.cullViewProjMatrix, bboxMin, bboxMax, instanceTransform, clipMin, clipMax, clipValid);
        // 用最新的 HiZ (索引1) 测试它是否真实露出来了
        isVisible = inFrustum && 
          (!clipValid || (intersectSize(clipMin, clipMax, 1.0) && intersectHiz(clipMin, clipMax, 1)));
      }
    }
  #endif
#else
  // 单遍模式：总是对比前一帧
  bool inFrustum = intersectFrustum(build.cullViewProjMatrixLast, bboxMin, bboxMax, instanceTransform, clipMin, clipMax, clipValid);
  bool isVisible = inFrustum && 
    (!clipValid || (intersectSize(clipMin, clipMax, 1.0) && intersectHiz(clipMin, clipMax, 0)));
#endif
  return isVisible;
}
#endif
/*
两遍剔除模式的逻辑
#if USE_TWO_PASS_CULLING
  bool useLast = !isNode || build.pass == 0;
useLast 的取值：
isNode	pass	useLast	含义	HiZ 索引
false (簇)	0	true	第一遍簇用前一帧	0
false (簇)	1	true	第二遍簇仍用前一帧	0
true (节点)	0	true	第一遍节点用前一帧	0
true (节点)	1	false	第二遍节点用当前帧	1
策略解释：
簇（cluster）：始终对比前一帧 HiZ（texHizFar[0]）
簇是叶子节点，是实际渲染的单位
需要与上一帧的渲染结果进行比较（temporal coherence）
节点（node）：
第一遍对比前一帧 HiZ
第二遍对比当前帧 HiZ（更新后的）
节点只用于剔除判断，不实际渲染
第二遍的簇双测试
#if !USE_SEPARATE_GROUPS
  if (!isNode && build.pass == 1)  // 只针对簇，只在第二遍
  {
    if (isVisible) {
      // 在第一遍中通过测试
      // → 已经渲染过，跳过
      isVisible = false;
    }
    else {
      // 在第一遍中未通过测试（对比 texHizFar[0]）
      // → 在第二遍中用更新的 HiZ（texHizFar[1]）重新测试
      inFrustum = intersectFrustum(view.viewProjMatrix, ...);
      isVisible = inFrustum && 
        (!clipValid || (intersectSize(...) && 
                        intersectHiz(..., 1)));  // 用索引 1
    }
  }
如果启用 USE_SEPARATE_GROUPS（分离处理），跳过这个逻辑
否则，对每个簇进行双测试：
第一遍已测试（against texHizFar[0]）→ 标记为已渲染 → 跳过
第一遍未测试（未通过 texHizFar[0]）→ 用 texHizFar[1] 重新测试
为什么需要这个：
第一遍使用前一帧的 HiZ，可能因为遮挡而漏掉某些簇
第二遍使用最新的 HiZ（刚在第一遍之后更新），能准确检测现在是否可见
如果簇通过第二遍测试，就会在第二遍被渲
*/
// 函数：processSubTask - 处理每一个被拆解开的具体任务
void processSubTask(const TraversalInfo subgroupTasks, uint taskID, uint taskSubID, bool isValid, uint threadReadIndex, uint pass)
{
// 利用 Subgroup Shuffle 从当前 Warp 中的源线程高速提取原始任务信息（不走内存）
  TraversalInfo traversalInfo;
  traversalInfo.instanceID               = subgroupShuffle(subgroupTasks.instanceID, taskID);
  traversalInfo.packedNode               = subgroupShuffle(subgroupTasks.packedNode, taskID);
  // ... 提取几何体信息等 ...
  uint instanceID     = traversalInfo.instanceID;
  bool forceCluster   = false;
// 再次确认是节点还是叶子
  bool isNode  = PACKED_GET(traversalInfo.packedNode, Node_packed_isGroup) == 0;
#if USE_SEPARATE_GROUPS
  isNode = true;
#endif

  uint geometryID   = instances[instanceID].geometryID;
  Geometry geometry = geometries[geometryID];
  // 【获取度量误差和包围盒】
  TraversalMetric traversalMetric;
#if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)
  BBox bbox;
#endif

  if (isNode)
  {
    // 如果是节点，读取当前子节点的具体数据
    // 读取子节点的包围盒和误差信息
    uint childIndex     = taskSubID;
    uint childNodeIndex = PACKED_GET(traversalInfo.packedNode, Node_packed_nodeChildOffset) + childIndex;
    Node childNode      = geometry.nodes.d[childNodeIndex];
    traversalMetric     = childNode.traversalMetric;// 屏幕空间误差阈值
  #if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)
    bbox = geometry.nodeBboxes.d[childNodeIndex];// 获取包围盒用于剔除
  #endif
// 准备将这个子节点信息存入结构，如果它后面需要继续遍历的话
    traversalInfo.packedNode = childNode.packed;
  }
#if !USE_SEPARATE_GROUPS
  else {
    // 如果是集群组，则测试其内部具体的微网格 (Cluster)
    uint clusterIndex = taskSubID;
    uint groupIndex   = PACKED_GET(traversalInfo.packedNode, Node_packed_groupIndex);
    
  #if USE_STREAMING
    // 通过指针读取流式组的实际内存地址
    Group_in groupRef = Group_in(geometry.streamingGroupAddresses.d[groupIndex]);
    Group group = groupRef.d;
  #else
    Group_in groupRef = Group_in(geometry.preloadedGroups.d[groupIndex]);
    Group group = groupRef.d;
  #endif
  #if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)
    bbox        = Group_getClusterBBox(groupRef, clusterIndex);// 获取细粒度微网格包围盒
  #endif
    uint32_t clusterGeneratingGroup = Group_getGeneratingGroup(groupRef, clusterIndex);
  #if USE_STREAMING
    if (clusterGeneratingGroup != SHADERIO_ORIGINAL_MESH_GROUP
        && geometry.streamingGroupAddresses.d[clusterGeneratingGroup] < STREAMING_INVALID_ADDRESS_START)
    {
      // 如果它是由更高级的 LOD 生成的，读取生成组的误差
traversalMetric = Group_in(geometry.streamingGroupAddresses.d[clusterGeneratingGroup]).d.traversalMetric;
    }
  #else
    if (clusterGeneratingGroup != SHADERIO_ORIGINAL_MESH_GROUP)
    {
      traversalMetric = Group_in(geometry.preloadedGroups.d[clusterGeneratingGroup]).d.traversalMetric;
}
  #endif
    else {
    // 找不到生成组（本身就是最高精度），强制渲染它
      traversalMetric = group.traversalMetric;
      forceCluster    = true;
    }
    // 把该集群打包
    traversalInfo.packedNode = group.clusterResidentID + clusterIndex;
  }
#endif
  









// ================= 【核心遍历与剔除逻辑】 =================
  mat4x3 worldMatrix = instances[instanceID].worldMatrix;
  float uniformScale = computeUniformScale(worldMatrix);// 获取缩放
  float errorScale   = 1.0;
#if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)
//  isValid            = isValid && queryWasVisible(worldMatrix, bbox);
//区分节点还是簇
// 【核心评估：LOD 与 剔除】
// 1. 剔除测试：如果不看它，直接标记无效
  isValid            = isValid && queryWasVisible(worldMatrix, bbox, isNode);
#endif
// 2. 误差测试：投影误差是否超过了我们设定的像素阈值？超了就要继续细化 (Traverse)
  bool traverse      = testForTraversal(mat4x3(build.traversalViewMatrix * toMat4(worldMatrix)), uniformScale, traversalMetric, errorScale);
  
  // 计算LOD过渡因子
  float lodTransitionFactor = 0.0;
  if (isNode) {
    // 对于节点，我们可以计算与子节点之间的过渡
    // 这里简化处理，实际实现可能需要访问子节点的metric
  }
  
  // 如果可见、是内部节点、且误差太大需要继续细分 -> 当前子节点入列，继续遍历（// 如果可见 + 是节点 + 误差太大 -> 加入队列继续下探）
  bool traverseNode  = isValid && isNode && (traverse);
  // 如果可见、已经是叶子了(或误差满足要求了) -> 发送到渲染队列（// 如果可见 + 是集群 + (误差达标或者别无选择) -> 输出准备渲染）
  bool renderCluster = isValid && !isNode && (!traverse || forceCluster);  
  bool isGroup = false;
#if USE_STREAMING || USE_SEPARATE_GROUPS
  if (traverseNode)
  {
    // ================= 【流式加载判定】 =================
    // 当我们需要渲染一个极其精细的 LOD 模型时，需要检查显存里有没有加载
         isGroup    = PACKED_GET(traversalInfo.packedNode, Node_packed_isGroup) != 0;
    uint groupIndex = PACKED_GET(traversalInfo.packedNode, Node_packed_groupIndex);
  
  #if USE_STREAMING
  // 地址包含了两个信息：正常指针，或者是无效标志+上次请求时间
    if (isGroup)
    {
      uint64_t groupAddress = geometry.streamingGroupAddresses.d[groupIndex];// 取地址
      if (groupAddress >= STREAMING_INVALID_ADDRESS_START) {
        // [没在显存里！] 停止下探当前节点，退回到上一个粗糙版本进行渲染！// [没在显存里！] 停止下探当前节点，退回到上一个粗糙版本进行渲染！
        traverseNode = false;
        {
          // 检查同一帧有没有别人已经帮我发过求救信号了？
          uint64_t lastRequestFrameIndex = atomicMax(geometry.streamingGroupAddresses.d[groupIndex], streaming.request.frameIndex);
          // 没人发过，我来发
          bool triggerRequest = lastRequestFrameIndex != streaming.request.frameIndex;
          // Subgroup Ballot 把当前 Warp 里的所有请求打包，只利用 1 次全局 Atomic Add 索要内存！
          uvec4 voteRequested  = subgroupBallot(triggerRequest);
          uint  countRequested = subgroupBallotBitCount(voteRequested);
          uint offsetRequested = 0;
          if (subgroupElect()) {
                 offsetRequested = atomicAdd(streamingRW.request.loadCounter, countRequested);// 获取全局队列索引
               }
          offsetRequested = subgroupBroadcastFirst(offsetRequested);// 把拿到的地址发给全员
          offsetRequested += subgroupBallotExclusiveBitCount(voteRequested);
          if (triggerRequest && offsetRequested <= streaming.request.maxLoads) {
          // 正式把缺失的 mesh ID 写入加载请求队列！让 CPU 或者 IO 线程去拉数据
                 streaming.request.loadGeometryGroups.d[offsetRequested] = uvec2(geometryID, groupIndex);
               }
        }
      }
    }
  #endif
  }
#endif
#if USE_SEPARATE_GROUPS
  renderCluster = isValid && traverseNode && isGroup;
  if (renderCluster)
    traverseNode = false;
#endif
// ================= 【写入输出队列】 =================
  // 【使用 Subgroup 优化原子写入】
  // 如果让每个线程自己去 atomicAdd 获取全局队列的写入位置，效率极低（严重竞争）。
  // 这里使用 Ballot 投票，统计当前 Warp（32个线程）里有多少个需要入队的 Node，多少个需要渲染的 Cluster。
  uvec4 voteNodes = subgroupBallot(traverseNode);
  uint countNodes = subgroupBallotBitCount(voteNodes);// 统计总数
  uvec4 voteClusters = subgroupBallot(renderCluster);
  uint countClusters = subgroupBallotBitCount(voteClusters);
  uint offsetNodes    = 0;
  uint offsetClusters = 0;
  // Subgroup 中序号最小的活跃线程当代表
  // 只有 Warp 中的第一个活跃线程 (Elect) 去全局显存执行一次 Atomic 操作，申请一整块连续内存！
  if (subgroupElect())
  {
    // increase global task counter
    atomicAdd(buildRW.traversalTaskCounter, int(countNodes));// 增加待处理任务数
    // get memory offsets
    offsetNodes    = atomicAdd(buildRW.traversalInfoWriteCounter, countNodes);
  #if USE_SEPARATE_GROUPS
    offsetClusters = atomicAdd(buildRW.traversalGroupCounter, countClusters);
  #else
    offsetClusters = atomicAdd(buildRW.renderClusterCounter, countClusters);
  #endif
  }
  memoryBarrierBuffer();
  // 把申请到的基地址广播给 Warp 内的所有线程，每个线程加上自己的局部偏移
  offsetNodes = subgroupBroadcastFirst(offsetNodes);
  offsetNodes += subgroupBallotExclusiveBitCount(voteNodes);
  offsetClusters = subgroupBroadcastFirst(offsetClusters);
  offsetClusters += subgroupBallotExclusiveBitCount(voteClusters);
// 安全检查边界防溢出
  traverseNode  = traverseNode && offsetNodes < build.maxTraversalInfos;
  
#if USE_SEPARATE_GROUPS
  renderCluster = renderCluster && offsetClusters < build.maxTraversalInfos;
#else
  renderCluster = renderCluster && offsetClusters < build.maxRenderClusters;
#endif
// 判断自己是否有需要写的活
  bool doStore = traverseNode || renderCluster;
  if (doStore)
  {
    uint writeIndex          = traverseNode ? offsetNodes : offsetClusters;// 选定内存槽
    uint64s_coh writePointer = uint64s_coh(traverseNode ? uint64_t(build.traversalNodeInfos) 
#if USE_SEPARATE_GROUPS
    : uint64_t(build.traversalGroupInfos)
#else
    : uint64_t(build.renderClusterInfos)
#endif
    );
  #if USE_ATOMIC_LOAD_STORE
  // 写入！完成一次循环！
    atomicStore(writePointer.d[writeIndex], packTraversalInfo(traversalInfo), gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
  #else
    writePointer.d[writeIndex] = packTraversalInfo(traversalInfo);
  #endif
    // 优化：只在需要时使用内存屏障
    // 由于我们使用了原子操作或顺序写入，这里可以减少内存屏障
  }
}
// ================= 【Warp 级任务打平（Flattening）算法】 =================
// 为什么需要这个？如果每个线程直接跑循环处理自己的孩子们，有人有 4 个娃，有人 0 个，就会导致严重的线程空转（SIMT Divergence）。
struct TaskInfo {
  uint taskID;
};
shared TaskInfo s_tasks[TRAVERSAL_RUN_WORKGROUP];
// 如果三个线程分别拥有 4，2，1 个子节点需要处理，传统的 for 循环会导致线程执行时间极度不均（Divergence）。
// 这个函数通过 Subgroup 指令，将 4+2+1 = 7 个任务平铺开，分配给 Warp 里的前 7 个线程【并行】执行，一轮搞定！
void processAllSubTasks(inout TraversalInfo traversalInfo, bool threadRunnable, int threadSubCount, uint threadReadIndex, uint pass)
{
  // This algorithm is described in detail in `vk_tessellated_clusters/shaders/render_raster_clusters_batched.mesh.glsl`
  // 计算当前 Warp 总共需要处理多少个子节点 (Inclusive/Exclusive Scan)
  // 把所有人所有的子任务数量全部累加（Scan），算出一个连续的分布带
  int endOffset    = subgroupInclusiveAdd(threadSubCount);
  int startOffset  = endOffset - threadSubCount;
  int totalThreads = subgroupShuffle(endOffset, SUBGROUP_SIZE - 1);// 这个 Warp 总共产生了几十个小任务？
  // 计算需要 Warp 整体运行多少轮 (比如总计 40 个子任务，Warp 只有 32 个线程，就需要跑 2 轮)
  int totalRuns    = (totalThreads + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;
  const uint subgroupOffset = gl_SubgroupID * gl_SubgroupSize;
  //代码中利用了 gl_Subgroup* 内置指令，在整个 Warp 内重新分配和压实（Compact）子任务。这极大提高了 SIMT 执行效率。
  bool hasTask     = threadSubCount > 0;
  uvec4 taskVote   = subgroupBallot(hasTask);
  uint taskCount   = subgroupBallotBitCount(taskVote);
  uint taskOffset  = subgroupBallotExclusiveBitCount(taskVote);
  // 将当前任务的原始 ID 写入 Shared Memory，方便扁平化后的线程查找自己应该处理哪个任务的数据
  // 记录任务归属表，放到 shared memory 里
  if (hasTask) {
    s_tasks[subgroupOffset + taskOffset].taskID = gl_SubgroupInvocationID;
  }
  memoryBarrierShared();
  uint sumEnqueues = 0;
  // 开始以扁平化（每个线程必满载）的方式迭代
  int taskBase = -1;
  for (int r = 0; r < totalRuns; r++)
  {
  // ... 极其复杂的位掩码计算，将扁平化后的线性索引 t 还原回原始的任务 ID 和 子任务 ID ...以下为极限位掩码计算，意图是将当前绝对的循环编号 t 还原回属于哪位老爹的哪个孩子
    int tFirst = r * SUBGROUP_SIZE;
    int t      = tFirst + int(gl_SubgroupInvocationID);
    int  relStart      = startOffset - tFirst;
#if SUBGROUP_SIZE > 32
    uvec2 startBits    = subgroupOr(unpack32(threadRunnable && relStart >= 0 && relStart < SUBGROUP_SIZE ? (uint64_t(1) << relStart) : uint64_t(0)));
    int  task          = bitCount(startBits.x & gl_SubgroupLeMask.x) + bitCount(startBits.y & gl_SubgroupLeMask.y) + taskBase;
#else
    uint startBits     = subgroupOr(threadRunnable && relStart >= 0 && relStart < SUBGROUP_SIZE ? (1 << relStart) : 0);
    int  task          = bitCount(startBits & gl_SubgroupLeMask.x) + taskBase;
#endif
    uint taskID        = s_tasks[subgroupOffset + task].taskID;// 查表找
    uint taskSubID     = t - subgroupShuffle(startOffset, taskID);// 算出我是第几个儿子
    uint taskSubCount  = subgroupShuffle(threadSubCount, taskID);
    #if 0
#else
    uint taskReadIndex = 0;
  #endif
    taskBase           = subgroupShuffle(task, SUBGROUP_SIZE - 1);
    bool taskValid     = taskSubID < taskSubCount;
    // 所有线程整齐划一地调用处理逻辑，无分支发散！
    processSubTask(traversalInfo, taskID, min(taskSubID,taskSubCount-1), taskValid, taskReadIndex, pass);
  }
}

// ================= 【入口点：持久化循环】 =================
//树状层次结构
void run()
{    
  // 生产者-消费者主循环
  // 这个 Kernel 启动后不会轻易结束，而是变成持久化线程（Persistent Threads），一直驻留在 GPU 上
  uint threadReadIndex = ~0;
  for(uint pass = 0; ; pass++)// 死循环监听队列
  {
  // 如果当前 Warp 没有工作了，就从全局任务队列去拿
    if (subgroupAll(threadReadIndex == ~0)) {
      // pull new work
      if (subgroupElect()){
      // 从全局队列一次切下 32 个任务供大家分配
        threadReadIndex = atomicAdd(buildRW.traversalInfoReadCounter, SUBGROUP_SIZE);
      }
      threadReadIndex = subgroupBroadcastFirst(threadReadIndex) + gl_SubgroupInvocationID; 
      threadReadIndex = threadReadIndex >= build.maxTraversalInfos ? ~0 : threadReadIndex;
    // 发现确实没活干了，直接退出线程
      if (subgroupAll(threadReadIndex == ~0)){
        break;
      }
    }
    bool threadRunnable = false;
    TraversalInfo nodeTraversalInfo;
    // 【等待数据同步】因为前面是原子计数申请，可能生成数据的人还没来得及写进内存，需要等待信号释放
    while(true)
    {
      if (threadReadIndex != ~0)
      {
    // 不停去嗅探对应地址是不是变成了有效的打包数据
      #if USE_ATOMIC_LOAD_STORE
        uint64_t rawValue = atomicLoad(build.traversalNodeInfos.d[threadReadIndex], gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
      #else
        // 优化：减少内存屏障，使用有条件的内存屏障
        memoryBarrierBuffer();
        uint64_t rawValue = build.traversalNodeInfos.d[threadReadIndex];
      #endif
        nodeTraversalInfo = unpackTraversalInfo(rawValue);
        // 如果不全为最大无符号数，说明数据写完了，这块肉熟了可以吃了
        threadRunnable    = nodeTraversalInfo.instanceID != ~0u && nodeTraversalInfo.packedNode != ~0u;
      }
      // 有任何一个人拿到肉了，就冲出去
      if (subgroupAny(threadRunnable))
        break;
        // 所有人都没拿到，有可能确实队列空了但计数器被占了，二次检查是不是总任务数归零了
      #if USE_ATOMIC_LOAD_STORE
        bool isEmpty = atomicLoad(buildRW.traversalTaskCounter, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire) == 0;
      #else
        memoryBarrierBuffer();
        bool isEmpty = buildRW.traversalTaskCounter == 0;
      #endif
      if (subgroupAny(isEmpty))
      {
        return;// 全局确实没活干了，散会下班
      }
    }
    if (subgroupAny(threadRunnable))
    {
      int threadSubCount = 0;
      if (threadRunnable)
      // 开始工作前，算出这个包裹里装了几个下级子包？
      {
        threadSubCount = int(setupTask(nodeTraversalInfo, threadReadIndex, pass));
      }
      // 发动之前定义的 Warp 打平神技，解决所有人任务数不一的问题，集体消化
      processAllSubTasks(nodeTraversalInfo, threadRunnable, threadSubCount, threadReadIndex, pass);
    #if USE_TWO_PASS_CULLING && TARGETS_RASTERIZATION
      // 第一遍如果结束了，把老数据标记为已处理，准备在下一个 pass 进行第二遍新深度测试
      if (build.pass == 0 && threadRunnable) {
        build.traversalNodeInfos.d[threadReadIndex] = uint64_t(packUint2x32(uvec2(~0, ~0)));
      }
    #endif
      // 做完手里的活之后，从全局任务池里核销。
      uint numRunnable = subgroupBallotBitCount(subgroupBallot(threadRunnable));
      if (subgroupElect()) {
        atomicAdd(buildRW.traversalTaskCounter, -int(numRunnable));
      }
      if (threadRunnable) {
      // 重置索引，宣告自己当前这轮空闲了，准备在死循环里接下一单
        threadReadIndex = ~0;
      }
    }
  }
}
void main()
{
  run();// 启动常驻跑团引擎
  // 分离 Group 处理模式下，通过第一号线程设置间接分发的 Grid 大小，用于后续 Compute Shader 的调用。
#if USE_SEPARATE_GROUPS
  uint threadID = getGlobalInvocationIndex(gl_GlobalInvocationID);
  if (threadID == 0) {
    // this sets up the grid for `traversal_run_separate_groups.comp.glsl`
    uint groupCount = atomicAdd(buildRW.traversalGroupCounter,0);
    groupCount = min(groupCount,build.maxTraversalInfos);
    uint workGroupCount = (groupCount + TRAVERSAL_GROUPS_WORKGROUP - 1) / TRAVERSAL_GROUPS_WORKGROUP;
  #if USE_16BIT_DISPATCH
    uvec3 grid = fit16bitLaunchGrid(workGroupCount); 
    buildRW.indirectDispatchGroups.gridX = grid.x;
    buildRW.indirectDispatchGroups.gridY = grid.y;
    buildRW.indirectDispatchGroups.gridZ = grid.z;
  #else
    buildRW.indirectDispatchGroups.gridX = workGroupCount;
  #endif
  }
#endif
}



/*

// --------------------------------------------------------
// 平铺式遍历 (Flat Traversal)
// 绕过所有 Node 树，直接对最高精度 (LOD0) 的 Group/Cluster 进行暴力遍历与剔除
// --------------------------------------------------------
void run_flat()
{
  uint threadId = gl_GlobalInvocationID.x;
  uint totalThreads = gl_WorkGroupSize.x * gl_NumWorkGroups.x;

  // 1. 遍历场景中的所有实例
  // 修正 1: 使用 build.numRenderInstances
  uint numInstances = build.numRenderInstances; 

  for (uint instID = 0; instID < numInstances; instID++)
  {
    RenderInstance inst = instances[instID];
    mat4x3 worldMatrix = inst.worldMatrix;
    
    #if (USE_CULLING || USE_BLAS_MERGING) && TARGETS_RAY_TRACING
    uint visibilityState = build.instanceVisibility.d[instID];
    if ((visibilityState & INSTANCE_VISIBLE_BIT) == 0) continue; // 实例总体不可见，直接跳过
    #endif

    uint geometryID = inst.geometryID;
    Geometry geometry = geometries[geometryID];

    // 修正 2: 从 lodLevels 数组中获取 LOD 0 的 group 数量和偏移量
    // 只渲染 LOD 0 可以避免平铺测试时出现多级细节重叠 (Z-fighting)
    uint groupCount = geometry.lodLevels.d[0].groupCount; 
    uint groupOffset = geometry.lodLevels.d[0].groupOffset;

    // 2. 网格步幅循环：利用所有的线程平铺遍历 LOD 0 的 Group 列表
    for (uint i = threadId; i < groupCount; i += totalThreads)
    {
      uint gID = groupOffset + i;

      // 提取 Group 数据
      #if USE_STREAMING
      uint64_t groupAddress = geometry.streamingGroupAddresses.d[gID];
      // 如果数据未被流送进显存，跳过
      if (groupAddress >= STREAMING_INVALID_ADDRESS_START) continue; 
      Group_in groupRef = Group_in(groupAddress);
      #else
      Group_in groupRef = Group_in(geometry.preloadedGroups.d[gID]);
      #endif
      
      Group group = groupRef.d;
      
      // 修正 3: 直接使用 group.clusterCount
      uint clusterCount = group.clusterCount; 

      // 3. 遍历 Group 内的所有 Cluster (这是最终的渲染图元)
      for (uint c = 0; c < clusterCount; c++)
      {
        #if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)
        BBox bbox = Group_getClusterBBox(groupRef, c);
        
        // 执行平铺式视锥与遮挡剔除（传入 isNode = false，因为现在测试的是叶子节点）
        if (!queryWasVisible(worldMatrix, bbox, false)) {
          continue; // 如果在视锥外或被遮挡，直接剔除！
        }
        #endif

        // ========== 通过剔除，加入渲染队列 ==========
        
        uint offset = atomicAdd(buildRW.renderClusterCounter, 1);
        if (offset < build.maxRenderClusters)
        {
          TraversalInfo tInfo;
          tInfo.instanceID = instID;
          tInfo.packedNode = group.clusterResidentID + c; // 将其打包为 Cluster 指针

          uint64s_coh writePointer = uint64s_coh(uint64_t(build.renderClusterInfos));
          
          #if USE_ATOMIC_LOAD_STORE
          atomicStore(writePointer.d[offset], packTraversalInfo(tInfo), gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
          #else
          writePointer.d[offset] = packTraversalInfo(tInfo);
          #endif
      
        }
      }
    }
  }
}
void main()
{
  run_flat();
  // 分离 Group 处理模式下，通过第一号线程设置间接分发的 Grid 大小，用于后续 Compute Shader 的调用。
}
*/



/*
// --------------------------------------------------------
// 平铺式连续 LOD 遍历 (Flat Continuous LOD Traversal)
// 绕过 Node 树，线性遍历所有 Group，并在 Cluster 级别手动进行 DAG 误差切割
// --------------------------------------------------------
void run_flat_clod()
{
  uint threadId = gl_GlobalInvocationID.x;
  uint totalThreads = gl_WorkGroupSize.x * gl_NumWorkGroups.x;
  
  // 1. 遍历场景中的所有实例
  uint numInstances = build.numRenderInstances; 

  for (uint instID = 0; instID < numInstances; instID++)
  {
    RenderInstance inst = instances[instID];
    mat4x3 worldMatrix = inst.worldMatrix;
    
    #if (USE_CULLING || USE_BLAS_MERGING) && TARGETS_RAY_TRACING
    uint visibilityState = build.instanceVisibility.d[instID];
    if ((visibilityState & INSTANCE_VISIBLE_BIT) == 0) continue; 
    #endif

    // 计算用于 LOD 评估的缩放和矩阵
    float uniformScale = computeUniformScale(worldMatrix);
    float errorScale   = 1.0;
    #if USE_CULLING && !USE_FORCED_INVISIBLE_CULLING && TARGETS_RAY_TRACING
    if ((visibilityState & INSTANCE_VISIBLE_BIT) == 0) errorScale = build.culledErrorScale;
    #endif
    mat4x3 viewMat = mat4x3(build.traversalViewMatrix * toMat4(worldMatrix));

    uint geometryID = inst.geometryID;
    Geometry geometry = geometries[geometryID];

    // 获取当前几何体中包含的所有 Group 总数（跨越所有 LOD 级别）
    uint lodCount = geometry.lodLevelsCount;
    if (lodCount == 0) continue;
    uint totalGroups = geometry.lodLevels.d[lodCount - 1].groupOffset + geometry.lodLevels.d[lodCount - 1].groupCount;

    // 2. 网格步幅循环：线性遍历所有的 Group
    for (uint gID = threadId; gID < totalGroups; gID += totalThreads)
    {
      #if USE_STREAMING
      uint64_t groupAddress = geometry.streamingGroupAddresses.d[gID];
      if (groupAddress >= STREAMING_INVALID_ADDRESS_START) continue; 
      Group_in groupRef = Group_in(groupAddress);
      #else
      Group_in groupRef = Group_in(geometry.preloadedGroups.d[gID]);
      #endif
      
      Group group = groupRef.d;
      
      // =======================================================
      // LOD 条件 1 (上限): 当前 Group 误差必须大于阈值
      // 模拟原树状遍历中 "父节点继续向下遍历" 的条件
      // =======================================================
      bool groupTraverse = testForTraversal(viewMat, uniformScale, group.traversalMetric, errorScale);
      
      // 如果 groupTraverse 为 false，说明当前精度已经足够高，
      // 我们不需要渲染它的细节（它的分支原本会在 Node 层被剔除），跳过！
      if (!groupTraverse) continue;

      uint clusterCount = group.clusterCount; 
      
      // 3. 遍历 Group 内的 Cluster
      for (uint c = 0; c < clusterCount; c++)
      {
        // --- 视锥与遮挡剔除 ---
        #if USE_CULLING && (TARGETS_RASTERIZATION || USE_FORCED_INVISIBLE_CULLING)
        BBox bbox = Group_getClusterBBox(groupRef, c);
        if (!queryWasVisible(worldMatrix, bbox, false)) {
          continue; 
        }
        #endif

        // =======================================================
        // LOD 条件 2 (下限): 生成它的高精度父级 Group 的误差必须小于等于阈值
        // =======================================================
        bool forceCluster = false;
        TraversalMetric genMetric;
        uint32_t genGroupIdx = Group_getGeneratingGroup(groupRef, c);
        
        if (genGroupIdx != SHADERIO_ORIGINAL_MESH_GROUP) {
          #if USE_STREAMING
          uint64_t genAddr = geometry.streamingGroupAddresses.d[genGroupIdx];
          if (genAddr >= STREAMING_INVALID_ADDRESS_START) {
            // 父级还没流送进显存，强制渲染当前 Cluster 兜底
            genMetric = group.traversalMetric; 
            forceCluster = true;
          } else {
            genMetric = Group_in(genAddr).d.traversalMetric;
          }
          #else
          genMetric = Group_in(geometry.preloadedGroups.d[genGroupIdx]).d.traversalMetric;
          #endif
        } else {
          // 当前已经是最高精度 LOD 0，强制渲染
          genMetric = group.traversalMetric;
          forceCluster = true;
        }

        bool genTraverse = testForTraversal(viewMat, uniformScale, genMetric, errorScale);
        
        // 如果 genTraverse 为 true，说明需要更高的精度，当前 Cluster 精度不够，跳过！
        if (genTraverse && !forceCluster) continue;

        // ========== 通过了视锥、遮挡以及双端 LOD 测试，加入渲染队列 ==========
        
        uint offset = atomicAdd(buildRW.renderClusterCounter, 1);
        if (offset < build.maxRenderClusters)
        {
          TraversalInfo tInfo;
          tInfo.instanceID = instID;
          tInfo.packedNode = group.clusterResidentID + c; 

          uint64s_coh writePointer = uint64s_coh(uint64_t(build.renderClusterInfos));
          
          #if USE_ATOMIC_LOAD_STORE
          atomicStore(writePointer.d[offset], packTraversalInfo(tInfo), gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
          #else
          writePointer.d[offset] = packTraversalInfo(tInfo);
          #endif
          
          #if TARGETS_RAY_TRACING
          atomicAdd(build.instanceBuildInfos.d[instID].clusterReferencesCount, 1);
          #endif
        }
      }
    }
  }
}

// 主函数调用
void main()
{
  run_flat_clod(); 
}
*/