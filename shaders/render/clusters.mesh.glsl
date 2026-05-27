//从显存中读取簇（Cluster）的顶点和索引数据、进行矩阵变换、执行可选的微小图元剔除，并最终将图元输出给硬件光栅化器。
#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_ballot : require
// 根据宏定义选择使用标准的 EXT 网格着色器扩展，还是 NVIDIA 专有的 NV 扩展
#if USE_EXT_MESH_SHADER
#extension GL_EXT_mesh_shader : require
#else
#extension GL_NV_mesh_shader : require
#endif
#extension GL_EXT_control_flow_attributes : require
#include "shaderio.h"
// 禁用无效的配置：如果开启了图元级剔除（USE_PRIMITIVE_CULLING），但使用的是 EXT Mesh Shader 或者全局关闭了剔除（USE_CULLING），则强制关闭图元剔除
#if USE_PRIMITIVE_CULLING && (USE_EXT_MESH_SHADER || !USE_CULLING)
#undef USE_PRIMITIVE_CULLING
#define USE_PRIMITIVE_CULLING 0
#endif
// 定义 Push Constant，用于传递小块的、快速更新的常量数据
layout(push_constant) uniform pushData
{
  uint instanceID;// 当前的实例 ID
}
push;
// 绑定 0 集合的各个 Buffer
layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;// 摄像机视图矩阵、投影矩阵等帧常量
};
layout(scalar,binding=BINDINGS_READBACK_SSBO,set=0) buffer readbackBuffer
{
  Readback readback;// 用于向 CPU 回读数据的 Buffer（如统计渲染了多少个三角形）
};
layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer
{
  RenderInstance instances[];// 场景中所有渲染实例的数组（包含模型矩阵等）
};
layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];// 场景中所有几何体数据的数组
};
layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer
{
  SceneBuilding build;  // 场景构建数据（只读 UBO），包含要渲染的 Cluster 列表
};
layout(scalar, binding = BINDINGS_SCENEBUILDING_SSBO, set = 0) buffer buildBufferRW
{
  SceneBuilding buildRW;  // 场景构建数据（可读写 SSBO，本着色器通常只读）
};
// 如果启用了流式加载系统，则绑定流式加载相关的 Buffer
#if USE_STREAMING
layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer
{
  SceneStreaming streaming;
};
layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) buffer streamingBufferRW
{
  SceneStreaming streamingRW;
};
#endif
// 当没有启用纯深度渲染模式（USE_DEPTH_ONLY）时，需要向 Fragment Shader 输出额外的着色数据
#if !USE_DEPTH_ONLY
layout(location = 0) out Interpolants
{
  flat uint clusterID;
  flat uint instanceID;
#if ALLOW_SHADING
  vec3      wPos;
#endif
}
OUT[];
#if ALLOW_SHADING && (ALLOW_VERTEX_NORMALS || ALLOW_VERTEX_TEXCOORDS)
// 可选：输出顶点 ID 给片段着色器，可能用于在片段着色器中手动提取法线/UV等（节省插值开销）
layout(location = 3) out Interpolants2
{
  flat uint vertexID;
}
OUTBARY[];
#endif
#endif
// 定义 Mesh Shader 的工作组大小（默认 32 个线程，即一个 Warp）
#ifndef MESHSHADER_WORKGROUP_SIZE
#define MESHSHADER_WORKGROUP_SIZE 32
#endif
// 指定输入的工作组大小
layout(local_size_x = MESHSHADER_WORKGROUP_SIZE) in;
// 指定输出的最大顶点数和最大图元（三角形）数
layout(max_vertices = CLUSTER_VERTEX_COUNT, max_primitives = CLUSTER_TRIANGLE_COUNT) out;
layout(triangles) out;// 指定输出图元类型为三角形
// 计算处理所有顶点和三角形需要循环迭代的次数（总数除以工作组线程数，向上取整）
const uint MESHLET_VERTEX_ITERATIONS = ((CLUSTER_VERTEX_COUNT + MESHSHADER_WORKGROUP_SIZE - 1) / MESHSHADER_WORKGROUP_SIZE);
const uint MESHLET_TRIANGLE_ITERATIONS = ((CLUSTER_TRIANGLE_COUNT + MESHSHADER_WORKGROUP_SIZE - 1) / MESHSHADER_WORKGROUP_SIZE);
// 如果启用了图元剔除或双面材质，则包含 culling.glsl 中的剔除辅助函数
#if USE_PRIMITIVE_CULLING || USE_TWO_SIDED
#define CULLING_NO_HIZ// 此阶段不需要 HiZ 深度图剔除，只需视锥或背面剔除
#include "culling.glsl"
#endif
// 在 EXT mesh shader 中，如果启用了双面材质，需要一个共享内存数组来暂存顶点位置，以便后续计算三角形朝向
#if USE_EXT_MESH_SHADER && USE_TWO_SIDED
shared vec4 s_vertices[CLUSTER_VERTEX_COUNT];
#endif

void main()
{
#if USE_EXT_MESH_SHADER
  // EXT 网格着色器可能分配超过实际需求的计算网格
uint workGroupID  = getWorkGroupIndexLinearized(gl_WorkGroupID); // 获取当前的工作组 ID
  bool isValid      = workGroupID < build.numRenderedClusters; // 检查 ID 是否超出实际要渲染的 Cluster 数量
  // 安全地获取当前工作组负责处理的 Cluster 信息
  ClusterInfo cinfo = build.renderClusterInfos.d[min(workGroupID, build.numRenderedClusters-1)];
#else
  // NV 扩展下的工作组 ID
  uint workGroupID  = gl_WorkGroupID.x;
  ClusterInfo cinfo = build.renderClusterInfos.d[workGroupID];
#endif
  // 提取实例 ID 和 簇 ID
  uint instanceID = cinfo.instanceID;
  uint clusterID  = cinfo.clusterID;
  // 根据 ID 获取实例对象和其对应的几何体信息
  RenderInstance instance = instances[instanceID];
  Geometry geometry       = geometries[instance.geometryID];
  // 根据是否启用流式加载，从不同的显存位置获取 Cluster 的元数据指针（Buffer Reference）
#if USE_STREAMING
  Cluster_in clusterRef = Cluster_in(streaming.resident.clusters.d[clusterID]);
#else
  Cluster_in clusterRef = Cluster_in(geometry.preloadedClusters.d[clusterID]);
#endif
  Cluster cluster = clusterRef.d; // 读取 Cluster 描述数据
  uint vertMax = cluster.vertexCountMinusOne; // 顶点数减1（最大顶点索引）
  uint triMax  = cluster.triangleCountMinusOne; // 三角形数减1（最大三角形索引）

#if USE_EXT_MESH_SHADER
  // 设置 EXT 规范要求的有效输出数量
  uint vertCount = isValid ? vertMax + 1 : 0;
  uint triCount  = isValid ? triMax + 1 : 0;
  SetMeshOutputsEXT(vertCount, triCount);// 通知管线即将输出的顶点和图元数量
  if (triCount == 0)
    return;// 如果无效或无图元，直接退出
#elif !USE_PRIMITIVE_CULLING
  // 在 NV 规范中，如果没有开启细粒度图元剔除，则由 0 号线程统一声明输出的三角形数量
  if (gl_LocalInvocationID.x == 0) {
    gl_PrimitiveCountNV = triMax + 1;
  }
#endif
// 统计渲染数据（原子操作更新 CPU 可读的回读 Buffer）
#if USE_RENDER_STATS
  if (gl_LocalInvocationID.x == 0) {
    atomicAdd(readback.numRenderedTriangles, uint(triMax + 1));
  #if !USE_PRIMITIVE_CULLING
    atomicAdd(readback.numRasteredTriangles, uint(triMax + 1));
  #endif
  }
#endif
// 获取顶点位置和三角形索引数组的物理地址指针
  vec3s_in  oVertices    = vec3s_in(Cluster_getVertexPositions(clusterRef));
  uint8s_in localIndices = uint8s_in(Cluster_getTriangleIndices(clusterRef));
  
  // 获取当前集群的LOD级别
  uint currentLodLevel = cluster.lodLevel;
  
  // 计算LOD过渡因子
  float lodTransitionFactor = 0.0;
  
  // 这里可以根据需要实现更复杂的LOD过渡逻辑
  // 例如，基于距离、时间和误差计算过渡因子
  
  // ==================== 顶点处理阶段 ====================
  [[unroll]] for(uint i = 0; i < uint(MESHLET_VERTEX_ITERATIONS); i++)
  {
  // 每个线程负责一个顶点。如果顶点数超过线程数，则循环迭代
    uint vert        = gl_LocalInvocationID.x + i * MESHSHADER_WORKGROUP_SIZE;
    uint vertLoad    = min(vert, vertMax);// 防止越界读取
    // 读取模型空间顶点位置，并乘以世界矩阵转换为世界空间
    vec3 oPos = oVertices.d[vertLoad]; 
    vec3 wPos = instance.worldMatrix * vec4(oPos, 1.0f);
    
    // 应用LOD平滑过渡效果
    // 这里可以根据lodTransitionFactor混合不同LOD级别的顶点位置
    // 例如，在相邻LOD级别之间进行位置插值
    
    // 如果当前线程对应的顶点有效
    if(vert <= vertMax)
    {
    // 乘以视口投影矩阵，转换到齐次裁剪空间
      vec4 hPos = view.viewProjMatrix * vec4(wPos,1);
      // 将位置写入到内置变量中，供光栅化器使用
    #if USE_EXT_MESH_SHADER
      gl_MeshVerticesEXT[vert].gl_Position = hPos;
    #else
      gl_MeshVerticesNV[vert].gl_Position = hPos;
    #endif
    // 如果启用了 EXT 双面材质支持，将其存入 shared memory 中供后续面的朝向测试使用
    #if USE_EXT_MESH_SHADER && USE_TWO_SIDED
      s_vertices[vert] = hPos;
    #endif
    // 写入片段着色器所需的其它插值数据
    #if !USE_DEPTH_ONLY
    #if ALLOW_SHADING
      OUT[vert].wPos                      = wPos.xyz;
    #endif
    #if ALLOW_SHADING && (ALLOW_VERTEX_NORMALS || ALLOW_VERTEX_TEXCOORDS)
      OUTBARY[vert].vertexID              = vert;
    #endif
      OUT[vert].clusterID                 = clusterID;
      OUT[vert].instanceID                = instanceID;
    #endif
    }
  }
  // ==================== 图元（三角形）处理阶段 ====================
  uint triOutCount = 0;// 记录最终成功输出的三角形数量
  [[unroll]] for(uint i = 0; i < uint(MESHLET_TRIANGLE_ITERATIONS); i++)
  {
    // 每个线程负责处理一个三角形
    uint tri     = gl_LocalInvocationID.x + i * MESHSHADER_WORKGROUP_SIZE;
    uint triLoad = min(tri, triMax);// 防止越界读取
    // 从 8 位索引 Buffer 中读取三个顶点的局部索引
    uvec3 indices = uvec3(localIndices.d[triLoad * 3 + 0],
                          localIndices.d[triLoad * 3 + 1],
                          localIndices.d[triLoad * 3 + 2]);
#if !USE_FORCED_TWO_SIDED
    // 处理实例镜像（负缩放）造成的绕序反转，或者处理双面材质的背面情况
    if (instance.flipWinding != 0
#if USE_TWO_SIDED && !USE_EXT_MESH_SHADER
// 在 NV 模式下利用内置的 gl_MeshVerticesNV 判断是否为正面
      || (instance.twoSided != 0 && !isFrontFacingHW(gl_MeshVerticesNV[indices.x].gl_Position,
                                                     gl_MeshVerticesNV[indices.y].gl_Position,
                                                     gl_MeshVerticesNV[indices.z].gl_Position))
#elif USE_TWO_SIDED && USE_EXT_MESH_SHADER
// 在 EXT 模式下利用刚刚存入 s_vertices 的坐标判断
      || (instance.twoSided != 0 && !isFrontFacingHW(s_vertices[indices.x],s_vertices[indices.y],s_vertices[indices.z]))
#endif
    )
    {
    // 翻转绕序 (比如把 v0,v1,v2 变成 v0,v2,v1)
      indices.xy = indices.yx;
    }
#endif
#if USE_PRIMITIVE_CULLING
// 如果启用了细粒度剔除，测试三角形是否有效（未越界且在视锥内/未退化等）
    bool isRendered = tri <= triMax
       && testTriangleHW( gl_MeshVerticesNV[indices.x].gl_Position,gl_MeshVerticesNV[indices.y].gl_Position,gl_MeshVerticesNV[indices.z].gl_Position);
    // 使用 Subgroup 级别的硬件指令将布尔结果打包
    uvec4 voteRendered = subgroupBallot(isRendered);
    // 计算当前存活的三角形应该被写入的密集输出数组的索引
    // subgroupBallotExclusiveBitCount 返回当前线程之前有多少个 true
    uint triOut = subgroupBallotExclusiveBitCount(voteRendered) + triOutCount;
    // 累加整个 Warp 内存活的三角形总数
    triOutCount += subgroupBallotBitCount(voteRendered);
#else
    // 如果未启用图元剔除，则按顺序全部输出
    bool isRendered = tri <= triMax;
    uint triOut     = tri;
#endif
    // 如果该三角形未被剔除，则将其索引输出给管线
    if(isRendered)
    {
    /*
    #if USE_EXT_MESH_SHADER
    gl_PrimitiveTriangleIndicesEXT[triOut] = indices;
    gl_MeshPrimitivesEXT[triOut].gl_PrimitiveID = int(tri);
    #else
    gl_PrimitiveIndicesNV[triOut * 3 + 0] = indices.x;
    gl_PrimitiveIndicesNV[triOut * 3 + 1] = indices.y;
    gl_PrimitiveIndicesNV[triOut * 3 + 2] = indices.z;
    gl_MeshPrimitivesNV[triOut].gl_PrimitiveID = int(tri);
    #endif
    */
    //gl_PrimitiveID 用于 debug visualization 等，深度模式不需要
    #if USE_EXT_MESH_SHADER
      gl_PrimitiveTriangleIndicesEXT[triOut] = indices;
      #if !USE_DEPTH_ONLY
      gl_MeshPrimitivesEXT[triOut].gl_PrimitiveID = int(tri);// EXT 规范写入索引的方式
      #endif
    #else
    // NV 规范写入索引的方式（需要分别写入3个角）
      gl_PrimitiveIndicesNV[triOut * 3 + 0] = indices.x;
      gl_PrimitiveIndicesNV[triOut * 3 + 1] = indices.y;
      gl_PrimitiveIndicesNV[triOut * 3 + 2] = indices.z;
      #if !USE_DEPTH_ONLY
      gl_MeshPrimitivesNV[triOut].gl_PrimitiveID = int(tri);
      #endif
    #endif
    }
  }
  // ==================== 善后工作 ====================
#if USE_PRIMITIVE_CULLING
// 在启用了剔除压缩后，需要在 NV 模式下由 0 号线程向管线宣告最终实际存活了多少三角形
  if (gl_LocalInvocationID.x == 0) {
    gl_PrimitiveCountNV = triOutCount;
  }
  // 统计最终光栅化了多少三角形
#if USE_RENDER_STATS
  if (gl_LocalInvocationID.x == 0) {
    atomicAdd(readback.numRasteredTriangles, triOutCount);
  }
#endif
#endif
}