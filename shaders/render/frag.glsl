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
// 如果允许着色且开启了法线/UV等属性，则启用重心坐标扩展。
// 这是一个高级特性：允许在像素着色器中直接获取三角形三个顶点的属性，由开发者自己计算插值，而不是依赖硬件的自动插值！
#if ALLOW_SHADING && (DEBUG_VISUALIZATION || ALLOW_VERTEX_NORMALS || ALLOW_VERTEX_TEXCOORDS)
#extension GL_EXT_fragment_shader_barycentric : enable
#endif
// 如果开启了软件光栅化，开启 64位 图像原子操作扩展
#if USE_SW_RASTER
#extension GL_EXT_shader_image_int64 : enable
#endif
#include "shaderio.h"

// ==================== 数据绑定 ====================
// 推送常量 (Push Constants)，存放在 GPU 寄存器中的极少量数据，访问极快
layout(push_constant) uniform pushData {
  uint instanceID; // 当前正在绘制的实例 ID
} push;
// 绑定帧常量 (相机矩阵、屏幕分辨率、鼠标位置等)
layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer { FrameConstants view; };
// 绑定回读缓冲 (用于将鼠标点击选中的物体 ID 传回给 CPU)
layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer { Readback readback; };
// 绑定场景实例数组 (包含所有物体的世界矩阵等)
layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer { RenderInstance instances[]; };
// 绑定几何体数组
layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer { Geometry geometries[]; };
// 绑定场景构建缓冲
layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer { SceneBuilding build; };
// 绑定流式加载缓冲
#if USE_STREAMING
layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer { SceneStreaming streaming; };
layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) buffer streamingBufferRW { SceneStreaming streamingRW; };
#endif
// 软硬光栅化统一的输出目标：一个支持 64位无符号整数 的 Atomic Image
#if USE_SW_RASTER
layout(set = 0, binding = BINDINGS_RASTER_ATOMIC, r64ui) uniform u64image2D imgRasterAtomic;
#endif
#include "attribute_encoding.h" // 引入法线/切线解压函数
#include "render_shading.glsl"  // 引入具体的光照计算函数
// ==================== 着色器输入输出 ====================
// 从顶点着色器/网格着色器传来的插值数据
layout(location = 0) in Interpolants {
  flat uint clusterID;  // 当前像素属于哪个微网格簇 (flat表示不插值)
  flat uint instanceID; // 属于哪个实例
#if ALLOW_SHADING
  vec3 wPos;            // 当前像素的世界坐标
#endif
} IN;
// pervertexEXT 扩展：直接获取构成当前三角形的三个顶点的原始 ID
#if ALLOW_SHADING && (ALLOW_VERTEX_NORMALS || ALLOW_VERTEX_TEXCOORDS)
layout(location = 3) pervertexEXT in Interpolants2 {
  uint vertexID; 
} INBARY[]; // 数组大小为3，代表三角形的三个顶点
#endif
// 输出颜色。如果是传统的硬件光栅化，输出到 location 0；否则自定义 vec4 变量
#if !USE_SW_RASTER
layout(location = 0, index = 0) out vec4 out_Color;
#else
vec4 out_Color;
#endif
// 强制开启提前深度测试 (Early-Z)，提高性能，被遮挡的像素直接丢弃，不执行后面的计算
layout(early_fragment_tests) in;
// ==================== 主函数 ====================
void main()
{
  vec4 wTangent  = vec4(1);
  vec3 wNormal   = vec3(1);
  vec2 oTexCoord = vec2(1);
  // 获取当前实例的数据
  RenderInstance instance = instances[IN.instanceID];
  // 通过物理指针，获取当前微网格簇(Cluster)的头部数据
#if USE_STREAMING
  Cluster_in clusterRef = Cluster_in(streaming.resident.clusters.d[IN.clusterID]);
#else
  Geometry   geometry   = geometries[instance.geometryID];
  Cluster_in clusterRef = Cluster_in(geometry.preloadedClusters.d[IN.clusterID]);
#endif

#if ALLOW_SHADING
// 准备解压顶点属性 (法线、UV)
#if ALLOW_VERTEX_NORMALS || ALLOW_VERTEX_TEXCOORDS
  Cluster    cluster    = clusterRef.d;
  // 拿到法线和 UV 的内存指针
  uint32s_in oNormals   = Cluster_getVertexNormals(clusterRef);
  vec2s_in   oTexCoords = Cluster_getVertexTexCoords(clusterRef);
  // 三角形的三个顶点在顶点缓冲中的局部索引
  uvec3 triangleIndices = uvec3(INBARY[0].vertexID, INBARY[1].vertexID, INBARY[2].vertexID);
#endif
// 【法线计算】
#if ALLOW_VERTEX_NORMALS
// 如果要求“刻面着色(Facet Shading)”或者模型压根没有法线
  if(view.facetShading != 0 || (cluster.attributeBits & CLUSTER_ATTRIBUTE_VERTEX_NORMAL) == 0)
#endif
  {
  // 利用屏幕空间导数 dFdx / dFdy 计算当前像素表面的切向量，然后叉乘得到纯平面的法线！(Low-poly 风格)
    wNormal = -cross(dFdx(IN.wPos), dFdy(IN.wPos));
    wNormal = normalize(wNormal);
  }
#if ALLOW_VERTEX_NORMALS
  else
  {
  // 否则，进行高精度插值：
    vec3 baryWeight   = gl_BaryCoordEXT; // 获取当前像素的重心坐标 (例如 0.5, 0.3, 0.2)
    mat3 worldMatrixI = mat3(instance.worldMatrixI); // 获取逆矩阵用于法线变换

    // 从显存中读出构成该三角形的 3个顶点的【压缩法线】(每个32位)
    uvec3 triNormalsPacked = uvec3(oNormals.d[triangleIndices.x], oNormals.d[triangleIndices.y], oNormals.d[triangleIndices.z]);
    vec3 triNormals[3];
    // 调用 attribute_encoding.h 里的函数，将压缩的整数解压回 3D 浮点向量
    triNormals[0] = normal_unpack(triNormalsPacked.x);
    triNormals[1] = normal_unpack(triNormalsPacked.y);
    triNormals[2] = normal_unpack(triNormalsPacked.z);
    // 手动用重心坐标对 3 个顶点的法线进行插值，得到平滑的像素法线
    vec3 oNormal = baryWeight.x * triNormals[0] + baryWeight.y * triNormals[1] + baryWeight.z * triNormals[2];
    // 将法线从模型空间转换到世界空间
    wNormal = normalize(vec3(oNormal * worldMatrixI));
    // 双面材质处理：如果是背面，就把法线反转
#if USE_FORCED_TWO_SIDED
    if (!gl_FrontFacing){
      wNormal = -wNormal;
    }
#elif USE_TWO_SIDED
  if (instance.twoSided != 0) {
    // 复杂的判断逻辑：检查顶点绕序是否改变，改变了则反转法线
    uint8s_in localIndices = uint8s_in(Cluster_getTriangleIndices(clusterRef));
    uint triangleIndicesRef = localIndices.d[gl_PrimitiveID * 3 + 0];
    if (triangleIndicesRef != triangleIndices.x) {
      wNormal = -wNormal;
    }
  }
#endif
// 【切线解压与插值】
#if ALLOW_VERTEX_TANGENTS
    if((cluster.attributeBits & CLUSTER_ATTRIBUTE_VERTEX_TANGENT) != 0)
    {
      // 切线也是压缩在法线同一个 32 位整数的高位中，右移后解包
      vec4 tangent0 = tangent_unpack(triNormals[0], triNormalsPacked.x >> ATTRENC_NORMAL_BITS);
      wTangent.w    = tangent0.w;
      // 重心坐标插值
      vec3 oTangent = baryWeight.x * tangent0.xyz
                      + baryWeight.y * tangent_unpack(triNormals[1], triNormalsPacked.y >> ATTRENC_NORMAL_BITS).xyz
                      + baryWeight.z * tangent_unpack(triNormals[2], triNormalsPacked.z >> ATTRENC_NORMAL_BITS).xyz;
      wTangent.xyz = oTangent * worldMatrixI;
    }
#endif
  }
#endif
// 【UV解压与插值】
#if ALLOW_VERTEX_TEXCOORD_0
  if((cluster.attributeBits & CLUSTER_ATTRIBUTE_VERTEX_TEX_0) != 0)
  {
  // 同样利用重心坐标手动对 3 个顶点的 UV 进行平滑插值
    oTexCoord = gl_BaryCoordEXT.x * oTexCoords.d[triangleIndices.x]
              + gl_BaryCoordEXT.y * oTexCoords.d[triangleIndices.y]
              + gl_BaryCoordEXT.z * oTexCoords.d[triangleIndices.z];
  }
#endif
#endif
// ==================== 调试与可视化 (Debug Visualization) ====================
  uint visData = IN.clusterID;
#if ALLOW_SHADING
  if(view.visualize == VISUALIZE_LOD || view.visualize == VISUALIZE_GROUP)
  {
    if(view.visualize == VISUALIZE_LOD)
    {
    // 可视化 LOD 层级，将层级值映射为一个伪颜色
      visData = floatBitsToUint(float(clusterRef.d.lodLevel) * instances[IN.instanceID].maxLodLevelRcp);
    }
    else
    {
    // 可视化 Group (渲染组)，用内存地址做个异或计算出随机色
      uvec2 baseAddress = unpackUint2x32(uint64_t(clusterRef) - clusterRef.d.groupChildIndex * Cluster_size);
      visData           = baseAddress.x ^ baseAddress.y;
    }
  }
  else if(view.visualize == VISUALIZE_TRIANGLE)
  {
  // 可视化单个三角形，给每个三角形上不同的颜色
    visData = IN.clusterID * 256 + uint(gl_PrimitiveID);
  }
  out_Color.w = 1.f;// Alpha 通道
  {
  // 简单的光照参数
    const float overHeadLight = 1.0f;
    const float ambientLight  = 0.7f;
    // 调用外部的 shading 函数计算 PBR / 基础光照
    out_Color = shading(IN.instanceID, IN.wPos, wNormal, wTangent, oTexCoord, visData, overHeadLight, ambientLight);
  #if DEBUG_VISUALIZATION
  // 如果需要画线框，通过重心坐标判断当前像素是否在三角形边缘
    if(view.doWireframe != 0)
    {
      out_Color.xyz = addWireframe(out_Color.xyz, gl_BaryCoordEXT, true, fwidthFine(gl_BaryCoordEXT), view.wireColor);
    }
  #endif
  }
#else
  {
    //uint triangleCountMinusOne = clusterRef.d.triangleCountMinusOne;
    // 不渲染光照，纯色显示 (通常用于极简模式或性能测试)
    uint triangleCountMinusOne = CLUSTER_TRIANGLE_COUNT-1;
    float relative = (float(gl_PrimitiveID) / float(triangleCountMinusOne)) * 0.25 + 0.75;
    out_Color = vec4(colorizeID(visData) * relative, 1.0);
  }
#endif
// ==================== 鼠标精确拾取 (Mouse Picking) ====================
  uvec2 pixelCoord = uvec2(gl_FragCoord.xy);// 当前像素在屏幕上的 XY 坐标
  if(pixelCoord == view.mousePosition)// 如果当前像素刚好在鼠标指着的地方
  {
    // 把簇ID和三角形ID打包
    uint32_t packedClusterTriangleId = (IN.clusterID << 8) | (gl_PrimitiveID & 0xFF);
    // 使用 atomicMax 原子操作写回显存。
    // 这里非常巧妙地把 "深度值(Z)" 放在了打包数据的最高位，
    // 这样 atomicMax 会自动选出 Z 最大的那个（即离屏幕最近的前景物体），实现无锁比较！
    atomicMax(readback.clusterTriangleId, packPickingValue(packedClusterTriangleId, gl_FragCoord.z));
    atomicMax(readback.instanceId, packPickingValue(IN.instanceID, gl_FragCoord.z));
  }
  // ==================== 软硬件光栅化统一输出 ====================
#if USE_SW_RASTER
  {
    // 如果整个引擎配置为统一混合光栅化（软+硬）：
    // 硬件光栅化不直接输出到颜色附件，而是把 计算出的 8位 RGBA 颜色 和 32位深度 打包成一个 64 位整数。
    uint64_t u64 = packUint2x32(uvec2(packUnorm4x8(out_Color), floatBitsToUint(gl_FragCoord.z)));
    // 强制使用 64位原子最大值操作写入统一的 Image 中。
    // 这也是 Nanite 的核心绝招：因为硬件光栅化和计算着色器(Compute Shader)软件光栅化是并行跑的，
    // 它们输出先后不确定，只有用 64位(颜色+深度)的 atomicMax 才能保证两边混合在一起时深度关系绝对正确！
    imageAtomicMax(imgRasterAtomic, ivec2(gl_FragCoord.xy), u64);
  }
#endif
}