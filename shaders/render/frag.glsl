//==============================================================================
// 文件：shaders/render/frag.glsl
// 模块定位：渲染阶段着色器，把遍历输出的 簇 转换为 帧缓冲 中的颜色、深度和调试信息。
// 数据流：读取 render 簇 list 和 组/簇 payload，输出硬件网格着色器 primitive 或计算软件光栅结果。
// 方法说明：系统同时支持硬件网格着色器和软件光栅路径，用不同执行模型覆盖不同大小和密度的 簇。
// 正确性约束：硬件与软件路径必须共享同一深度编码和材质解释；簇 payload 地址必须来自有效驻留数据。
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


#if ALLOW_SHADING && (DEBUG_VISUALIZATION || ALLOW_VERTEX_NORMALS || ALLOW_VERTEX_TEXCOORDS)
#extension GL_EXT_fragment_shader_barycentric : enable
#endif

#if USE_SW_RASTER
#extension GL_EXT_shader_image_int64 : enable
#endif


// 依赖说明：引入共享布局、剔除、着色或阶段间复用的着色器片段。
// 这些 include 共同决定本文件能访问的结构布局、数学辅助函数和编译期宏。
#include "shaderio.h"


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(push_constant) uniform pushData {
  uint instanceID;
} push;


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer { FrameConstants view; };


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer { Readback readback; };


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer { RenderInstance instances[]; };


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer { Geometry geometries[]; };


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_SCENEBUILDING_UBO, set = 0) uniform buildBuffer { SceneBuilding build; };

#if USE_STREAMING


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer { SceneStreaming streaming; };


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) buffer streamingBufferRW { SceneStreaming streamingRW; };
#endif

#if USE_SW_RASTER


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(set = 0, binding = BINDINGS_RASTER_ATOMIC, r64ui) uniform u64image2D imgRasterAtomic;
#endif
#include "attribute_encoding.h"
#include "render_shading.glsl"


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(location = 0) in Interpolants {
  flat uint clusterID;
  flat uint instanceID;
#if ALLOW_SHADING
  vec3 wPos;
#endif
} IN;

#if ALLOW_SHADING && (ALLOW_VERTEX_NORMALS || ALLOW_VERTEX_TEXCOORDS)


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(location = 3) pervertexEXT in Interpolants2 {
  uint vertexID;
} INBARY[];
#endif

#if !USE_SW_RASTER


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(location = 0, index = 0) out vec4 out_Color;
#else
vec4 out_Color;
#endif


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(early_fragment_tests) in;


// 函数：main。作为本着色器阶段入口，按绑定资源执行当前 GPU 工作。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该入口位于控制流根部，调用顺序决定后续资源生命周期和数据依赖。
void main()
{

  vec4 wTangent  = vec4(1);

  vec3 wNormal   = vec3(1);

  vec2 oTexCoord = vec2(1);

  RenderInstance instance = instances[IN.instanceID];

#if USE_STREAMING

  Cluster_in clusterRef = Cluster_in(streaming.resident.clusters.d[IN.clusterID]);
#else
  Geometry   geometry   = geometries[instance.geometryID];

  Cluster_in clusterRef = Cluster_in(geometry.preloadedClusters.d[IN.clusterID]);
#endif

#if ALLOW_SHADING

#if ALLOW_VERTEX_NORMALS || ALLOW_VERTEX_TEXCOORDS
  Cluster    cluster    = clusterRef.d;


  uint32s_in oNormals   = Cluster_getVertexNormals(clusterRef);

  vec2s_in   oTexCoords = Cluster_getVertexTexCoords(clusterRef);


  uvec3 triangleIndices = uvec3(INBARY[0].vertexID, INBARY[1].vertexID, INBARY[2].vertexID);
#endif

#if ALLOW_VERTEX_NORMALS

  if(view.facetShading != 0 || (cluster.attributeBits & CLUSTER_ATTRIBUTE_VERTEX_NORMAL) == 0)
#endif
  {

    wNormal = -cross(dFdx(IN.wPos), dFdy(IN.wPos));

    wNormal = normalize(wNormal);
  }
#if ALLOW_VERTEX_NORMALS
  else
  {

    vec3 baryWeight   = gl_BaryCoordEXT;

    mat3 worldMatrixI = mat3(instance.worldMatrixI);


    uvec3 triNormalsPacked = uvec3(oNormals.d[triangleIndices.x], oNormals.d[triangleIndices.y], oNormals.d[triangleIndices.z]);
    vec3 triNormals[3];


    triNormals[0] = normal_unpack(triNormalsPacked.x);

    triNormals[1] = normal_unpack(triNormalsPacked.y);

    triNormals[2] = normal_unpack(triNormalsPacked.z);

    vec3 oNormal = baryWeight.x * triNormals[0] + baryWeight.y * triNormals[1] + baryWeight.z * triNormals[2];

    wNormal = normalize(vec3(oNormal * worldMatrixI));

#if USE_FORCED_TWO_SIDED
    if (!gl_FrontFacing){
      wNormal = -wNormal;
    }
#elif USE_TWO_SIDED
  if (instance.twoSided != 0) {

    uint8s_in localIndices = uint8s_in(Cluster_getTriangleIndices(clusterRef));
    uint triangleIndicesRef = localIndices.d[gl_PrimitiveID * 3 + 0];
    if (triangleIndicesRef != triangleIndices.x) {
      wNormal = -wNormal;
    }
  }
#endif

#if ALLOW_VERTEX_TANGENTS
    if((cluster.attributeBits & CLUSTER_ATTRIBUTE_VERTEX_TANGENT) != 0)
    {


      vec4 tangent0 = tangent_unpack(triNormals[0], triNormalsPacked.x >> ATTRENC_NORMAL_BITS);
      wTangent.w    = tangent0.w;

      vec3 oTangent = baryWeight.x * tangent0.xyz
                      + baryWeight.y * tangent_unpack(triNormals[1], triNormalsPacked.y >> ATTRENC_NORMAL_BITS).xyz
                      + baryWeight.z * tangent_unpack(triNormals[2], triNormalsPacked.z >> ATTRENC_NORMAL_BITS).xyz;
      wTangent.xyz = oTangent * worldMatrixI;
    }
#endif
  }
#endif

#if ALLOW_VERTEX_TEXCOORD_0
  if((cluster.attributeBits & CLUSTER_ATTRIBUTE_VERTEX_TEX_0) != 0)
  {

    oTexCoord = gl_BaryCoordEXT.x * oTexCoords.d[triangleIndices.x]
              + gl_BaryCoordEXT.y * oTexCoords.d[triangleIndices.y]
              + gl_BaryCoordEXT.z * oTexCoords.d[triangleIndices.z];
  }
#endif
#endif

  uint visData = IN.clusterID;
#if ALLOW_SHADING
  if(view.visualize == VISUALIZE_LOD || view.visualize == VISUALIZE_GROUP)
  {
    if(view.visualize == VISUALIZE_LOD)
    {

      visData = floatBitsToUint(float(clusterRef.d.lodLevel) * instances[IN.instanceID].maxLodLevelRcp);
    }
    else
    {

      uvec2 baseAddress = unpackUint2x32(uint64_t(clusterRef) - clusterRef.d.groupChildIndex * Cluster_size);
      visData           = baseAddress.x ^ baseAddress.y;
    }
  }
  else if(view.visualize == VISUALIZE_TRIANGLE)
  {


    visData = IN.clusterID * 256 + uint(gl_PrimitiveID);
  }
  out_Color.w = 1.f;
  {

    const float overHeadLight = 1.0f;
    const float ambientLight  = 0.7f;


    out_Color = shading(IN.instanceID, IN.wPos, wNormal, wTangent, oTexCoord, visData, overHeadLight, ambientLight);
  #if DEBUG_VISUALIZATION

    if(view.doWireframe != 0)
    {
      out_Color.xyz = addWireframe(out_Color.xyz, gl_BaryCoordEXT, true, fwidthFine(gl_BaryCoordEXT), view.wireColor);
    }
  #endif
  }
#else
  {


    uint triangleCountMinusOne = CLUSTER_TRIANGLE_COUNT-1;
    float relative = (float(gl_PrimitiveID) / float(triangleCountMinusOne)) * 0.25 + 0.75;
    out_Color = vec4(colorizeID(visData) * relative, 1.0);
  }
#endif


  uvec2 pixelCoord = uvec2(gl_FragCoord.xy);
  if(pixelCoord == view.mousePosition)
  {

    uint32_t packedClusterTriangleId = (IN.clusterID << 8) | (gl_PrimitiveID & 0xFF);


    atomicMax(readback.clusterTriangleId, packPickingValue(packedClusterTriangleId, gl_FragCoord.z));
    atomicMax(readback.instanceId, packPickingValue(IN.instanceID, gl_FragCoord.z));
  }

#if USE_SW_RASTER
  {


    uint64_t u64 = packUint2x32(uvec2(packUnorm4x8(out_Color), floatBitsToUint(gl_FragCoord.z)));


    imageAtomicMax(imgRasterAtomic, ivec2(gl_FragCoord.xy), u64);
  }
#endif
}
