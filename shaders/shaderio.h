#ifndef _SHADERIO_H_
#define _SHADERIO_H_
#include "shaderio_core.h"
#include "shaderio_scene.h"
#include "shaderio_streaming.h"
#include "shaderio_building.h"
#include "nvshaders/sky_io.h.slang"

/////////////////////////////////////////

#define VISUALIZE_MATERIAL 0
#define VISUALIZE_GREY 1
#define VISUALIZE_VIS_BUFFER 2
#define VISUALIZE_CLUSTER 3
#define VISUALIZE_GROUP 4
#define VISUALIZE_LOD 5
#define VISUALIZE_TRIANGLE 6
#define VISUALIZE_DEPTH_ONLY 7

#define MESHSHADER_BBOX_VERTICES 8
#define MESHSHADER_BBOX_LINES 12
#define MESHSHADER_BBOX_THREADS 4

/////////////////////////////////////////

#define BINDINGS_FRAME_UBO 0
#define BINDINGS_READBACK_SSBO 1
#define BINDINGS_GEOMETRIES_SSBO 2
#define BINDINGS_RENDERINSTANCES_SSBO 3
#define BINDINGS_SCENEBUILDING_SSBO 4
#define BINDINGS_SCENEBUILDING_UBO 5
#define BINDINGS_HIZ_TEX 6
#define BINDINGS_STREAMING_UBO 7
#define BINDINGS_STREAMING_SSBO 8
#define BINDINGS_RASTER_ATOMIC 9
// DLSS buffers start here as well
#define BINDINGS_RENDER_TARGET 10

/////////////////////////////////////////

#define BUILD_SETUP_TRAVERSAL_RUN 1
#define BUILD_SETUP_DRAW 2
#define BUILD_SETUP_BLAS_INSERTION 3

/////////////////////////////////////////

#define STREAM_SETUP_COMPACTION_OLD_NO_UNLOADS 0
#define STREAM_SETUP_COMPACTION_STATUS 1
#define STREAM_SETUP_ALLOCATOR_FREEINSERT 2
#define STREAM_SETUP_ALLOCATOR_STATUS 3

/////////////////////////////////////////

#define TRAVERSAL_PRESORT_WORKGROUP 128
#define TRAVERSAL_INIT_WORKGROUP 64
#define TRAVERSAL_RUN_WORKGROUP 64
#define TRAVERSAL_GROUPS_WORKGROUP 64
#define TRAVERSAL_BLAS_MERGING_WORKGROUP 64
#define BLAS_SETUP_INSERTION_WORKGROUP 64
#define BLAS_INSERT_CLUSTERS_WORKGROUP 64
#define INSTANCES_ASSIGN_BLAS_WORKGROUP 64
#define INSTANCES_CLASSIFY_LOD_WORKGROUP 64
#define BLAS_CACHING_SETUP_BUILD_WORKGROUP 64
#define BLAS_CACHING_SETUP_COPY_WORKGROUP 64

// must be power of 2
// 优化：根据现代GPU特性调整工作组大小，提高并行处理效率
// 对于NVIDIA GPU，warp size为32，工作组大小应是32的倍数
#define STREAM_UPDATE_SCENE_WORKGROUP 64         // 适中大小，适合更新场景操作
#define STREAM_AGEFILTER_GROUPS_WORKGROUP 96     // 稍微减小，提高线程利用率
#define STREAM_COMPACTION_NEW_CLAS_WORKGROUP 96   // 稍微减小，提高线程利用率
#define STREAM_COMPACTION_OLD_CLAS_WORKGROUP 64   // 适中大小，适合压缩操作
#define STREAM_ALLOCATOR_LOAD_GROUPS_WORKGROUP 64 // 适中大小，适合加载操作
#define STREAM_ALLOCATOR_UNLOAD_GROUPS_WORKGROUP 64 // 适中大小，适合卸载操作
#define STREAM_ALLOCATOR_BUILD_FREEGAPS_WORKGROUP 64 // 适中大小，适合构建空闲间隙
#define STREAM_ALLOCATOR_FREEGAPS_INSERT_WORKGROUP 64 // 适中大小，适合插入空闲间隙
#define STREAM_ALLOCATOR_SETUP_INSERTION_WORKGROUP 64 // 适中大小，适合设置插入

#define FORCE_INVISIBLE_CULLED_REMOVES_INSTANCE 1

#ifdef __cplusplus
namespace shaderio {
using namespace glm;

#else

#ifndef ALLOW_SHADING
#define ALLOW_SHADING 1
#endif

#ifndef ALLOW_VERTEX_NORMALS
#define ALLOW_VERTEX_NORMALS 1
#endif

#ifndef ALLOW_VERTEX_TEXCOORDS
#define ALLOW_VERTEX_TEXCOORDS 1
#endif

#ifndef ALLOW_VERTEX_TEXCOORD_1
#define ALLOW_VERTEX_TEXCOORD_1 1
#endif

#ifndef ALLOW_VERTEX_TANGENTS
#define ALLOW_VERTEX_TANGENTS 1
#endif

#ifndef USE_SW_RASTER
#define USE_SW_RASTER 0
#endif

#ifndef USE_RENDER_STATS
#define USE_RENDER_STATS 1
#endif

#ifndef USE_MEMORY_STATS
#define USE_MEMORY_STATS 1
#endif

#ifndef USE_CULLING
#define USE_CULLING 1
#endif

#ifndef USE_FORCED_INVISIBLE_CULLING
#define USE_FORCED_INVISIBLE_CULLING 1
#endif
//两遍剔除
#ifndef USE_TWO_PASS_CULLING
#define USE_TWO_PASS_CULLING 1
#endif

// 
#ifndef USE_PRIMITIVE_CULLING
#define USE_PRIMITIVE_CULLING 1
#endif

#ifndef USE_INSTANCE_SORTING
#define USE_INSTANCE_SORTING 1
#endif



#ifndef USE_STREAMING
#define USE_STREAMING 0
#endif

#ifndef USE_TWO_SIDED
#define USE_TWO_SIDED 1
#endif

#ifndef USE_FORCED_TWO_SIDED
#define USE_FORCED_TWO_SIDED 0
#endif

#ifndef MAX_VISIBLE_CLUSTERS
#define MAX_VISIBLE_CLUSTERS 1024
#endif

#ifndef TARGETS_RASTERIZATION
#define TARGETS_RASTERIZATION 1
#endif




#endif

struct FrameConstants
{
  mat4 projMatrix;
  mat4 projMatrixI;

  mat4 viewProjMatrix;
  mat4 viewProjMatrixI;
  mat4 viewMatrix;
  mat4 viewMatrixI;
  vec4 viewPos;
  vec4 viewDir;
  vec4 viewPlane;

  mat4 skyProjMatrixI;

  // for motion vectors
  mat4 viewProjMatrixPrev;

  ivec2 viewport;
  vec2  viewportf;

  vec2 viewPixelSize;
  vec2 viewClipSize;

  vec3  wLightPos;
  float lightMixer;

  vec3  wUpDir;
  float sceneSize;
  //uint  _pad1;
  //调试着色
  uint  colorXor;
  uint  visualize;
  float fov;

  float   nearPlane;
  float   farPlane;
  float   ambientOcclusionRadius;
  int32_t ambientOcclusionSamples;

  vec4 hizSizeFactors;
  vec4 nearSizeFactors;

  float hizSizeMax;
  int   facetShading;
  vec2  jitter;

  uint  dbgUint;
  float dbgFloat;
  uint  frame;
  uint  doShadow;

  vec4 bgColor;

  uvec2 mousePosition;
  float wireThickness;
  float wireSmoothing;

  vec3 wireColor;
  uint wireStipple;

  vec3  wireBackfaceColor;
  float wireStippleRepeats;

  float wireStippleLength;
  uint  doWireframe;
  uint  visFilterInstanceID;
  uint  visFilterClusterID;

  // LOD smooth transition
  float time;
  float deltaTime;
  float lodTransitionSpeed; // 控制LOD过渡速度

  SkySimpleParameters skyParams;
};

struct Readback
{
  uint     numRenderClusters;
  //要渲染的簇数
  uint     numRenderClustersSW;
  //遍历任务数
  uint     numTraversalTasks;
  //已遍历的任务数
  uint     numTraversedTasks;
  uint     numBlasBuilds;
  uint     numRenderedClusters;
  uint     numRenderedClustersSW;
  uint64_t numRenderedTriangles;
  uint64_t numRenderedTrianglesSW;
  uint64_t numRasteredTriangles;
  uint64_t numRasteredTrianglesSW;



#ifdef __cplusplus
  uint32_t clusterTriangleId;
  uint32_t _packedDepth0;

  uint32_t instanceId;
  uint32_t _packedDepth1;
#else
  uint64_t clusterTriangleId;
  uint64_t instanceId;
#endif

  uint64_t debugU64;

  int  debugI;
  uint debugUI;
  uint debugF;

  uint debugA[64];
  uint debugB[64];
  uint debugC[64];
};


#ifdef __cplusplus
}
#endif
#endif  // _SHADERIO_H_
