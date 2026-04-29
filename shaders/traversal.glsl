#define FLT_MAX 3.402823466e+38f
TraversalInfo unpackTraversalInfo(uint64_t packed64)
{
  u32vec2       data = unpack32(packed64);
  TraversalInfo info;
  info.instanceID = data.x;
  info.packedNode = data.y;
  return info;
}
uint64_t packTraversalInfo(TraversalInfo info)
{
  return pack64(u32vec2(info.instanceID, info.packedNode));
}

float computeUniformScale(mat4 transform)
{
  return max(max(length(vec3(transform[0])), length(vec3(transform[1]))), length(vec3(transform[2])));
}

float computeUniformScale(mat4x3 transform)
{
  return max(max(length(vec3(transform[0])), length(vec3(transform[1]))), length(vec3(transform[2])));
}

vec3 TraversalMetric_getSphere(TraversalMetric metric)
{
  return vec3(metric.boundingSphereX, metric.boundingSphereY, metric.boundingSphereZ);
}
void TraversalMetric_setSphere(inout TraversalMetric metric, vec3 sphere)
{
  metric.boundingSphereX = sphere.x;
  metric.boundingSphereY = sphere.y;
  metric.boundingSphereZ = sphere.z;
}
/*
LOD 的切换并非基于离散的距离，而是基于屏幕空间误差。
在 traversal.glsl 中，testForTraversal 函数是 LOD 评估的核心：

OpenGL Shading Language
float errorDistance = max(minDistance, sphereDistance - metric.boundingSphereRadius * uniformScale);
float errorOverDistance = metric.maxQuadricError * uniformScale / errorDistance;
return errorOverDistance >= build.errorOverDistanceThreshold * errorScale;
通过计算几何体简化所带来的 Quadric Error 在屏幕上的投影像素大小，如果误差小于设定的阈值，则停止下钻（Descend）LOD树，直接将当前层级的 Cluster 提交渲染。
*/

// key function for the lod metric evaluation
// returns true if error is over threshold ("coarse enough")
bool testForTraversal(mat4x3 instanceToEye, float uniformScale, TraversalMetric metric, float errorScale)
{
  vec3  boundingSpherePos = vec3(metric.boundingSphereX, metric.boundingSphereY, metric.boundingSphereZ);
  float minDistance       = view.nearPlane;
  float sphereDistance    = length(vec3(instanceToEye * vec4(boundingSpherePos, 1.0f)));
  float errorDistance     = max(minDistance, sphereDistance - metric.boundingSphereRadius * uniformScale);
  float errorOverDistance = metric.maxQuadricError * uniformScale / errorDistance;
  
  // error is over threshold, we are coarse enough
  return errorOverDistance >= build.errorOverDistanceThreshold * errorScale;
}

// LOD smooth transition function
// Returns a transition factor [0, 1] between current LOD and next LOD
float computeLodTransitionFactor(float currentError, float nextError, float threshold)
{
  // Calculate transition range
  float transitionStart = threshold * 0.8;
  float transitionEnd = threshold * 1.2;
  
  // Calculate error ratio
  float errorRatio = (currentError + nextError) * 0.5;
  
  // Compute smooth transition factor using smoothstep
  return smoothstep(transitionStart, transitionEnd, errorRatio);
}

// Enhanced LOD selection with smooth transition
float evaluateLodTransition(mat4x3 instanceToEye, float uniformScale, TraversalMetric currentMetric, TraversalMetric nextMetric, float errorScale)
{
  vec3  boundingSpherePos = vec3(currentMetric.boundingSphereX, currentMetric.boundingSphereY, currentMetric.boundingSphereZ);
  float minDistance       = view.nearPlane;
  float sphereDistance    = length(vec3(instanceToEye * vec4(boundingSpherePos, 1.0f)));
  float errorDistance     = max(minDistance, sphereDistance - currentMetric.boundingSphereRadius * uniformScale);
  
  float currentError = currentMetric.maxQuadricError * uniformScale / errorDistance;
  float nextError = nextMetric.maxQuadricError * uniformScale / errorDistance;
  
  return computeLodTransitionFactor(currentError, nextError, build.errorOverDistanceThreshold * errorScale);
}

// variant of the above, assumes world space for view position and metric sphere position
bool testForTraversal(vec3 wViewPos, float uniformScale, TraversalMetric metric, float errorScale)
{
  vec3  boundingSpherePos = vec3(metric.boundingSphereX, metric.boundingSphereY, metric.boundingSphereZ);
  float minDistance       = view.nearPlane;
  float sphereDistance    = length(wViewPos - boundingSpherePos);
  float errorDistance     = max(minDistance, sphereDistance - metric.boundingSphereRadius * uniformScale);
  float errorOverDistance = metric.maxQuadricError * uniformScale / errorDistance;
  
  // error is over threshold, we are coarse enough
  return errorOverDistance >= build.errorOverDistanceThreshold * errorScale;
}

