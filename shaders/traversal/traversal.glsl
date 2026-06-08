//==============================================================================
// 文件：shaders/traversal/traversal.glsl
// 模块定位：LOD 遍历着色器，负责从实例层次中选择本帧需要渲染或请求加载的 簇。
// 数据流：读取实例、几何层次、Hi-Z 和 流式加载 地址，输出 traversal queue、组 queue、render 簇 list 和 request。
// 方法说明：遍历阶段把屏幕空间误差、视锥剔除和遮挡剔除合并为并行剪枝问题，以减少后续光栅工作量。
// 正确性约束：队列计数必须原子更新；流式加载 地址无效时只能发请求，不能解引用；two-阶段 状态必须区分上一帧和当前帧 Hi-Z。
// 注释风格：使用中文解释 GPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
// 宏配置说明：定义编译期常量或功能开关，让 CPU 与 GPU 按同一套布局和路径工作。
// 宏值通常会影响 buffer 大小、工作组规模或条件编译分支，修改后需要同时检查 C++ 和着色器侧。
#define FLT_MAX 3.402823466e+38f


// 函数：unpackTraversalInfo。在紧凑编码和逻辑结构之间转换，减少带宽或便于着色器访问。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：编码位宽、符号位和特殊值必须与写入端/读取端完全一致，否则会产生难以定位的跨阶段错误。
TraversalInfo unpackTraversalInfo(uint64_t packed64)
{

  u32vec2       data = unpack32(packed64);
  TraversalInfo info;
  info.instanceID = data.x;
  info.packedNode = data.y;
  return info;
}


// 函数：packTraversalInfo。在紧凑编码和逻辑结构之间转换，减少带宽或便于着色器访问。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：编码位宽、符号位和特殊值必须与写入端/读取端完全一致，否则会产生难以定位的跨阶段错误。
uint64_t packTraversalInfo(TraversalInfo info)
{
  return pack64(u32vec2(info.instanceID, info.packedNode));
}


// 函数：computeUniformScale。计算派生值，供后续剔除、LOD、统计或资源规划使用。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：计算结果通常参与阈值比较或内存规划，数值稳定性和边界条件需要特别注意。
float computeUniformScale(mat4 transform)
{
  return max(max(length(vec3(transform[0])), length(vec3(transform[1]))), length(vec3(transform[2])));
}


// 函数：computeUniformScale。计算派生值，供后续剔除、LOD、统计或资源规划使用。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：计算结果通常参与阈值比较或内存规划，数值稳定性和边界条件需要特别注意。
float computeUniformScale(mat4x3 transform)
{
  return max(max(length(vec3(transform[0])), length(vec3(transform[1]))), length(vec3(transform[2])));
}


// 函数：TraversalMetric_getSphere。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
vec3 TraversalMetric_getSphere(TraversalMetric metric)
{
  return vec3(metric.boundingSphereX, metric.boundingSphereY, metric.boundingSphereZ);
}


// 函数：TraversalMetric_setSphere。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void TraversalMetric_setSphere(inout TraversalMetric metric, vec3 sphere)
{
  metric.boundingSphereX = sphere.x;
  metric.boundingSphereY = sphere.y;
  metric.boundingSphereZ = sphere.z;
}


// 函数：testForTraversal。返回条件判断结果，用于调用方选择后续分支或验证输入状态。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：谓词函数应保持无副作用或低副作用，使调用方可以安全地把它用于断言、过滤和早退。
bool testForTraversal(mat4x3 instanceToEye, float uniformScale, TraversalMetric metric, float errorScale)
{

  vec3  boundingSpherePos = vec3(metric.boundingSphereX, metric.boundingSphereY, metric.boundingSphereZ);
  float minDistance       = view.nearPlane;
  float sphereDistance    = length(vec3(instanceToEye * vec4(boundingSpherePos, 1.0f)));

  float errorDistance     = max(minDistance, sphereDistance - metric.boundingSphereRadius * uniformScale);
  float errorOverDistance = metric.maxQuadricError * uniformScale / errorDistance;


  return errorOverDistance >= build.errorOverDistanceThreshold * errorScale;
}


// 函数：computeLodTransitionFactor。计算派生值，供后续剔除、LOD、统计或资源规划使用。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：计算结果通常参与阈值比较或内存规划，数值稳定性和边界条件需要特别注意。
float computeLodTransitionFactor(float currentError, float nextError, float threshold)
{

  float transitionStart = threshold * 0.8;
  float transitionEnd = threshold * 1.2;


  float errorRatio = (currentError + nextError) * 0.5;


  return smoothstep(transitionStart, transitionEnd, errorRatio);
}


// 函数：evaluateLodTransition。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
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


// 函数：testForTraversal。返回条件判断结果，用于调用方选择后续分支或验证输入状态。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：谓词函数应保持无副作用或低副作用，使调用方可以安全地把它用于断言、过滤和早退。
bool testForTraversal(vec3 wViewPos, float uniformScale, TraversalMetric metric, float errorScale)
{

  vec3  boundingSpherePos = vec3(metric.boundingSphereX, metric.boundingSphereY, metric.boundingSphereZ);
  float minDistance       = view.nearPlane;

  float sphereDistance    = length(wViewPos - boundingSpherePos);

  float errorDistance     = max(minDistance, sphereDistance - metric.boundingSphereRadius * uniformScale);
  float errorOverDistance = metric.maxQuadricError * uniformScale / errorDistance;


  return errorOverDistance >= build.errorOverDistanceThreshold * errorScale;
}
