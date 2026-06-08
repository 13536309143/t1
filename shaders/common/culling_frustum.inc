//==============================================================================
// 文件：shaders/common/culling_frustum.inc
// 模块定位：着色器公共函数片段，集中提供属性编码、剔除、屏幕空间估计和着色辅助逻辑。
// 数据流：多个计算、网格和片元阶段通过 include 复用这些函数，避免同一数学逻辑在不同阶段分叉。
// 方法说明：公共函数将几何、可见性和材质计算标准化，使 traversal 与 render 对同一对象得到一致判断。
// 正确性约束：公共函数不能依赖某个单独 阶段 的私有状态；所有宏开关都应有明确默认值。
// 注释风格：使用中文解释 GPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
// 函数：intersectSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
bool intersectSize(vec4 clipMin, vec4 clipMax, float threshold)
{


  vec2 rect = (clipMax.xy - clipMin.xy) * 0.5 * view.viewportf.xy;

  vec2 clipThreshold = vec2(threshold);

  return any(greaterThan(rect,clipThreshold));
}


// 函数：getClip。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
vec4 getClip(vec4 hPos, out bool valid) {

  valid = !(-c_epsilon < hPos.w && hPos.w < c_epsilon);

  return vec4(hPos.xyz / abs(hPos.w), hPos.w);
}


// 函数：getCullBits。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
uint getCullBits(vec4 hPos)
{
  uint cullBits = 0;
  cullBits |= hPos.x < -hPos.w ?  1 : 0;
  cullBits |= hPos.x >  hPos.w ?  2 : 0;
  cullBits |= hPos.y < -hPos.w ?  4 : 0;
  cullBits |= hPos.y >  hPos.w ?  8 : 0;
  cullBits |= hPos.z <  0      ? 16 : 0;
  cullBits |= hPos.z >  hPos.w ? 32 : 0;
  cullBits |= hPos.w <= 0      ? 64 : 0;
  return cullBits;
}


// 函数：getBoxCorner。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
vec4 getBoxCorner(vec3 bboxMin, vec3 bboxMax, int n)
{

  bvec3 useMax = bvec3((n & 1) != 0, (n & 2) != 0, (n & 4) != 0);
  return vec4(mix(bboxMin, bboxMax, useMax),1);
}


// 函数：intersectFrustum。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
bool intersectFrustum(mat4 viewProjMatrix, vec3 bboxMin, vec3 bboxMax, mat4x3 worldTM, out vec4 oClipmin, out vec4 oClipmax, out bool oClipvalid)
{


  mat4 worldViewProjTM = viewProjMatrix * toMat4(worldTM);
  bool valid;


  vec4 hPos     = worldViewProjTM * getBoxCorner(bboxMin, bboxMax, 0);

  vec4 clip     = getClip(hPos, valid);

  uint bits     = getCullBits(hPos);

  vec4 clipMin  = clip;
  vec4 clipMax  = clip;
  bool clipValid = valid;

  [[unroll]]
  for (int n = 1; n < 8; n++){

    hPos  = worldViewProjTM * getBoxCorner(bboxMin, bboxMax, n);

    clip  = getClip(hPos, valid);


    bits &= getCullBits(hPos);


    clipMin = min(clipMin,clip);

    clipMax = max(clipMax,clip);

    clipValid = clipValid && valid;
  }

  oClipvalid = clipValid;

  oClipmin = vec4(clamp(clipMin.xy, vec2(-1), vec2(1)), clipMin.zw);
  oClipmax = vec4(clamp(clipMax.xy, vec2(-1), vec2(1)), clipMax.zw);


  return bits == 0;
}
