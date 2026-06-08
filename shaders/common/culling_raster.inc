//==============================================================================
// 文件：shaders/common/culling_raster.inc
// 模块定位：着色器公共函数片段，集中提供属性编码、剔除、屏幕空间估计和着色辅助逻辑。
// 数据流：多个计算、网格和片元阶段通过 include 复用这些函数，避免同一数学逻辑在不同阶段分叉。
// 方法说明：公共函数将几何、可见性和材质计算标准化，使 traversal 与 render 对同一对象得到一致判断。
// 正确性约束：公共函数不能依赖某个单独 阶段 的私有状态；所有宏开关都应有明确默认值。
// 注释风格：使用中文解释 GPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
// 结构：RasterVertex。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct RasterVertex {
  vec2  xy;
  float z;
};


// 函数：getScreenPos。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
vec2 getScreenPos(vec4 hPos)
{
  return vec2(((hPos.xy/hPos.w) * 0.5 + 0.5) * view.viewportf.xy);
}


// 函数：getRasterVertex。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
RasterVertex getRasterVertex(vec4 hPos)
{
  RasterVertex vtx;

  vtx.xy       = getScreenPos(hPos);
  vtx.z        = hPos.z/hPos.w;

  return vtx;
}


// 函数：isFrontFacingSW。返回条件判断结果，用于调用方选择后续分支或验证输入状态。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：谓词函数应保持无副作用或低副作用，使调用方可以安全地把它用于断言、过滤和早退。
bool isFrontFacingSW(RasterVertex a, RasterVertex b, RasterVertex c)
{
  vec2 ab = b.xy - a.xy;
  vec2 ac = c.xy - a.xy;

  float cross_product = ab.y * ac.x - ab.x * ac.y;
  return cross_product >= 0;
}


// 函数：isFrontFacingHW。返回条件判断结果，用于调用方选择后续分支或验证输入状态。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：谓词函数应保持无副作用或低副作用，使调用方可以安全地把它用于断言、过滤和早退。
bool isFrontFacingHW(vec4 ha, vec4 hb, vec4 hc)
{

  return determinant(mat3(ha.xyw, hb.xyw, hc.xyw)) <= 0;
}


// 函数：pixelBboxEpsilon。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void pixelBboxEpsilon(inout vec2 pixelMin, inout vec2 pixelMax)
{


  const float epsilon = (1.0 / 256);
  pixelMin -= epsilon;
  pixelMax += epsilon;


  pixelMin = round(pixelMin);

  pixelMax = round(pixelMax);
}


// 函数：pixelBboxCull。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
bool pixelBboxCull(vec2 pixelMin, vec2 pixelMax){


  bool cull = ( ( pixelMin.x == pixelMax.x) || ( pixelMin.y == pixelMax.y));
  return cull;
}


// 函数：pixelViewportCull。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
bool pixelViewportCull(vec2 pixelMin, vec2 pixelMax)
{
  return ((pixelMax.x < 0) || (pixelMin.x >= view.viewportf.x) || (pixelMax.y < 0) || (pixelMin.y >= view.viewportf.y));
}


// 函数：testTrianglePixel。返回条件判断结果，用于调用方选择后续分支或验证输入状态。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：谓词函数应保持无副作用或低副作用，使调用方可以安全地把它用于断言、过滤和早退。
bool testTrianglePixel(vec2 a, vec2 b, vec2 c, bool frustum, out vec2 pixelMin, out vec2 pixelMax)
{


  pixelMin = min(a,min(b,c));
  pixelMax = max(a,max(b,c));


  pixelBboxEpsilon(pixelMin, pixelMax);


  if (frustum && pixelViewportCull(pixelMin, pixelMax)) return false;

  if (pixelBboxCull(pixelMin, pixelMax)) return false;
  return true;
}


// 函数：testTriangleSW。返回条件判断结果，用于调用方选择后续分支或验证输入状态。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：谓词函数应保持无副作用或低副作用，使调用方可以安全地把它用于断言、过滤和早退。
bool testTriangleSW(RasterVertex a, RasterVertex b, RasterVertex c, out vec2 pixelMin, out vec2 pixelMax, out float triArea)
{

  vec2 ab = b.xy - a.xy;
  vec2 ac = c.xy - a.xy;
  float cross_product = ab.y * ac.x - ab.x * ac.y;

  triArea = cross_product;
#if !USE_FORCED_TWO_SIDED
  if (cross_product < 0) return false;
#endif

  return testTrianglePixel(a.xy, b.xy, c.xy, false, pixelMin, pixelMax);
}


// 函数：testTriangleHW。返回条件判断结果，用于调用方选择后续分支或验证输入状态。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：谓词函数应保持无副作用或低副作用，使调用方可以安全地把它用于断言、过滤和早退。
bool testTriangleHW(vec4 ha, vec4 hb, vec4 hc)
{
#if !USE_FORCED_TWO_SIDED
  if (!isFrontFacingHW(ha,hb,hc))
  {
    return false;
  }
#endif


  RasterVertex a = getRasterVertex(ha);

  RasterVertex b = getRasterVertex(hb);

  RasterVertex c = getRasterVertex(hc);

  vec2 pixelMin;
  vec2 pixelMax;

  return testTrianglePixel(a.xy, b.xy, c.xy, false, pixelMin, pixelMax);
}
