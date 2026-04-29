/*

软件和硬件光栅化相关函数
- RasterVertex 结构体 - 用于软件光栅化的顶点结构
- getScreenPos - 转换裁剪坐标为屏幕像素坐标
- getRasterVertex - 获取光栅化顶点
- isFrontFacingSW - 软件模式的背面剔除
- isFrontFacingHW - 硬件模式的背面剔除
- pixelBboxEpsilon - 亚像素误差处理
- pixelBboxCull - 像素包围盒剔除
- pixelViewportCull - 视口越界裁剪
- testTrianglePixel - 基础的三角形像素级测试
- testTriangleSW - 软件光栅的三角形综合测试
- testTriangleHW - 硬件光栅的三角形综合测试
*/

// 用于软件光栅化的 2D 顶点结构
struct RasterVertex {
  vec2  xy;// 屏幕像素坐标
  float z;// 深度
};

// 把裁剪坐标转换为实际的屏幕像素坐标 (例如 1920x1080)
vec2 getScreenPos(vec4 hPos)
{
  return vec2(((hPos.xy/hPos.w) * 0.5 + 0.5) * view.viewportf.xy);
}

RasterVertex getRasterVertex(vec4 hPos)
{
  RasterVertex vtx;
  vtx.xy       = getScreenPos(hPos);
  vtx.z        = hPos.z/hPos.w;

  return vtx;
}

// 软件模式（基于 2D 屏幕坐标）的背面剔除
bool isFrontFacingSW(RasterVertex a, RasterVertex b, RasterVertex c)
{
  vec2 ab = b.xy - a.xy;
  vec2 ac = c.xy - a.xy;
  // 使用 2D 向量的叉积来判断三角形是顺时针还是逆时针（正面还是背面）
  float cross_product = ab.y * ac.x - ab.x * ac.y;
  return cross_product >= 0;
}

// 硬件模式（基于 3D 齐次坐标）的背面剔除
bool isFrontFacingHW(vec4 ha, vec4 hb, vec4 hc)
{
// 直接使用 3x3 矩阵的行列式符号来判断绕向，这是一种数学上等价且非常适合 GPU 的做法
  return determinant(mat3(ha.xyw, hb.xyw, hc.xyw)) <= 0;
}

//亚像素剔除 (Sub-pixel Culling)
void pixelBboxEpsilon(inout vec2 pixelMin, inout vec2 pixelMax)
{
  // apply some safety around the bbox to take into account fixed point rasterization
  // (our rasterization grid is 1/256)
  // 给像素包围盒加上一点误差（1/256 像素），这是为了应对光栅化器的定点精度问题
  const float epsilon = (1.0 / 256);
  pixelMin -= epsilon;
  pixelMax += epsilon;
  // 四舍五入到最近的整数像素边界
  pixelMin = round(pixelMin);
  pixelMax = round(pixelMax);
}

bool pixelBboxCull(vec2 pixelMin, vec2 pixelMax){
  // bbox culling
  // 如果三角形的包围盒在 X 轴的最小像素等于最大像素（宽度为0），或 Y 轴高度为 0
  // 说明它无法覆盖任何像素的中心，直接剔除（Sub-pixel Culling）
  bool cull = ( ( pixelMin.x == pixelMax.x) || ( pixelMin.y == pixelMax.y));
  return cull;
}

// 视口越界剔除：如果整个像素包围盒跑到屏幕范围外了，剔除
bool pixelViewportCull(vec2 pixelMin, vec2 pixelMax)
{
  return ((pixelMax.x < 0) || (pixelMin.x >= view.viewportf.x) || (pixelMax.y < 0) || (pixelMin.y >= view.viewportf.y));
}

// 基础的三角形像素级测试
bool testTrianglePixel(vec2 a, vec2 b, vec2 c, bool frustum, out vec2 pixelMin, out vec2 pixelMax)
{
  // compute the min and max in each X and Y direction
  // 计算三角形在屏幕上的 AABB
  pixelMin = min(a,min(b,c));
  pixelMax = max(a,max(b,c));
  // 应用亚像素误差
  pixelBboxEpsilon(pixelMin, pixelMax);
  // 如果需要做视锥体剔除，并且确实超出了视口，返回 false (不可见)
  // (通常在 Cluster 级别已经做过视锥体剔除了，这里看情况使用)
  // we may not test against frustum / viewport, given the hierarchical culling was doing that on a cluster level already
  if (frustum && pixelViewportCull(pixelMin, pixelMax)) return false;
  // 亚像素测试，如果面积为 0，返回 false (不可见)
  if (pixelBboxCull(pixelMin, pixelMax)) return false;
  return true;
}

// 用于 Compute Shader 软件光栅的三角形综合测试
bool testTriangleSW(RasterVertex a, RasterVertex b, RasterVertex c, out vec2 pixelMin, out vec2 pixelMax, out float triArea)
{
  // 计算背面剔除
  vec2 ab = b.xy - a.xy;
  vec2 ac = c.xy - a.xy;
  float cross_product = ab.y * ac.x - ab.x * ac.y;

  triArea = cross_product;
#if !USE_FORCED_TWO_SIDED// 如果没有强制双面渲染
  if (cross_product < 0) return false;// 它是背面，剔除
#endif
// 进行像素包围盒测试
  return testTrianglePixel(a.xy, b.xy, c.xy, false, pixelMin, pixelMax);
}

// 用于 Mesh Shader / 硬件光栅化前的三角形综合测试
bool testTriangleHW(vec4 ha, vec4 hb, vec4 hc)
{
#if !USE_FORCED_TWO_SIDED
  if (!isFrontFacingHW(ha,hb,hc))
  {
    return false;
  }
#endif
// 转换为屏幕坐标
  RasterVertex a = getRasterVertex(ha);
  RasterVertex b = getRasterVertex(hb);
  RasterVertex c = getRasterVertex(hc);

  vec2 pixelMin;
  vec2 pixelMax;
  // 进行像素包围盒测试
  return testTrianglePixel(a.xy, b.xy, c.xy, false, pixelMin, pixelMax);
}
