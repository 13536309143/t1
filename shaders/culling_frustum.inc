/*

视锥体剔除相关函数
- intersectSize - 计算包围盒在屏幕上的大小
- getClip - 执行透视除法，转换为标准设备坐标
- getCullBits - 位运算视锥体剔除方法
- getBoxCorner - 获取 3D 包围盒的顶点
- intersectFrustum - 测试包围盒是否与视锥体相交
*/

bool intersectSize(vec4 clipMin, vec4 clipMax, float threshold)
{
  // 根据当前帧的视口分辨率(view.viewportf.xy)，计算裁剪空间包围盒在屏幕上的实际像素大小
  // clip 坐标范围是 [-1, 1]，长度为 2。所以乘以 0.5 映射到 [0, 1]，再乘以视口宽高
  //vec2 rect = (clipMax.xy - clipMin.xy) * 0.5 * viewLast.viewportf.xy;根据当前帧的视口计算投影大小，而不是前一帧的
  vec2 rect = (clipMax.xy - clipMin.xy) * 0.5 * view.viewportf.xy;
  vec2 clipThreshold = vec2(threshold);
  // 如果包围盒的宽或高，任意一个大于指定的阈值，就返回 true
  return any(greaterThan(rect,clipThreshold));
}

// 将齐次裁剪坐标 (x, y, z, w) 执行透视除法，转换为标准设备坐标 (NDC)
vec4 getClip(vec4 hPos, out bool valid) {
// 检查 w 是否合法（不能太过接近 0）
  valid = !(-c_epsilon < hPos.w && hPos.w < c_epsilon);
  // 返回除以 w 后的坐标。注意这里除以了 abs(w)，是为了处理相机后面的点
  return vec4(hPos.xyz / abs(hPos.w), hPos.w);
}

// 高效的位运算视锥体剔除方法，为点生成一个 7 位的状态掩码
// 如果一个点在视锥体的某个面之外，对应的位就会被置为 1
uint getCullBits(vec4 hPos)
{
  uint cullBits = 0;
  cullBits |= hPos.x < -hPos.w ?  1 : 0; // 左裁剪面外
  cullBits |= hPos.x >  hPos.w ?  2 : 0; // 右裁剪面外
  cullBits |= hPos.y < -hPos.w ?  4 : 0; // 下裁剪面外
  cullBits |= hPos.y >  hPos.w ?  8 : 0; // 上裁剪面外
  cullBits |= hPos.z <  0      ? 16 : 0; // 近裁剪面外 (Vulkan/DX 的 Z 是 0~W)
  cullBits |= hPos.z >  hPos.w ? 32 : 0; // 远裁剪面外
  cullBits |= hPos.w <= 0      ? 64 : 0; // 跑到相机平面后面了
  return cullBits;
}

// 根据索引 n (0~7) 获取 3D 包围盒(AABB)的 8 个顶点中的一个
vec4 getBoxCorner(vec3 bboxMin, vec3 bboxMax, int n)
{
// 使用二进制位 (000 到 111) 来决定当前顶点的 xyz 是取 Min 还是 Max
  bvec3 useMax = bvec3((n & 1) != 0, (n & 2) != 0, (n & 4) != 0);
  return vec4(mix(bboxMin, bboxMax, useMax),1);
}

/*
可以传入不同的 viewProjMatrix（当前帧或前一帧）
双遍剔除策略：
第一遍：输入 viewLast.viewProjMatrix（使用前一帧的视图）
第二遍：输入 view.viewProjMatrix（使用当前帧的视图）
*/
//bool intersectFrustum(vec3 bboxMin, vec3 bboxMax, mat4x3 worldTM, out vec4 oClipmin, out vec4 oClipmax, out bool oClipvalid)
// 测试包围盒是否与视锥体相交，并输出它在屏幕上的 2D 投影范围 (oClipmin/max)
bool intersectFrustum(mat4 viewProjMatrix, vec3 bboxMin, vec3 bboxMax, mat4x3 worldTM, out vec4 oClipmin, out vec4 oClipmax, out bool oClipvalid)
{
  //mat4 worldViewProjTM = viewLast.viewProjMatrix * toMat4(worldTM);
  // 计算从世界空间到裁剪空间的变换矩阵
  mat4 worldViewProjTM = viewProjMatrix * toMat4(worldTM);
  bool valid;
  // 计算第 0 个顶点的齐次坐标、NDC 坐标以及它在视锥体中的状态位
  vec4 hPos     = worldViewProjTM * getBoxCorner(bboxMin, bboxMax, 0);
  vec4 clip     = getClip(hPos, valid);
  uint bits     = getCullBits(hPos);
  // 初始化包围盒的屏幕 2D 投影范围
  vec4 clipMin  = clip;
  vec4 clipMax  = clip;
  bool clipValid = valid;
  // 展开循环：遍历剩余的 7 个顶点
  [[unroll]]
  for (int n = 1; n < 8; n++){
    hPos  = worldViewProjTM * getBoxCorner(bboxMin, bboxMax, n);
    clip  = getClip(hPos, valid);
    // 关键位运算：将所有点的状态位进行"按位与(&)"
    // 如果结果不为 0，说明这 8 个顶点全部位于视锥体的同一个面的外侧，即完全不可见
    bits &= getCullBits(hPos);
    // 扩展包围盒在屏幕上的 2D 投影边界
    clipMin = min(clipMin,clip);
    clipMax = max(clipMax,clip);

    clipValid = clipValid && valid;
  }

  oClipvalid = clipValid;
  // 将最终的 2D 投影范围裁剪(Clamp)到屏幕视口范围 [-1, 1]
  oClipmin = vec4(clamp(clipMin.xy, vec2(-1), vec2(1)), clipMin.zw);
  oClipmax = vec4(clamp(clipMax.xy, vec2(-1), vec2(1)), clipMax.zw);
  // 如果 bits == 0，说明包围盒和视锥体有交集（可见），返回 true
  //return true;
  return bits == 0;
}
