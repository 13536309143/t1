//==============================================================================
// 文件：shaders/post/fullscreen_atomic.frag.glsl
// 模块定位：后处理着色器，负责全屏背景、软件光栅结果合并和 Hi-Z 层级深度生成。
// 数据流：读取 帧缓冲、atomic raster image 或 depth texture，输出最终颜色或下一帧剔除使用的层级深度。
// 方法说明：后处理阶段把异步生成的中间结果规约为统一图像，并为下一帧可见性判断建立反馈。
// 正确性约束：atomic resolve 的深度比较语义要与写入端一致；Hi-Z 归约必须保守，不能错误剔除可见对象。
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
#extension GL_EXT_shader_image_int64 : enable


// 依赖说明：引入共享布局、剔除、着色或阶段间复用的着色器片段。
// 这些 include 共同决定本文件能访问的结构布局、数学辅助函数和编译期宏。
#include "shaderio.h"


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(set = 0, binding = BINDINGS_RASTER_ATOMIC, r64ui) uniform u64image2D imgRasterAtomic;


// 绑定布局说明：声明本阶段访问的描述符、推送常量、输入输出或工作组配置。
// 这些声明构成 Vulkan pipeline layout 与 GLSL 代码之间的显式契约。
layout(location = 0, index = 0) out vec4 out_Color;


// 函数：main。作为本着色器阶段入口，按绑定资源执行当前 GPU 工作。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该入口位于控制流根部，调用顺序决定后续资源生命周期和数据依赖。
void main()
{

  ivec2 coord  = ivec2(gl_FragCoord.xy);
  uvec2 loaded = unpackUint2x32(imageLoad(imgRasterAtomic, coord).x);

  gl_FragDepth = loaded.y == uint(0) ? 0.0 : uintBitsToFloat(loaded.y);

  out_Color    = loaded.y == uint(0) ? vec4(0,0,0,1) : unpackUnorm4x8(loaded.x);
}
