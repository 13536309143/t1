//==============================================================================
// 文件：src/meshlod/meshlod.cpp
// 模块定位：meshlod 编译单元，汇入头文件实现并让公共 API 以目标文件方式参与链接。
// 数据流：输入来自包含的实现头；输出是链接器可解析的 clodBuild、clodBuild_iterationTask 和 clodLocalIndices 符号。
// 方法说明：将模板或内联密集的算法实现集中实例化，可减少调用方重复编译成本。
// 正确性约束：不要在此处引入与算法无关的全局状态，否则会破坏 meshlod 的独立性。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <meshoptimizer.h>
#include "meshlod.h"
#include "meshlod_bounds.h"
#include "meshlod_clustering.h"
#include "meshlod_simplify.h"
#include "meshlod_build.h"
#include "meshlod_local_indices.h"
