//==============================================================================
// 文件：src/meshlod/meshlod_local_indices.h
// 模块定位：簇 局部索引生成工具，把全局索引转换为局部顶点表和 8 位三角索引。
// 数据流：输入是全局 index 缓冲 与 簇 三角形范围；输出是局部顶点重映射和紧凑三角索引。
// 方法说明：局部索引显著降低 组 数据体积，并使 着色器 可以用小整数索引访问 簇 内顶点。
// 正确性约束：簇 顶点数量必须不超过 8 位索引可表达范围；局部重映射必须保持三角形拓扑等价。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#pragma once


// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include "meshlod_types.h"
#include <cassert>


// 函数：clodLocalIndices。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
size_t clodLocalIndices(unsigned int* vertices, unsigned char* triangles, const unsigned int* indices, size_t index_count)
{
	size_t unique = 0;

	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int v = indices[i];
		size_t local = 0;

		for (; local < unique; ++local)
			if (vertices[local] == v)
				break;

		if (local == unique)
			vertices[unique++] = v;


		assert(local < 256);
		triangles[i] = (unsigned char)local;
	}

	return unique;
}
