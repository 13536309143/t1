//==============================================================================
// 文件：src/meshlod/meshlod_bounds.h
// 模块定位：几何范围辅助，用于计算顶点包围盒、合并范围并为简化误差提供空间尺度。
// 数据流：输入是顶点数组或已有 bounds；输出是 clodBounds 或合并后的范围。
// 方法说明：空间范围是 LOD 误差归一化和剔除层次构建的基础，必须在算法层保持轻量且无渲染依赖。
// 正确性约束：空输入要有可定义行为；合并操作应保持单调扩张，不得缩小已有范围。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#pragma once


// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include "meshlod_impl.h"

namespace clod
{


// 函数：boundsCompute。计算派生值，供后续剔除、LOD、统计或资源规划使用。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：计算结果通常参与阈值比较或内存规划，数值稳定性和边界条件需要特别注意。
clodBounds boundsCompute(const clodMesh& mesh, const std::vector<unsigned int>& indices, float error)
{
	meshopt_Bounds bounds = meshopt_computeClusterBounds(&indices[0], indices.size(), mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride);
	clodBounds result;
	result.center[0] = bounds.center[0];
	result.center[1] = bounds.center[1];
	result.center[2] = bounds.center[2];
	result.radius = bounds.radius;
	result.error = error;
	return result;
}


// 函数：boundsMerge。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
clodBounds boundsMerge(const std::vector<Cluster>& clusters, const std::vector<int>& group)
{


	// 函数：bounds。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<clodBounds> bounds(group.size());
	for (size_t j = 0; j < group.size(); ++j)
		bounds[j] = clusters[group[j]].bounds;
	meshopt_Bounds merged = meshopt_computeSphereBounds(&bounds[0].center[0], bounds.size(), sizeof(clodBounds), &bounds[0].radius, sizeof(clodBounds));
	clodBounds result = {};
	result.center[0] = merged.center[0];
	result.center[1] = merged.center[1];
	result.center[2] = merged.center[2];
	result.radius = merged.radius;
	result.error = 0.f;
	for (size_t j = 0; j < group.size(); ++j)

		result.error = std::max(result.error, clusters[group[j]].bounds.error);

	return result;
}

}
