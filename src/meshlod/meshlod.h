//==============================================================================
// 文件：src/meshlod/meshlod.h
// 模块定位：簇 LOD 构建算法公共 API，向 C 和 C++ 调用方暴露构建、迭代任务和局部索引生成入口。
// 数据流：输入是 clodMesh 与 clodConfig；输出通过回调产生 clodCluster 和 clodGroup。
// 方法说明：API 采用回调式输出，便于算法层与 Scene 存储层解耦，也方便并行任务逐步提交构建结果。
// 正确性约束：回调接收的数据只在调用期间保证有效；调用方需要按配置上限分配或转存输出。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#pragma once


// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include "meshlod_types.h"
#include "meshlod_config.h"

#ifdef __cplusplus
extern "C"
{
#endif


// 函数：clodDefaultConfig。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
clodConfig clodDefaultConfig(size_t max_triangles);


// 函数：clodBuild。构建派生数据结构，通常用于 LOD、层次结构、间接命令或加速访问。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：构建结果会被后续阶段高频读取，必须保证布局紧凑、索引合法并与共享结构定义一致。
size_t clodBuild(clodConfig config, clodMesh mesh, void* output_context, clodOutput output_callback, clodIteration iteration_callback);


// 函数：clodBuild_iterationTask。构建派生数据结构，通常用于 LOD、层次结构、间接命令或加速访问。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：构建结果会被后续阶段高频读取，必须保证布局紧凑、索引合法并与共享结构定义一致。
void clodBuild_iterationTask(void* iteration_context, void* output_context, size_t task_index, unsigned int thread_index);


// 函数：clodLocalIndices。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
size_t clodLocalIndices(unsigned int* vertices, unsigned char* triangles, const unsigned int* indices, size_t index_count);

#ifdef __cplusplus
}
template <typename Output>


// 函数：clodBuild。构建派生数据结构，通常用于 LOD、层次结构、间接命令或加速访问。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：构建结果会被后续阶段高频读取，必须保证布局紧凑、索引合法并与共享结构定义一致。
size_t clodBuild(clodConfig config, clodMesh mesh, Output output)
{


	// 结构：Call。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
	// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
	// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
	struct Call
	{


		// 函数：output。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
		// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
		// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
		static int output(void* output_context, clodGroup group, const clodCluster* clusters, size_t cluster_count, size_t task_index, unsigned int thread_index)
		{
			return (*static_cast<Output*>(output_context))(group, clusters, cluster_count);
		}
	};

	return clodBuild(config, mesh, &output, &Call::output, nullptr);
}
#endif
