//==============================================================================
// 文件：src/meshlod/meshlod_types.h
// 模块定位：meshlod 公共数据结构定义，包括输入网格、构建配置、边界、输出簇、输出组和回调类型。
// 数据流：调用方填充 clodMesh 与 clodConfig；算法层通过 clodCluster/clodGroup 把构建结果返回给上层。
// 方法说明：这些类型构成算法层和应用层之间的最小公共语言，避免 meshlod 依赖 Vulkan 或 Scene 内部类型。
// 正确性约束：结构字段顺序和语义应保持稳定；stride、count 和指针必须由调用方保证合法。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#pragma once


// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <stddef.h>


// 结构：clodConfig。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct clodConfig
{
	size_t max_vertices;
	size_t min_triangles;
	size_t max_triangles;
	bool partition_spatial;
	bool partition_sort;
	size_t partition_size;
	bool cluster_spatial;
	float cluster_fill_weight;
	float cluster_split_factor;
	float simplify_ratio;
	float simplify_threshold;
	float simplify_error_merge_previous;
	float simplify_error_merge_additive;
	float simplify_error_factor_sloppy;
	float simplify_error_edge_limit;
	bool simplify_permissive;
	bool simplify_fallback_permissive;
	bool simplify_fallback_sloppy;
	bool simplify_regularize;
	bool optimize_bounds;
	bool optimize_clusters;
	float curvature_adaptive_strength;
	float curvature_window_radius;
	float feature_edge_threshold;
	float perceptual_weight;
	float silhouette_preservation;
};


// 结构：clodMesh。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct clodMesh
{
	const unsigned int* indices;
	size_t index_count;
	size_t vertex_count;
	const float* vertex_positions;
	size_t vertex_positions_stride;
	const float* vertex_attributes;
	size_t vertex_attributes_stride;
	const unsigned char* vertex_lock;
	const float* attribute_weights;
	size_t attribute_count;
	unsigned int attribute_protect_mask;
};


// 结构：clodBounds。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct clodBounds
{
	float center[3];
	float radius;
	float error;
};


// 结构：clodCluster。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct clodCluster
{
	int refined;
	clodBounds bounds;
	const unsigned int* indices;
	size_t index_count;
	size_t vertex_count;
};


// 结构：clodGroup。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct clodGroup
{
	int depth;
	clodBounds simplified;
};


// 函数：int。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
typedef int (*clodOutput)(void* output_context, clodGroup group, const clodCluster* clusters, size_t cluster_count, size_t task_index, unsigned int thread_index);


// 函数：void。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
typedef void (*clodIteration)(void* iteration_context, void* output_context, int depth, size_t task_count);
