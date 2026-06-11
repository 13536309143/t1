//==============================================================================
// 文件：src/meshlod/meshlod_build.h
// 模块定位：LOD 构建主算法，管理多轮网格简化、簇 化、组 输出和并行迭代任务。
// 数据流：输入 clodMesh、clodConfig 与输出回调；输出多级 簇/组 以及可由上层构建层次树的中间结果。
// 方法说明：算法通过逐级降低几何复杂度形成层次细节表达，使运行时可依据屏幕误差选择适当精度。
// 正确性约束：每轮简化必须保持索引有效；输出顺序需可被 Scene 映射到 LOD level；并行任务不得共享未同步可变状态。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#pragma once


// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include "meshlod_impl.h"
#include "meshlod_bounds.h"
#include "meshlod_clustering.h"
#include "meshlod_simplify.h"

namespace clod
{

size_t estimateSplitClusterCapacity(const clodConfig& config, const std::vector<Cluster>& clusters, const std::vector<std::vector<int>>& groups)
{
	size_t capacity = 0;
	const size_t minTriangles = std::max<size_t>(1, config.min_triangles);

	for (const std::vector<int>& group : groups)
	{
		size_t groupTriangles = 0;

		for (int clusterIndex : group)
		{
			assert(clusterIndex >= 0);
			groupTriangles += clusters[size_t(clusterIndex)].indices.size() / 3;
		}

		const size_t estimatedSplitCount = std::max<size_t>(1, (groupTriangles + minTriangles - 1) / minTriangles);
		capacity += estimatedSplitCount;
	}

	return capacity;
}


// 函数：outputGroup。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
int outputGroup(const clodConfig& config, const clodMesh& mesh, const std::vector<Cluster>& clusters, const std::vector<int>& group, const clodBounds& simplified, int depth, void* output_context, clodOutput output_callback, size_t task_index, unsigned int thread_index)
{


	// 函数：group_clusters。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<clodCluster> group_clusters(group.size());

	for (size_t i = 0; i < group.size(); ++i)
	{
		const Cluster& cluster = clusters[group[i]];
		clodCluster& result = group_clusters[i];
		result.refined = cluster.refined;
		result.bounds = (config.optimize_bounds && cluster.refined != -1) ? boundsCompute(mesh, cluster.indices, cluster.bounds.error) : cluster.bounds;

		result.indices = cluster.indices.data();

		result.index_count = cluster.indices.size();
		result.vertex_count = cluster.vertices;
	}

	return output_callback ? output_callback(output_context, {depth, simplified}, group_clusters.data(), group_clusters.size(), task_index, thread_index) : -1;
}

}


// 函数：clodDefaultConfig。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
clodConfig clodDefaultConfig(size_t max_triangles)
{

	assert(max_triangles >= 4 && max_triangles <= 256);
	clodConfig config = {};
	config.max_vertices = max_triangles;
	config.min_triangles = max_triangles / 3;
	config.max_triangles = max_triangles;
#if MESHOPTIMIZER_VERSION < 1000
	config.min_triangles &= ~3;
#endif
	config.partition_spatial = true;
	config.partition_size = 16;

	config.cluster_spatial = false;
	config.cluster_split_factor = 2.0f;

	config.optimize_clusters = true;
	config.simplify_ratio = 0.5f;
	config.simplify_threshold = 0.85f;

	config.simplify_error_merge_previous = 1.0f;
	config.simplify_error_factor_sloppy = 2.0f;
	config.simplify_permissive = true;
	config.simplify_fallback_permissive = true;
	config.simplify_fallback_sloppy = true;
	config.learned_importance_enable = false;
	config.learned_importance_strength = 1.0f;
	config.learned_importance_protect_threshold = 0.92f;
	config.learned_importance_target_boost = 0.35f;
	config.learned_importance_error_scale = 1.25f;
	config.learned_importance_topology_edge_limit = 3000000u;

	return config;
}


// 函数：clodBuild_iterationTask。构建派生数据结构，通常用于 LOD、层次结构、间接命令或加速访问。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：构建结果会被后续阶段高频读取，必须保证布局紧凑、索引合法并与共享结构定义一致。
void clodBuild_iterationTask(void* iteration_context, void* output_context, size_t i, unsigned int thread_index)
{
	using namespace clod;

	IterationContext& context = *(IterationContext*)iteration_context;
	const std::vector<std::vector<int>>& groups = context.groups;
	std::vector<Cluster>& clusters = context.clusters;
	const std::vector<unsigned char>& locks = context.locks;
	const clodMesh& mesh = context.mesh;
	const clodConfig& config = context.config;
	int                                  depth = context.depth;

	std::vector<unsigned int> merged;
	merged.reserve(groups[i].size() * config.max_triangles * 3);

	for (size_t j = 0; j < groups[i].size(); ++j)
		merged.insert(merged.end(), clusters[groups[i][j]].indices.begin(), clusters[groups[i][j]].indices.end());

	size_t target_size = size_t((merged.size() / 3) * config.simplify_ratio) * 3;


	clodBounds bounds = boundsMerge(clusters, groups[i]);

	float error = 0.f;


	std::vector<unsigned int> simplified = simplify(config, mesh, merged, locks, target_size, &error);

	if (simplified.size() > merged.size() * config.simplify_threshold)
	{
		bounds.error = FLT_MAX;
		outputGroup(config, mesh, clusters, groups[i], bounds, depth, output_context, context.output_callback, i, thread_index);
		return;
	}

	bounds.error = std::max(bounds.error * config.simplify_error_merge_previous, error) + error * config.simplify_error_merge_additive;


	int refined = outputGroup(config, mesh, clusters, groups[i], bounds, depth, output_context, context.output_callback, i, thread_index);

	for (size_t j = 0; j < groups[i].size(); ++j)
		clusters[groups[i][j]].indices = std::vector<unsigned int>();

	std::vector<Cluster> split = clusterize(config, mesh, simplified.data(), simplified.size());

	size_t cluster_index = context.next_cluster.fetch_add(split.size());
	size_t pending_index = context.next_pending.fetch_add(split.size());

	for (Cluster& cluster : split)
	{
		cluster.refined = refined;

		cluster.bounds = bounds;

		assert(pending_index < context.pending.size());
		assert(cluster_index < context.clusters.size());


		context.pending[pending_index++] = int(cluster_index);

		context.clusters[cluster_index++] = std::move(cluster);
	}
}


// 函数：clodBuild。构建派生数据结构，通常用于 LOD、层次结构、间接命令或加速访问。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：构建结果会被后续阶段高频读取，必须保证布局紧凑、索引合法并与共享结构定义一致。
size_t clodBuild(clodConfig config, clodMesh mesh, void* output_context, clodOutput output_callback, clodIteration iteration_callback)
{
	using namespace clod;

	assert(mesh.vertex_attributes_stride % sizeof(float) == 0);
	assert(mesh.attribute_count * sizeof(float) <= mesh.vertex_attributes_stride);
	assert(mesh.attribute_protect_mask < (1u << (mesh.vertex_attributes_stride / sizeof(float))));

	IterationContext context;
	context.config = config;
	context.mesh = mesh;
	context.output_callback = output_callback;

	context.locks.resize(mesh.vertex_count);

	context.remap.resize(mesh.vertex_count);


	meshopt_generatePositionRemap(&context.remap[0], mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride);
	if (config.learned_importance_enable)
	{
		context.learned_importance = computeLearnedImportance(config, mesh);

		for (size_t i = 0; i < mesh.vertex_count; ++i)
		{
			unsigned int r = context.remap[i];
			if (r < context.learned_importance.size() && i < context.learned_importance.size())
				context.learned_importance[r] = std::max(context.learned_importance[r], context.learned_importance[i]);
		}

		for (size_t i = 0; i < mesh.vertex_count; ++i)
		{
			unsigned int r = context.remap[i];
			if (r < context.learned_importance.size())
				context.learned_importance[i] = context.learned_importance[r];
		}

		context.mesh.learned_importance = context.learned_importance.data();
	}

	if (mesh.attribute_protect_mask)
	{
		size_t max_attributes = mesh.vertex_attributes_stride / sizeof(float);

		for (size_t i = 0; i < mesh.vertex_count; ++i)
		{
			unsigned int r = context.remap[i];
			for (size_t j = 0; j < max_attributes; ++j)
				if (r != i && (mesh.attribute_protect_mask & (1u << j)) && mesh.vertex_attributes[i * max_attributes + j] != mesh.vertex_attributes[r * max_attributes + j])
					context.locks[i] |= meshopt_SimplifyVertex_Protect;
		}
	}


	context.clusters = clusterize(config, mesh, mesh.indices, mesh.index_count);

	context.next_cluster = context.clusters.size();

	for (Cluster& cluster : context.clusters)

		cluster.bounds = boundsCompute(mesh, cluster.indices, 0.f);

	context.pending.resize(context.clusters.size());
	for (size_t i = 0; i < context.clusters.size(); ++i)

		context.pending[i] = int(i);

	while (context.pending.size() > 1)
	{

		context.groups = partition(config, mesh, context.clusters, context.pending, context.remap);

		const size_t splitCapacity = estimateSplitClusterCapacity(config, context.clusters, context.groups);

		context.clusters.resize(context.clusters.size() + splitCapacity);
		context.pending.resize(splitCapacity);
		context.next_pending = 0;


		lockBoundary(context.locks, context.groups, context.clusters, context.remap, context.mesh.vertex_lock);

		if (iteration_callback)
		{
			iteration_callback(&context, output_context, context.depth, context.groups.size());
		}
		else
		{
			for (size_t i = 0; i < context.groups.size(); ++i)
			{

				clodBuild_iterationTask(&context, output_context, i, 0);
			}
		}


		context.pending.resize(context.next_pending);

		context.clusters.resize(context.next_cluster);

		context.depth++;
	}

	if (context.pending.size())
	{


		clodBounds bounds = boundsMerge(context.clusters, context.pending);
		bounds.error = FLT_MAX;

		outputGroup(config, mesh, context.clusters, context.pending, bounds, context.depth, output_context, output_callback, 0, 0);
	}

	return context.clusters.size();
}
