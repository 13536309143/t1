#pragma once
#include "meshlod_impl.h"
#include "meshlod_bounds.h"
#include "meshlod_clustering.h"
#include "meshlod_simplify.h"

namespace clod
{

int outputGroup(const clodConfig& config, const clodMesh& mesh, const std::vector<Cluster>& clusters, const std::vector<int>& group, const clodBounds& simplified, int depth, void* output_context, clodOutput output_callback, size_t task_index, unsigned int thread_index)
{
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

	config.curvature_adaptive_strength = 0.3f;
	config.curvature_window_radius = 0.5f;
	config.feature_edge_threshold = 0.5f;
	config.perceptual_weight = 0.15f;
	config.silhouette_preservation = 0.2f;

	return config;
}

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

		context.clusters.resize(context.clusters.size() + context.pending.size() + context.groups.size());
		context.pending.resize(context.pending.size() + context.groups.size());
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
		/////////////////////////////////////////////////////////////////////////
        // 移除了最高级别LOD必须只有一个cluster的强制要求		
		// assert(context.pending.size() == 1);
		// const Cluster& cluster = context.clusters[context.pending[0]];
		// clodBounds bounds = cluster.bounds;
		/////////////////////////////////////////////////////////////////////////
		// 不再强制要求只有一个cluster，允许最高级别LOD有多个clusters
		clodBounds bounds = boundsMerge(context.clusters, context.pending);
		bounds.error = FLT_MAX;
		outputGroup(config, mesh, context.clusters, context.pending, bounds, context.depth, output_context, output_callback, 0, 0);
	}

	return context.clusters.size();
}
