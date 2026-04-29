#pragma once
#include <stddef.h>

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

struct clodBounds
{
	float center[3];
	float radius;
	float error;
};

struct clodCluster
{
	int refined;
	clodBounds bounds;
	const unsigned int* indices;
	size_t index_count;
	size_t vertex_count;
};

struct clodGroup
{
	int depth;
	clodBounds simplified;
};

typedef int (*clodOutput)(void* output_context, clodGroup group, const clodCluster* clusters, size_t cluster_count, size_t task_index, unsigned int thread_index);
typedef void (*clodIteration)(void* iteration_context, void* output_context, int depth, size_t task_count);
