#pragma once
#include "meshlod_types.h"
#include <float.h>
#include <math.h>
#include <string.h>
#include <cassert>
#include <algorithm>
#include <vector>
#include <atomic>
#include <meshoptimizer.h>

namespace clod
{

struct Cluster
{
	size_t vertices;
	std::vector<unsigned int> indices;
	int group;
	int refined;
	clodBounds bounds;
};

struct SloppyVertex
{
	float x, y, z;
	unsigned int id;
};

struct IterationContext
{
	clodConfig config;
	clodMesh   mesh;
	clodOutput output_callback = nullptr;
	std::vector<unsigned char> locks;
	std::vector<unsigned int>  remap;

	int depth = 0;
	std::vector<Cluster> clusters;
	std::atomic<size_t>  next_cluster = {};
	std::vector<std::vector<int>> groups;

	std::vector<int>    pending;
	std::atomic<size_t> next_pending = {};
};

clodBounds boundsCompute(const clodMesh& mesh, const std::vector<unsigned int>& indices, float error);
clodBounds boundsMerge(const std::vector<Cluster>& clusters, const std::vector<int>& group);
std::vector<Cluster> clusterize(const clodConfig& config, const clodMesh& mesh, const unsigned int* indices, size_t index_count);
std::vector<std::vector<int> > partition(const clodConfig& config, const clodMesh& mesh, const std::vector<Cluster>& clusters, const std::vector<int>& pending, const std::vector<unsigned int>& remap);
void lockBoundary(std::vector<unsigned char>& locks, const std::vector<std::vector<int> >& groups, const std::vector<Cluster>& clusters, const std::vector<unsigned int>& remap, const unsigned char* vertex_lock);
float computeVertexCurvature(const float* positions, size_t stride, const unsigned int* indices, size_t index_count, unsigned int vertex, float radius);
float computeAverageEdgeLength(const clodMesh& mesh, const std::vector<unsigned int>& indices);
void computeFeatureWeights(const clodConfig& config,
                           const clodMesh& mesh,
                           const std::vector<unsigned int>& indices,
                           std::vector<float>& feature_weights,
                           std::vector<float>& curvature_values,
                           std::vector<unsigned char>& enhanced_locks);
float perceptualError(float geometric_error, float vertex_count, float original_count);
void simplifyFallback(std::vector<unsigned int>& lod, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, size_t target_count, float* error);
std::vector<unsigned int> simplify(const clodConfig& config, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, size_t target_count, float* error);
int outputGroup(const clodConfig& config, const clodMesh& mesh, const std::vector<Cluster>& clusters, const std::vector<int>& group, const clodBounds& simplified, int depth, void* output_context, clodOutput output_callback, size_t task_index, unsigned int thread_index);

}
