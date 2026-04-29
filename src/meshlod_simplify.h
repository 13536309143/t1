#pragma once
#include "meshlod_impl.h"

namespace clod
{

float computeAverageEdgeLength(const clodMesh& mesh, const std::vector<unsigned int>& indices)
{
	size_t positions_stride = mesh.vertex_positions_stride / sizeof(float);
	double edge_sum = 0.0;
	size_t edge_count = 0;

	for (size_t i = 0; i + 2 < indices.size(); i += 3)
	{
		unsigned int tri[3] = {indices[i], indices[i + 1], indices[i + 2]};
		for (int e = 0; e < 3; ++e)
		{
			const float* a = &mesh.vertex_positions[tri[e] * positions_stride];
			const float* b = &mesh.vertex_positions[tri[(e + 1) % 3] * positions_stride];
			const float dx = a[0] - b[0];
			const float dy = a[1] - b[1];
			const float dz = a[2] - b[2];
			edge_sum += sqrt(double(dx * dx + dy * dy + dz * dz));
			edge_count++;
		}
	}

	return edge_count ? float(edge_sum / double(edge_count)) : 0.0f;
}

float computeVertexCurvature(const float* positions, size_t stride, const unsigned int* indices, size_t index_count, unsigned int vertex, float radius)
{
	float pos[3] = {positions[vertex * stride + 0], positions[vertex * stride + 1], positions[vertex * stride + 2]};
	float mean_normal[3] = {0.f, 0.f, 0.f};
	int count = 0;
	float r2 = radius * radius;

	for (size_t i = 0; i < index_count; i += 3)
	{
		unsigned int v0 = indices[i], v1 = indices[i + 1], v2 = indices[i + 2];
		if (v0 != vertex && v1 != vertex && v2 != vertex)
			continue;
		float p[3][3];
		for (int j = 0; j < 3; ++j)
		{
			unsigned int vi = (j == 0) ? v0 : (j == 1) ? v1 : v2;
			p[j][0] = positions[vi * stride + 0] - pos[0];
			p[j][1] = positions[vi * stride + 1] - pos[1];
			p[j][2] = positions[vi * stride + 2] - pos[2];
		}
		float edge1[3] = {p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]};
		float edge2[3] = {p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]};
		float normal[3] = {edge1[1] * edge2[2] - edge1[2] * edge2[1],
			edge1[2] * edge2[0] - edge1[0] * edge2[2],
			edge1[0] * edge2[1] - edge1[1] * edge2[0]};
		float len = sqrtf(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
		if (len > 1e-8f)
		{
			mean_normal[0] += normal[0] / len;
			mean_normal[1] += normal[1] / len;
			mean_normal[2] += normal[2] / len;
			++count;
		}
	}
	if (count == 0)
		return 0.f;
	float nlen = sqrtf(mean_normal[0] * mean_normal[0] + mean_normal[1] * mean_normal[1] + mean_normal[2] * mean_normal[2]);
	if (nlen < 1e-8f)
		return 0.f;
	mean_normal[0] /= nlen;
	mean_normal[1] /= nlen;
	mean_normal[2] /= nlen;

	float variance = 0.f;
	for (size_t i = 0; i < index_count; i += 3)
	{
		unsigned int v0 = indices[i], v1 = indices[i + 1], v2 = indices[i + 2];
		if (v0 != vertex && v1 != vertex && v2 != vertex)
			continue;
		for (int j = 0; j < 3; ++j)
		{
			unsigned int vi = (j == 0) ? v0 : (j == 1) ? v1 : v2;
			float vp[3] = {positions[vi * stride + 0] - pos[0], positions[vi * stride + 1] - pos[1], positions[vi * stride + 2] - pos[2]};
			float dist2 = vp[0] * vp[0] + vp[1] * vp[1] + vp[2] * vp[2];
			if (dist2 <= r2)
			{
				float proj = vp[0] * mean_normal[0] + vp[1] * mean_normal[1] + vp[2] * mean_normal[2];
				variance += (dist2 - proj * proj);
			}
		}
	}
	return variance / float(count > 0 ? count : 1);
}

void computeFeatureWeights(const clodConfig& config,
                           const clodMesh& mesh,
                           const std::vector<unsigned int>& indices,
                           std::vector<float>& feature_weights,
                           std::vector<float>& curvature_values,
                           std::vector<unsigned char>& enhanced_locks)
{
	size_t positions_stride = mesh.vertex_positions_stride / sizeof(float);
	float radius = config.curvature_window_radius > 0 ? config.curvature_window_radius : 1.0f;
	float average_edge_length = computeAverageEdgeLength(mesh, indices);
	float edge_thresh = std::max(average_edge_length * std::max(config.feature_edge_threshold, 0.05f), 1e-5f);
	float edge_bias = std::max(0.25f, 0.5f + config.curvature_adaptive_strength);

	for (size_t i = 0; i < mesh.vertex_count; ++i)
	{
		feature_weights[i] = 1.0f;
		curvature_values[i] = 0.0f;
	}

	for (size_t i = 0; i < indices.size(); i += 3)
	{
		unsigned int a = indices[i], b = indices[i + 1], c = indices[i + 2];
		float pa[3] = {mesh.vertex_positions[a * positions_stride + 0], mesh.vertex_positions[a * positions_stride + 1], mesh.vertex_positions[a * positions_stride + 2]};
		float pb[3] = {mesh.vertex_positions[b * positions_stride + 0], mesh.vertex_positions[b * positions_stride + 1], mesh.vertex_positions[b * positions_stride + 2]};
		float pc[3] = {mesh.vertex_positions[c * positions_stride + 0], mesh.vertex_positions[c * positions_stride + 1], mesh.vertex_positions[c * positions_stride + 2]};

		float eab = sqrtf((pa[0] - pb[0]) * (pa[0] - pb[0]) + (pa[1] - pb[1]) * (pa[1] - pb[1]) + (pa[2] - pb[2]) * (pa[2] - pb[2]));
		float eac = sqrtf((pa[0] - pc[0]) * (pa[0] - pc[0]) + (pa[1] - pc[1]) * (pa[1] - pc[1]) + (pa[2] - pc[2]) * (pa[2] - pc[2]));
		float ebc = sqrtf((pb[0] - pc[0]) * (pb[0] - pc[0]) + (pb[1] - pc[1]) * (pb[1] - pc[1]) + (pb[2] - pc[2]) * (pb[2] - pc[2]));

		if (eab > edge_thresh)
		{
			float edge_importance = std::min((eab / edge_thresh) - 1.0f, 4.0f);
			feature_weights[a] += edge_importance * edge_bias;
			feature_weights[b] += edge_importance * edge_bias;
		}
		if (eac > edge_thresh)
		{
			float edge_importance = std::min((eac / edge_thresh) - 1.0f, 4.0f);
			feature_weights[a] += edge_importance * edge_bias;
			feature_weights[c] += edge_importance * edge_bias;
		}
		if (ebc > edge_thresh)
		{
			float edge_importance = std::min((ebc / edge_thresh) - 1.0f, 4.0f);
			feature_weights[b] += edge_importance * edge_bias;
			feature_weights[c] += edge_importance * edge_bias;
		}
	}

	float max_curvature = 0.0f;
	for (size_t i = 0; i < mesh.vertex_count; ++i)
	{
		float curvature = computeVertexCurvature(mesh.vertex_positions, positions_stride, indices.data(), indices.size(), (unsigned int)i, radius);
		curvature_values[i] = curvature;
		max_curvature = std::max(max_curvature, curvature);
	}

	float max_feature_weight = 1.0f;
	for (size_t i = 0; i < mesh.vertex_count; ++i)
	{
		float normalized_curvature = max_curvature > 1e-8f ? curvature_values[i] / max_curvature : 0.0f;
		feature_weights[i] += normalized_curvature * (1.0f + config.curvature_adaptive_strength * 2.0f);
		max_feature_weight = std::max(max_feature_weight, feature_weights[i]);
	}

	float feature_span = std::max(max_feature_weight - 1.0f, 0.0f);
	float protect_threshold = std::max(0.2f, 0.7f - config.curvature_adaptive_strength * 0.4f);
	for (size_t i = 0; i < mesh.vertex_count; ++i)
	{
		float normalized_feature = feature_span > 1e-6f ? (feature_weights[i] - 1.0f) / feature_span : 0.0f;
		if (normalized_feature >= protect_threshold)
		{
			enhanced_locks[i] |= meshopt_SimplifyVertex_Protect;
		}
	}
}

float perceptualError(float geometric_error, float vertex_count, float original_count)
{
	float reduced_vertices = std::max(vertex_count, 1.0f);
	float source_vertices = std::max(original_count, 1.0f);
	return geometric_error * powf(source_vertices / reduced_vertices, 0.3f);
}

void simplifyFallback(std::vector<unsigned int>& lod, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, size_t target_count, float* error)
{
	std::vector<SloppyVertex> subset(indices.size());
	std::vector<unsigned char> subset_locks(indices.size());
	lod.resize(indices.size());
	size_t positions_stride = mesh.vertex_positions_stride / sizeof(float);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		unsigned int v = indices[i];
		subset[i].x = mesh.vertex_positions[v * positions_stride + 0];
		subset[i].y = mesh.vertex_positions[v * positions_stride + 1];
		subset[i].z = mesh.vertex_positions[v * positions_stride + 2];
		subset[i].id = v;

		subset_locks[i] = locks[v];
		lod[i] = unsigned(i);
	}

	lod.resize(meshopt_simplifySloppy(&lod[0], &lod[0], lod.size(), &subset[0].x, subset.size(), sizeof(SloppyVertex), subset_locks.data(), target_count, FLT_MAX, error));
	*error *= meshopt_simplifyScale(&subset[0].x, subset.size(), sizeof(SloppyVertex));

	for (size_t i = 0; i < lod.size(); ++i)
		lod[i] = subset[lod[i]].id;
}

std::vector<unsigned int> simplify(const clodConfig& config, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, size_t target_count, float* error)
{
	if (target_count > indices.size())
		return indices;

	size_t original_count = indices.size();
	size_t positions_stride = mesh.vertex_positions_stride / sizeof(float);

	std::vector<float> feature_weights(mesh.vertex_count, 1.0f);
	std::vector<float> curvature_values(mesh.vertex_count, 0.0f);
	std::vector<unsigned char> enhanced_locks(locks);
	if (config.curvature_adaptive_strength > 0 || config.feature_edge_threshold > 0)
	{
		computeFeatureWeights(config, mesh, indices, feature_weights, curvature_values, enhanced_locks);
	}

	float adaptive_ratio = config.simplify_ratio;
	if (config.curvature_adaptive_strength > 0 || config.feature_edge_threshold > 0 || config.perceptual_weight > 0)
	{
		std::vector<char> vertex_used(mesh.vertex_count, 0);
		for (size_t i = 0; i < indices.size(); ++i)
		{
			vertex_used[indices[i]] = 1;
		}

		float avg_feature_importance = 0.0f;
		float max_feature_importance = 0.0f;
		float avg_normalized_curvature = 0.0f;
		float max_curvature = 0.0f;
		float used_count = 0.0f;

		for (size_t i = 0; i < mesh.vertex_count; ++i)
		{
			if (!vertex_used[i])
			{
				continue;
			}

			float feature_importance = std::max(feature_weights[i] - 1.0f, 0.0f);
			avg_feature_importance += feature_importance;
			max_feature_importance = std::max(max_feature_importance, feature_importance);
			max_curvature = std::max(max_curvature, curvature_values[i]);
			used_count += 1.0f;
		}

		if (used_count > 0.0f)
		{
			avg_feature_importance /= used_count;
			for (size_t i = 0; i < mesh.vertex_count; ++i)
			{
				if (!vertex_used[i])
				{
					continue;
				}

				float normalized_curvature = max_curvature > 1e-8f ? curvature_values[i] / max_curvature : 0.0f;
				avg_normalized_curvature += normalized_curvature;
			}
			avg_normalized_curvature /= used_count;

			float feature_boost = std::min(0.35f, avg_feature_importance * (0.08f + config.curvature_adaptive_strength * 0.20f));
			float curvature_boost = avg_normalized_curvature * config.curvature_adaptive_strength * 0.25f;
			float perceptual_boost = (1.0f - config.simplify_ratio) * config.perceptual_weight * 0.35f;

			if (max_feature_importance > 0.0f)
			{
				feature_boost += std::min(0.15f, max_feature_importance * 0.03f);
			}

			adaptive_ratio = std::min(0.98f, config.simplify_ratio + feature_boost + curvature_boost + perceptual_boost);
		}
	}

	size_t adaptive_target = size_t((indices.size() / 3) * adaptive_ratio) * 3;
	adaptive_target = std::max(adaptive_target, target_count);
	adaptive_target = std::min(adaptive_target, indices.size());

	std::vector<unsigned int> lod(indices.size());
	unsigned int options = meshopt_SimplifySparse | meshopt_SimplifyErrorAbsolute | (config.simplify_permissive ? meshopt_SimplifyPermissive : 0) | (config.simplify_regularize ? meshopt_SimplifyRegularize : 0);

	lod.resize(meshopt_simplifyWithAttributes(&lod[0], &indices[0], indices.size(),
		mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
		mesh.vertex_attributes, mesh.vertex_attributes_stride, mesh.attribute_weights, mesh.attribute_count,
		&enhanced_locks[0], adaptive_target, FLT_MAX, options, error));

	if (lod.size() > adaptive_target && config.simplify_fallback_permissive && !config.simplify_permissive)
	{
		lod.resize(meshopt_simplifyWithAttributes(&lod[0], &indices[0], indices.size(),
			mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
			mesh.vertex_attributes, mesh.vertex_attributes_stride, mesh.attribute_weights, mesh.attribute_count,
			&enhanced_locks[0], adaptive_target, FLT_MAX, options | meshopt_SimplifyPermissive, error));
	}

	if (lod.size() > adaptive_target && config.simplify_fallback_sloppy)
	{
		simplifyFallback(lod, mesh, indices, enhanced_locks, adaptive_target, error);
		*error *= config.simplify_error_factor_sloppy;
	}

	if (config.simplify_error_edge_limit > 0)
	{
		float max_edge_sq = 0;
		for (size_t i = 0; i < indices.size(); i += 3)
		{
			unsigned int a = indices[i + 0], b = indices[i + 1], c = indices[i + 2];

			const float* va = &mesh.vertex_positions[a * positions_stride];
			const float* vb = &mesh.vertex_positions[b * positions_stride];
			const float* vc = &mesh.vertex_positions[c * positions_stride];
			float eab = (va[0] - vb[0]) * (va[0] - vb[0]) + (va[1] - vb[1]) * (va[1] - vb[1]) + (va[2] - vb[2]) * (va[2] - vb[2]);
			float eac = (va[0] - vc[0]) * (va[0] - vc[0]) + (va[1] - vc[1]) * (va[1] - vc[1]) + (va[2] - vc[2]) * (va[2] - vc[2]);
			float ebc = (vb[0] - vc[0]) * (vb[0] - vc[0]) + (vb[1] - vc[1]) * (vb[1] - vc[1]) + (vb[2] - vc[2]) * (vb[2] - vc[2]);

			float emax = std::max(std::max(eab, eac), ebc);
			float emin = std::min(std::min(eab, eac), ebc);
			max_edge_sq = std::max(max_edge_sq, std::max(emin, emax / 4));
		}
		*error = std::min(*error, sqrtf(max_edge_sq) * config.simplify_error_edge_limit);
	}

	if (config.perceptual_weight > 0 && error != nullptr)
	{
		float unique_vertices = 0.f;
		std::vector<char> vertex_used(mesh.vertex_count, 0);
		for (size_t i = 0; i < lod.size(); ++i)
		{
			if (!vertex_used[lod[i]])
			{
				vertex_used[lod[i]] = 1;
				unique_vertices += 1.f;
			}
		}
		*error = perceptualError(*error, unique_vertices, (float)mesh.vertex_count) * config.perceptual_weight + *error * (1.f - config.perceptual_weight);
	}

	return lod;
}

}
