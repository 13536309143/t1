#pragma once
#include "meshlod_impl.h"

namespace clod
{

struct FeatureEdge
{
	unsigned int a;
	unsigned int b;
	unsigned int v0;
	unsigned int v1;
	unsigned int face;
	float length;
};

static float clamp01(float v)
{
	return std::min(1.f, std::max(0.f, v));
}

static void positionSub(float r[3], const float* a, const float* b)
{
	r[0] = a[0] - b[0];
	r[1] = a[1] - b[1];
	r[2] = a[2] - b[2];
}

static float dot3(const float* a, const float* b)
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static float normalize3(float v[3])
{
	float len = sqrtf(dot3(v, v));
	if (len > 1e-20f)
	{
		v[0] /= len;
		v[1] /= len;
		v[2] /= len;
	}
	return len;
}

static float length3(const float* v)
{
	return sqrtf(dot3(v, v));
}

std::vector<float> computeFeatureImportance(const clodConfig& config, const clodMesh& mesh, const std::vector<unsigned int>& remap)
{
	std::vector<float> importance(mesh.vertex_count, 0.f);
	if (!mesh.indices || mesh.index_count < 3 || !mesh.vertex_positions)
		return importance;

	const size_t stride = mesh.vertex_positions_stride / sizeof(float);
	const size_t face_count = mesh.index_count / 3;
	const float sharp_threshold = config.feature_edge_threshold > 0.f ? config.feature_edge_threshold : 0.5f;
	const float cos_threshold = cosf(sharp_threshold);

	std::vector<float> face_normals(face_count * 3, 0.f);
	std::vector<float> mean_normals(mesh.vertex_count * 3, 0.f);
	std::vector<float> normal_variation(mesh.vertex_count, 0.f);
	std::vector<float> thin_wall(mesh.vertex_count, 0.f);
	std::vector<float> boundary(mesh.vertex_count, 0.f);
	std::vector<float> non_manifold(mesh.vertex_count, 0.f);
	std::vector<unsigned int> boundary_degree(mesh.vertex_count, 0);
	std::vector<float> local_min_edge(mesh.vertex_count, FLT_MAX);
	std::vector<float> local_max_edge(mesh.vertex_count, 0.f);
	std::vector<float> local_edge_sum(mesh.vertex_count, 0.f);
	std::vector<unsigned int> local_edge_count(mesh.vertex_count, 0);
	std::vector<FeatureEdge> edges;
	edges.reserve(mesh.index_count);

	float bbox_min[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
	float bbox_max[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
	for (size_t v = 0; v < mesh.vertex_count; ++v)
	{
		const float* p = &mesh.vertex_positions[v * stride];
		for (int c = 0; c < 3; ++c)
		{
			bbox_min[c] = std::min(bbox_min[c], p[c]);
			bbox_max[c] = std::max(bbox_max[c], p[c]);
		}
	}
	float bbox_diag_vec[3] = {bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1], bbox_max[2] - bbox_min[2]};
	float model_scale = std::max(length3(bbox_diag_vec), 1e-6f);

	auto updateLocalScale = [&](unsigned int v, float length) {
		if (v >= mesh.vertex_count || length <= 1e-12f)
			return;
		local_min_edge[v] = std::min(local_min_edge[v], length);
		local_max_edge[v] = std::max(local_max_edge[v], length);
		local_edge_sum[v] += length;
		local_edge_count[v]++;
	};

	for (size_t f = 0; f < face_count; ++f)
	{
		unsigned int tri[3] = {mesh.indices[f * 3 + 0], mesh.indices[f * 3 + 1], mesh.indices[f * 3 + 2]};
		if (tri[0] >= mesh.vertex_count || tri[1] >= mesh.vertex_count || tri[2] >= mesh.vertex_count)
			continue;

		const float* p0 = &mesh.vertex_positions[tri[0] * stride];
		const float* p1 = &mesh.vertex_positions[tri[1] * stride];
		const float* p2 = &mesh.vertex_positions[tri[2] * stride];

		float e0[3], e1[3];
		positionSub(e0, p1, p0);
		positionSub(e1, p2, p0);
		float e2[3];
		positionSub(e2, p0, p2);

		float edge_length[3] = {length3(e0), length3(&e1[0]), length3(e2)};
		float min_edge = std::max(1e-12f, std::min(edge_length[0], std::min(edge_length[1], edge_length[2])));
		float max_edge = std::max(edge_length[0], std::max(edge_length[1], edge_length[2]));
		float aspect_score = clamp01((max_edge / min_edge - 4.f) / 14.f);

		updateLocalScale(tri[0], edge_length[0]);
		updateLocalScale(tri[1], edge_length[0]);
		updateLocalScale(tri[1], edge_length[1]);
		updateLocalScale(tri[2], edge_length[1]);
		updateLocalScale(tri[2], edge_length[2]);
		updateLocalScale(tri[0], edge_length[2]);

		for (int i = 0; i < 3; ++i)
			thin_wall[tri[i]] = std::max(thin_wall[tri[i]], aspect_score);

		float n[3] = {
			e0[1] * e1[2] - e0[2] * e1[1],
			e0[2] * e1[0] - e0[0] * e1[2],
			e0[0] * e1[1] - e0[1] * e1[0],
		};

		float area2 = normalize3(n);
		if (area2 <= 1e-20f)
			continue;

		face_normals[f * 3 + 0] = n[0];
		face_normals[f * 3 + 1] = n[1];
		face_normals[f * 3 + 2] = n[2];

		for (int i = 0; i < 3; ++i)
		{
			unsigned int v = tri[i];
			mean_normals[v * 3 + 0] += n[0] * area2;
			mean_normals[v * 3 + 1] += n[1] * area2;
			mean_normals[v * 3 + 2] += n[2] * area2;
		}

		for (int e = 0; e < 3; ++e)
		{
			unsigned int v0 = tri[e];
			unsigned int v1 = tri[(e + 1) % 3];
			unsigned int r0 = remap.empty() ? v0 : remap[v0];
			unsigned int r1 = remap.empty() ? v1 : remap[v1];
			if (r0 == r1)
				continue;
			float ev[3];
			positionSub(ev, &mesh.vertex_positions[v0 * stride], &mesh.vertex_positions[v1 * stride]);
			edges.push_back({std::min(r0, r1), std::max(r0, r1), v0, v1, unsigned(f), length3(ev)});
		}
	}

	for (size_t v = 0; v < mesh.vertex_count; ++v)
		normalize3(&mean_normals[v * 3]);

	std::sort(edges.begin(), edges.end(), [](const FeatureEdge& lhs, const FeatureEdge& rhs) {
		return lhs.a == rhs.a ? lhs.b < rhs.b : lhs.a < rhs.a;
	});

	for (size_t begin = 0; begin < edges.size();)
	{
		size_t end = begin + 1;
		while (end < edges.size() && edges[end].a == edges[begin].a && edges[end].b == edges[begin].b)
			end++;

		float edge_score = 0.f;

		if (end - begin == 1)
		{
			edge_score = 1.f; // model boundary / hole loop
			boundary[edges[begin].v0] = 1.f;
			boundary[edges[begin].v1] = 1.f;
			boundary_degree[edges[begin].v0]++;
			boundary_degree[edges[begin].v1]++;
		}
		else if (end - begin > 2)
		{
			edge_score = 1.f; // non-manifold industrial CAD seam
			for (size_t i = begin; i < end; ++i)
			{
				non_manifold[edges[i].v0] = 1.f;
				non_manifold[edges[i].v1] = 1.f;
			}
		}
		else
		{
			const float* n0 = &face_normals[edges[begin + 0].face * 3];
			const float* n1 = &face_normals[edges[begin + 1].face * 3];
			float d = std::min(1.f, std::max(-1.f, dot3(n0, n1)));
			if (d < cos_threshold)
				edge_score = clamp01((cos_threshold - d) / std::max(cos_threshold + 1.f, 1e-5f));
		}

		if (edge_score > 0.f)
		{
			for (size_t i = begin; i < end; ++i)
			{
				importance[edges[i].v0] = std::max(importance[edges[i].v0], edge_score);
				importance[edges[i].v1] = std::max(importance[edges[i].v1], edge_score);
			}
		}

		begin = end;
	}

	for (size_t f = 0; f < face_count; ++f)
	{
		const float* fn = &face_normals[f * 3];
		if (dot3(fn, fn) == 0.f)
			continue;

		for (int i = 0; i < 3; ++i)
		{
			unsigned int v = mesh.indices[f * 3 + i];
			const float* mn = &mean_normals[v * 3];
			float curvature = clamp01((1.f - dot3(fn, mn)) * std::max(0.1f, config.curvature_window_radius));
			normal_variation[v] = std::max(normal_variation[v], curvature);
		}
	}

	for (size_t v = 0; v < mesh.vertex_count; ++v)
	{
		float avg_edge = local_edge_count[v] ? local_edge_sum[v] / float(local_edge_count[v]) : 0.f;
		float min_edge = local_min_edge[v] < FLT_MAX ? local_min_edge[v] : avg_edge;
		float slender_score = (avg_edge > 0.f && min_edge > 0.f) ? clamp01((avg_edge / min_edge - 2.f) / 6.f) : 0.f;
		float small_scale_score = avg_edge > 0.f ? clamp01((model_scale * 0.015f - avg_edge) / (model_scale * 0.015f)) : 0.f;
		float hole_loop_score = boundary_degree[v] >= 2 ? 1.f : boundary[v];

		thin_wall[v] = std::max(thin_wall[v], slender_score * 0.7f + small_scale_score * 0.3f);

		float weighted =
			boundary[v] * 0.28f +
			hole_loop_score * 0.18f +
			non_manifold[v] * 0.22f +
			importance[v] * 0.22f +
			normal_variation[v] * 0.18f +
			thin_wall[v] * 0.14f;

		importance[v] = std::max(importance[v], clamp01(weighted));
	}

	std::vector<float> propagated(importance);
	for (size_t i = 0; i < edges.size(); ++i)
	{
		float line_strength = std::max(importance[edges[i].v0], importance[edges[i].v1]);
		if (line_strength > 0.45f)
		{
			float attenuated = line_strength * 0.55f;
			propagated[edges[i].v0] = std::max(propagated[edges[i].v0], attenuated);
			propagated[edges[i].v1] = std::max(propagated[edges[i].v1], attenuated);
		}
	}
	importance.swap(propagated);

	if (mesh.vertex_attributes && mesh.attribute_protect_mask && mesh.vertex_attributes_stride)
	{
		size_t attribute_stride = mesh.vertex_attributes_stride / sizeof(float);
		for (size_t v = 0; v < mesh.vertex_count; ++v)
		{
			unsigned int r = remap.empty() ? unsigned(v) : remap[v];
			if (r == v || r >= mesh.vertex_count)
				continue;

			bool protected_seam = false;
			for (size_t a = 0; a < attribute_stride && a < 32; ++a)
			{
				if ((mesh.attribute_protect_mask & (1u << a)) && mesh.vertex_attributes[v * attribute_stride + a] != mesh.vertex_attributes[r * attribute_stride + a])
				{
					protected_seam = true;
					break;
				}
			}

			if (protected_seam)
				importance[v] = std::max(importance[v], 0.85f);
		}
	}

	if (!remap.empty())
	{
		std::vector<float> representative(mesh.vertex_count, 0.f);
		for (size_t v = 0; v < mesh.vertex_count; ++v)
			representative[remap[v]] = std::max(representative[remap[v]], importance[v]);
		for (size_t v = 0; v < mesh.vertex_count; ++v)
			importance[v] = std::max(importance[v], representative[remap[v]]);
	}

	return importance;
}

static void featureStats(const std::vector<unsigned int>& indices, const std::vector<float>& feature_importance, float& avg, float& max_feature)
{
	avg = 0.f;
	max_feature = 0.f;
	float strong_count = 0.f;

	if (feature_importance.empty() || indices.empty())
		return;

	for (size_t i = 0; i < indices.size(); ++i)
	{
		unsigned int v = indices[i];
		if (v < feature_importance.size())
		{
			float feature = feature_importance[v];
			avg += feature;
			max_feature = std::max(max_feature, feature);
			if (feature > 0.65f)
				strong_count += 1.f;
		}
	}

	avg /= float(indices.size());
	float strong_ratio = strong_count / float(indices.size());
	avg = clamp01(avg * 0.65f + strong_ratio * 0.35f);
}

static size_t featureAdaptiveTarget(const clodConfig& config, const std::vector<unsigned int>& indices, const std::vector<float>& feature_importance, size_t target_count)
{
	float avg_feature = 0.f;
	float max_feature = 0.f;
	featureStats(indices, feature_importance, avg_feature, max_feature);

	float pressure = clamp01(avg_feature * 0.7f + max_feature * 0.3f);
	float strength = clamp01(config.curvature_adaptive_strength + config.silhouette_preservation);
	float preserve = 1.f - config.simplify_ratio;
	size_t relaxed = target_count + size_t(float(indices.size() - target_count) * preserve * strength * pressure);
	relaxed = (relaxed / 3) * 3;

	return std::max(target_count, std::min(indices.size(), relaxed));
}

static void applyFeatureLocks(const clodConfig& config, const std::vector<unsigned int>& indices, const std::vector<float>& feature_importance, std::vector<unsigned char>& locks)
{
	if (feature_importance.empty() || config.silhouette_preservation <= 0.f)
		return;

	float lock_threshold = 1.f - 0.2f * clamp01(config.silhouette_preservation);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		unsigned int v = indices[i];
		if (v < feature_importance.size() && feature_importance[v] >= lock_threshold)
			locks[v] |= meshopt_SimplifyVertex_Protect;
	}
}

float perceptualError(float geometric_error, float vertex_count, float original_count)
{
	return geometric_error * powf(vertex_count / (original_count > 0 ? original_count : 1), 0.3f);
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

std::vector<unsigned int> simplify(const clodConfig& config, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, const std::vector<float>& feature_importance, size_t target_count, float* error)
{
	if (target_count > indices.size())
		return indices;

	size_t positions_stride = mesh.vertex_positions_stride / sizeof(float);
	size_t adaptive_target = featureAdaptiveTarget(config, indices, feature_importance, target_count);

	float avg_feature = 0.f;
	float max_feature = 0.f;
	featureStats(indices, feature_importance, avg_feature, max_feature);

	std::vector<unsigned char> enhanced_locks(locks);
	applyFeatureLocks(config, indices, feature_importance, enhanced_locks);

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

	if (lod.size() > adaptive_target && config.simplify_fallback_sloppy && avg_feature < 0.35f)
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

	float feature_error_scale = 1.f + clamp01(config.curvature_adaptive_strength + config.silhouette_preservation) * (avg_feature * 0.75f + max_feature * 0.25f);
	if (error)
		*error *= feature_error_scale;

	return lod;
}

}
