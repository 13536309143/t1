#pragma once
#include "meshlod_impl.h"

namespace clod
{

namespace detail
{

struct FaceInfo
{
	float normal[3] = {};
	float min_edge = 0.f;
	float max_edge = 0.f;
};

struct EdgeUse
{
	unsigned int a = 0;
	unsigned int b = 0;
	unsigned int face = 0;
};

inline float clamp01(float v)
{
	return std::max(0.f, std::min(1.f, v));
}

inline float dot3(const float* a, const float* b)
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline float length3(const float* v)
{
	return sqrtf(dot3(v, v));
}

inline float distance3(const float* a, const float* b)
{
	float d[3] = {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
	return length3(d);
}

}  // namespace detail

IndustrialFeatureStats computeIndustrialFeatureStats(const clodConfig& config,
                                                     const clodMesh&   mesh,
                                                     const std::vector<unsigned int>& indices)
{
	IndustrialFeatureStats stats = {};
	if(!config.industrial_feature_preservation || indices.size() < 3)
		return stats;

	const size_t positions_stride = mesh.vertex_positions_stride / sizeof(float);
	const size_t triangle_count   = indices.size() / 3;

	std::vector<detail::FaceInfo> faces(triangle_count);
	std::vector<detail::EdgeUse>  edges;
	edges.reserve(triangle_count * 3);

	std::vector<unsigned int> unique_vertices = indices;
	std::sort(unique_vertices.begin(), unique_vertices.end());
	unique_vertices.erase(std::unique(unique_vertices.begin(), unique_vertices.end()), unique_vertices.end());
	stats.unique_vertices = unique_vertices.size();

	float normal_sum[3] = {};
	size_t valid_faces = 0;

	for(size_t t = 0; t < triangle_count; ++t)
	{
		const unsigned int ia = indices[t * 3 + 0];
		const unsigned int ib = indices[t * 3 + 1];
		const unsigned int ic = indices[t * 3 + 2];

		const float* a = &mesh.vertex_positions[ia * positions_stride];
		const float* b = &mesh.vertex_positions[ib * positions_stride];
		const float* c = &mesh.vertex_positions[ic * positions_stride];

		float ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
		float ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
		float n[3]  = {ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2],
                     ab[0] * ac[1] - ab[1] * ac[0]};
		float nlen = detail::length3(n);
		if(nlen > 1e-12f)
		{
			faces[t].normal[0] = n[0] / nlen;
			faces[t].normal[1] = n[1] / nlen;
			faces[t].normal[2] = n[2] / nlen;
			normal_sum[0] += faces[t].normal[0];
			normal_sum[1] += faces[t].normal[1];
			normal_sum[2] += faces[t].normal[2];
			valid_faces++;
		}

		const float eab = detail::distance3(a, b);
		const float ebc = detail::distance3(b, c);
		const float eca = detail::distance3(c, a);
		faces[t].min_edge = std::min(eab, std::min(ebc, eca));
		faces[t].max_edge = std::max(eab, std::max(ebc, eca));
		if(faces[t].max_edge > 1e-12f && faces[t].min_edge / faces[t].max_edge < 0.08f)
			stats.thin += 1.f;

		unsigned int tri[3] = {ia, ib, ic};
		for(int e = 0; e < 3; ++e)
		{
			unsigned int x = tri[e];
			unsigned int y = tri[(e + 1) % 3];
			if(x > y)
				std::swap(x, y);
			edges.push_back({x, y, unsigned(t)});
		}
	}

	if(valid_faces)
	{
		const float mean_len = detail::length3(normal_sum) / float(valid_faces);
		stats.normal_variation = detail::clamp01(1.f - mean_len);
	}
	stats.thin = triangle_count ? detail::clamp01(stats.thin / float(triangle_count)) : 0.f;

	std::sort(edges.begin(), edges.end(), [](const detail::EdgeUse& lhs, const detail::EdgeUse& rhs) {
		return lhs.a == rhs.a ? lhs.b < rhs.b : lhs.a < rhs.a;
	});

	std::vector<unsigned int> feature_vertices;
	feature_vertices.reserve(edges.size());

	const float sharp_dot_threshold = detail::clamp01(config.feature_edge_threshold > 0.f ? config.feature_edge_threshold : 0.5f);
	size_t edge_count = 0;
	for(size_t i = 0; i < edges.size();)
	{
		size_t j = i + 1;
		while(j < edges.size() && edges[j].a == edges[i].a && edges[j].b == edges[i].b)
			j++;

		const size_t uses = j - i;
		edge_count++;

		bool structural = false;
		if(uses != 2)
		{
			stats.boundary += 1.f;
			structural = true;
		}
		else
		{
			const detail::FaceInfo& f0 = faces[edges[i].face];
			const detail::FaceInfo& f1 = faces[edges[i + 1].face];
			const float normal_dot = fabsf(detail::dot3(f0.normal, f1.normal));
			if(normal_dot < sharp_dot_threshold)
			{
				stats.sharp += 1.f;
				structural = true;
			}
		}

		if(structural)
		{
			feature_vertices.push_back(edges[i].a);
			feature_vertices.push_back(edges[i].b);
		}

		i = j;
	}

	if(edge_count)
	{
		stats.boundary = detail::clamp01(stats.boundary / float(edge_count));
		stats.sharp    = detail::clamp01(stats.sharp / float(edge_count));
	}

	float feature_vertex_ratio = 0.f;
	if(!feature_vertices.empty() && stats.unique_vertices)
	{
		std::sort(feature_vertices.begin(), feature_vertices.end());
		feature_vertices.erase(std::unique(feature_vertices.begin(), feature_vertices.end()), feature_vertices.end());
		feature_vertex_ratio = float(feature_vertices.size()) / float(stats.unique_vertices);
	}

	const float curvature_strength = std::max(config.curvature_adaptive_strength, 0.f);
	const float silhouette_strength = std::max(config.silhouette_preservation, 0.f);

	stats.importance = detail::clamp01(feature_vertex_ratio * 0.45f + stats.boundary * 0.25f
	                                   + stats.sharp * (0.35f + 0.25f * curvature_strength)
	                                   + stats.thin * (0.20f + 0.30f * silhouette_strength)
	                                   + stats.normal_variation * 0.20f);
	return stats;
}

float perceptualError(float geometric_error, float vertex_count, float original_count)
{
	return geometric_error * powf(vertex_count / (original_count > 0 ? original_count : 1), 0.3f);
}

void simplifyFallback(std::vector<unsigned int>& lod,
                      const clodMesh&           mesh,
                      const std::vector<unsigned int>& indices,
                      const std::vector<unsigned char>& locks,
                      size_t                    target_count,
                      float*                    error)
{
	std::vector<SloppyVertex> subset(indices.size());
	std::vector<unsigned char> subset_locks(indices.size());
	lod.resize(indices.size());
	size_t positions_stride = mesh.vertex_positions_stride / sizeof(float);
	for(size_t i = 0; i < indices.size(); ++i)
	{
		unsigned int v = indices[i];
		subset[i].x = mesh.vertex_positions[v * positions_stride + 0];
		subset[i].y = mesh.vertex_positions[v * positions_stride + 1];
		subset[i].z = mesh.vertex_positions[v * positions_stride + 2];
		subset[i].id = v;

		subset_locks[i] = locks[v];
		lod[i] = unsigned(i);
	}

	lod.resize(meshopt_simplifySloppy(&lod[0], &lod[0], lod.size(), &subset[0].x, subset.size(), sizeof(SloppyVertex),
	                                  subset_locks.data(), target_count, FLT_MAX, error));
	*error *= meshopt_simplifyScale(&subset[0].x, subset.size(), sizeof(SloppyVertex));

	for(size_t i = 0; i < lod.size(); ++i)
		lod[i] = subset[lod[i]].id;
}

std::vector<unsigned int> simplify(const clodConfig& config,
                                   const clodMesh&   mesh,
                                   const std::vector<unsigned int>& indices,
                                   const std::vector<unsigned char>& locks,
                                   size_t            target_count,
                                   float*            error)
{
	if(target_count > indices.size())
		return indices;

	size_t original_count = indices.size();
	size_t positions_stride = mesh.vertex_positions_stride / sizeof(float);

	const IndustrialFeatureStats feature_stats = computeIndustrialFeatureStats(config, mesh, indices);
	float effective_ratio = config.simplify_ratio;
	if(config.industrial_feature_preservation && feature_stats.importance > 0.f)
	{
		const float preserve = detail::clamp01(config.silhouette_preservation + config.curvature_adaptive_strength);
		effective_ratio = std::min(0.95f, config.simplify_ratio + (1.f - config.simplify_ratio) * feature_stats.importance * preserve);
	}

	size_t adaptive_target = size_t((indices.size() / 3) * effective_ratio) * 3;
	adaptive_target = std::max(adaptive_target, target_count);
	adaptive_target = std::min(adaptive_target, indices.size());

	std::vector<unsigned int> lod(indices.size());
	unsigned int options = meshopt_SimplifySparse | meshopt_SimplifyErrorAbsolute
	                       | (config.simplify_permissive ? meshopt_SimplifyPermissive : 0)
	                       | (config.simplify_regularize ? meshopt_SimplifyRegularize : 0);

	lod.resize(meshopt_simplifyWithAttributes(&lod[0], &indices[0], indices.size(),
		mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
		mesh.vertex_attributes, mesh.vertex_attributes_stride, mesh.attribute_weights, mesh.attribute_count,
		locks.data(), adaptive_target, FLT_MAX, options, error));

	if(lod.size() > adaptive_target && config.simplify_fallback_permissive && !config.simplify_permissive)
	{
		lod.resize(meshopt_simplifyWithAttributes(&lod[0], &indices[0], indices.size(),
			mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
			mesh.vertex_attributes, mesh.vertex_attributes_stride, mesh.attribute_weights, mesh.attribute_count,
			locks.data(), adaptive_target, FLT_MAX, options | meshopt_SimplifyPermissive, error));
	}

	if(lod.size() > adaptive_target && config.simplify_fallback_sloppy)
	{
		simplifyFallback(lod, mesh, indices, locks, adaptive_target, error);
		*error *= config.simplify_error_factor_sloppy;
	}

	if(config.simplify_error_edge_limit > 0)
	{
		float max_edge_sq = 0;
		for(size_t i = 0; i < indices.size(); i += 3)
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

	if(config.perceptual_weight > 0 && error != nullptr)
	{
		std::vector<unsigned int> used = lod;
		std::sort(used.begin(), used.end());
		used.erase(std::unique(used.begin(), used.end()), used.end());
		*error = perceptualError(*error, float(used.size()), float(mesh.vertex_count)) * config.perceptual_weight
		         + *error * (1.f - config.perceptual_weight);
	}

	if(config.industrial_feature_preservation && error != nullptr)
	{
		const float feature_error_scale = 1.f + feature_stats.importance * (0.75f + 1.25f * std::max(config.silhouette_preservation, 0.f));
		*error *= feature_error_scale;
	}

	return lod;
}

}  // namespace clod
