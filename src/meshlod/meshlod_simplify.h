//==============================================================================
// 文件：src/meshlod/meshlod_simplify.h
// 模块定位：meshoptimizer 原始风格的 LOD 简化路径，只保留边界锁、属性权重和回退简化。
// 数据流：输入当前 LOD group 的索引、顶点和属性；输出降低复杂度后的索引和简化误差。
// 方法说明：本文件刻意不再使用孔洞、薄壁、圆柱、功能区等人工特征约束，回到基础 cluster LOD 生成策略。
//==============================================================================
#pragma once

#include "meshlod_impl.h"
#include "meshlod_learned_importance.h"

namespace clod
{

void simplifyFallback(std::vector<unsigned int>& lod, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, size_t target_count, float* error)
{
	std::vector<SloppyVertex> subset(indices.size());
	std::vector<unsigned char> subset_locks(indices.size());

	lod.resize(indices.size());

	size_t positions_stride = mesh.vertex_positions_stride / sizeof(float);

	for (size_t i = 0; i < indices.size(); ++i)
	{
		unsigned int v = indices[i];
		assert(v < mesh.vertex_count);

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

	float learnedAverage = 0.0f;
	float learnedMaximum = 0.0f;
	if (config.learned_importance_enable && mesh.learned_importance && !indices.empty())
	{
		double sum = 0.0;
		for (size_t i = 0; i < indices.size(); ++i)
		{
			unsigned int v = indices[i];
			if (v < mesh.vertex_count)
			{
				float importance = mesh.learned_importance[v];
				sum += double(importance);
				learnedMaximum = std::max(learnedMaximum, importance);
			}
		}

		learnedAverage = float(sum / double(indices.size()));
		float preserve = learnedClamp(learnedAverage * config.learned_importance_target_boost, 0.0f, 0.85f);
		size_t boostedTarget = target_count + size_t(float(indices.size() - target_count) * preserve);
		target_count = std::min(indices.size(), boostedTarget);
		target_count = std::max<size_t>(3, (target_count / 3) * 3);
	}

	std::vector<unsigned char> learnedLocks;
	const unsigned char* simplifyLocks = locks.empty() ? nullptr : &locks[0];
	if (config.learned_importance_enable && mesh.learned_importance && config.learned_importance_protect_threshold < 1.0f)
	{
		learnedLocks = locks;
		float threshold = learnedClamp(config.learned_importance_protect_threshold, 0.0f, 1.0f);

		for (size_t i = 0; i < mesh.vertex_count && i < learnedLocks.size(); ++i)
		{
			if (mesh.learned_importance[i] >= threshold)
				learnedLocks[i] |= meshopt_SimplifyVertex_Protect;
		}

		simplifyLocks = learnedLocks.empty() ? simplifyLocks : learnedLocks.data();
	}

	std::vector<unsigned int> lod(indices.size());

	unsigned int options = meshopt_SimplifySparse | meshopt_SimplifyErrorAbsolute | (config.simplify_permissive ? meshopt_SimplifyPermissive : 0) | (config.simplify_regularize ? meshopt_SimplifyRegularize : 0);

	lod.resize(meshopt_simplifyWithAttributes(&lod[0], &indices[0], indices.size(),
	    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
	    mesh.vertex_attributes, mesh.vertex_attributes_stride, mesh.attribute_weights, mesh.attribute_count,
	    simplifyLocks, target_count, FLT_MAX, options, error));

	if (lod.size() > target_count && config.simplify_fallback_permissive && !config.simplify_permissive)
	{
		lod.resize(meshopt_simplifyWithAttributes(&lod[0], &indices[0], indices.size(),
		    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
		    mesh.vertex_attributes, mesh.vertex_attributes_stride, mesh.attribute_weights, mesh.attribute_count,
		    simplifyLocks, target_count, FLT_MAX, options | meshopt_SimplifyPermissive, error));
	}

	if (lod.size() > target_count && config.simplify_fallback_sloppy)
	{
		simplifyFallback(lod, mesh, indices, learnedLocks.empty() ? locks : learnedLocks, target_count, error);
		*error *= config.simplify_error_factor_sloppy;
	}

	if (config.simplify_error_edge_limit > 0)
	{
		float max_edge_sq = 0;
		size_t positions_stride = mesh.vertex_positions_stride / sizeof(float);

		for (size_t i = 0; i < indices.size(); i += 3)
		{
			unsigned int a = indices[i + 0], b = indices[i + 1], c = indices[i + 2];
			assert(a < mesh.vertex_count && b < mesh.vertex_count && c < mesh.vertex_count);

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

	if (config.learned_importance_enable && mesh.learned_importance)
	{
		float scale = 1.0f + learnedClamp(config.learned_importance_error_scale, 0.0f, 8.0f) * (learnedAverage * 0.7f + learnedMaximum * 0.3f);
		*error *= scale;
	}

	return lod;
}

}
