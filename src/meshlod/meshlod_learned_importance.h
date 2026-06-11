//==============================================================================
// Lightweight learned feature-importance inference for cluster LOD generation.
// The descriptor extraction is deterministic; the final importance score is
// produced only by the small MLP below so weights can be replaced by trained data.
//==============================================================================
#pragma once

#include "meshlod_impl.h"
#include <stdint.h>

namespace clod
{

static inline float learnedClamp(float v, float lo, float hi)
{
	return std::max(lo, std::min(v, hi));
}

static inline float learnedSafeSqrt(float v)
{
	return sqrtf(std::max(v, 0.0f));
}

static inline float learnedRelu(float v)
{
	return std::max(v, 0.0f);
}

static inline float learnedSigmoid(float v)
{
	if (v >= 0.0f)
	{
		float e = expf(-v);
		return 1.0f / (1.0f + e);
	}

	float e = expf(v);
	return e / (1.0f + e);
}

static inline uint64_t learnedEdgeKey(unsigned int a, unsigned int b)
{
	unsigned int lo = std::min(a, b);
	unsigned int hi = std::max(a, b);
	return (uint64_t(lo) << 32) | uint64_t(hi);
}

static inline unsigned int learnedEdgeA(uint64_t key)
{
	return unsigned(key >> 32);
}

static inline unsigned int learnedEdgeB(uint64_t key)
{
	return unsigned(key & 0xffffffffu);
}

static float learnedImportanceMlp(const float x[12])
{
	static const float w0[16][12] = {
	    { 0.10f,  1.15f,  0.10f, -0.10f,  0.20f,  0.35f,  1.30f,  1.45f,  0.15f,  0.20f,  0.45f,  1.20f},
	    {-0.15f,  0.35f,  0.20f,  0.60f,  0.10f,  0.70f,  0.85f,  0.95f,  0.25f, -0.10f,  0.20f,  0.80f},
	    { 0.35f, -0.10f,  0.70f, -0.20f,  0.35f,  0.15f,  0.05f,  0.20f,  0.70f,  0.15f,  0.10f,  0.10f},
	    {-0.25f,  0.90f, -0.10f,  0.25f,  0.50f,  0.75f,  0.55f,  0.70f, -0.15f,  0.35f,  0.25f,  0.35f},
	    { 0.15f,  0.25f,  0.05f,  1.00f, -0.10f,  0.45f,  0.30f,  0.20f,  0.10f,  0.10f,  0.15f,  0.25f},
	    { 0.25f,  0.10f,  0.40f, -0.10f,  0.85f,  0.35f,  0.15f,  0.10f,  0.20f,  0.15f,  0.10f,  0.20f},
	    {-0.20f,  0.40f,  0.15f,  0.25f,  0.25f,  1.05f,  0.55f,  0.75f,  0.20f,  0.10f,  0.70f,  0.30f},
	    { 0.30f,  0.20f,  0.20f,  0.15f,  0.15f,  0.25f,  0.10f,  0.20f,  0.85f,  0.35f,  0.20f,  0.10f},
	    { 0.05f,  0.75f,  0.10f,  0.10f,  0.10f,  0.20f,  1.10f,  1.20f,  0.10f,  0.10f,  0.35f,  0.70f},
	    { 0.45f,  0.05f,  0.15f, -0.10f,  0.20f,  0.15f,  0.05f,  0.05f,  0.25f,  0.80f,  0.15f,  0.10f},
	    {-0.15f,  0.15f,  0.25f,  0.70f,  0.70f,  0.80f,  0.25f,  0.45f,  0.20f,  0.15f,  0.65f,  0.25f},
	    { 0.20f,  0.55f,  0.10f,  0.35f,  0.25f,  0.30f,  0.35f,  0.40f,  0.15f,  0.15f,  0.20f,  0.45f},
	    { 0.15f, -0.10f,  0.65f,  0.05f,  0.10f,  0.15f,  0.05f,  0.10f,  0.70f,  0.45f,  0.10f,  0.05f},
	    {-0.10f,  0.45f,  0.05f,  0.15f,  0.25f,  0.40f,  0.75f,  0.85f,  0.10f,  0.05f,  0.25f,  0.55f},
	    { 0.10f,  0.25f,  0.15f,  0.35f,  0.35f,  0.55f,  0.35f,  0.45f,  0.15f,  0.15f,  0.35f,  0.30f},
	    { 0.25f,  0.35f,  0.25f,  0.20f,  0.20f,  0.25f,  0.20f,  0.25f,  0.25f,  0.25f,  0.25f,  0.25f},
	};

	static const float b0[16] = {
	    -0.95f, -0.75f, -0.55f, -0.85f, -0.70f, -0.70f, -0.90f, -0.80f,
	    -0.90f, -0.65f, -0.95f, -0.70f, -0.60f, -0.80f, -0.75f, -0.85f,
	};

	static const float w1[8][16] = {
	    {0.90f, 0.35f, 0.10f, 0.55f, 0.20f, 0.10f, 0.45f, 0.10f, 0.75f, 0.05f, 0.35f, 0.25f, 0.05f, 0.65f, 0.35f, 0.15f},
	    {0.20f, 0.70f, 0.15f, 0.30f, 0.75f, 0.35f, 0.65f, 0.15f, 0.45f, 0.10f, 0.65f, 0.35f, 0.10f, 0.50f, 0.55f, 0.15f},
	    {0.15f, 0.20f, 0.65f, 0.10f, 0.10f, 0.70f, 0.20f, 0.55f, 0.10f, 0.50f, 0.25f, 0.15f, 0.65f, 0.10f, 0.25f, 0.20f},
	    {0.65f, 0.45f, 0.10f, 0.60f, 0.30f, 0.15f, 0.80f, 0.15f, 0.70f, 0.05f, 0.60f, 0.50f, 0.05f, 0.70f, 0.55f, 0.25f},
	    {0.10f, 0.15f, 0.35f, 0.20f, 0.20f, 0.25f, 0.15f, 0.60f, 0.10f, 0.55f, 0.20f, 0.15f, 0.45f, 0.10f, 0.15f, 0.35f},
	    {0.40f, 0.55f, 0.20f, 0.35f, 0.45f, 0.35f, 0.50f, 0.20f, 0.45f, 0.15f, 0.55f, 0.40f, 0.15f, 0.45f, 0.50f, 0.20f},
	    {0.20f, 0.20f, 0.45f, 0.15f, 0.15f, 0.45f, 0.20f, 0.45f, 0.15f, 0.40f, 0.20f, 0.20f, 0.55f, 0.15f, 0.25f, 0.35f},
	    {0.75f, 0.65f, 0.25f, 0.70f, 0.45f, 0.30f, 0.75f, 0.30f, 0.80f, 0.20f, 0.65f, 0.55f, 0.20f, 0.75f, 0.60f, 0.40f},
	};

	static const float b1[8] = {-0.50f, -0.45f, -0.35f, -0.65f, -0.30f, -0.50f, -0.35f, -0.75f};
	static const float w2[8] = {0.95f, 0.85f, 0.55f, 1.10f, 0.45f, 0.80f, 0.50f, 1.00f};
	static const float b2 = -1.05f;

	float h0[16];
	for (size_t i = 0; i < 16; ++i)
	{
		float v = b0[i];
		for (size_t j = 0; j < 12; ++j)
			v += w0[i][j] * x[j];
		h0[i] = learnedRelu(v);
	}

	float h1[8];
	for (size_t i = 0; i < 8; ++i)
	{
		float v = b1[i];
		for (size_t j = 0; j < 16; ++j)
			v += w1[i][j] * h0[j];
		h1[i] = learnedRelu(v);
	}

	float y = b2;
	for (size_t i = 0; i < 8; ++i)
		y += w2[i] * h1[i];

	return learnedClamp(learnedSigmoid(y), 0.0f, 1.0f);
}

std::vector<float> computeLearnedImportance(const clodConfig& config, const clodMesh& mesh)
{
	std::vector<float> importance(mesh.vertex_count, 0.0f);
	if (!config.learned_importance_enable || mesh.vertex_count == 0 || mesh.index_count < 3)
		return importance;

	const size_t positionStride = mesh.vertex_positions_stride / sizeof(float);
	const float invStrength = learnedClamp(config.learned_importance_strength, 0.0f, 4.0f);

	std::vector<uint32_t> faceCount(mesh.vertex_count, 0);
	std::vector<float> normalX(mesh.vertex_count, 0.0f), normalY(mesh.vertex_count, 0.0f), normalZ(mesh.vertex_count, 0.0f);
	std::vector<float> areaSum(mesh.vertex_count, 0.0f), edgeSum(mesh.vertex_count, 0.0f);
	std::vector<float> minEdge(mesh.vertex_count, FLT_MAX), maxEdge(mesh.vertex_count, 0.0f);
	std::vector<unsigned char> boundary(mesh.vertex_count, 0), nonmanifold(mesh.vertex_count, 0);

	float bmin[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
	float bmax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

	for (size_t i = 0; i < mesh.vertex_count; ++i)
	{
		const float* p = &mesh.vertex_positions[i * positionStride];
		bmin[0] = std::min(bmin[0], p[0]); bmin[1] = std::min(bmin[1], p[1]); bmin[2] = std::min(bmin[2], p[2]);
		bmax[0] = std::max(bmax[0], p[0]); bmax[1] = std::max(bmax[1], p[1]); bmax[2] = std::max(bmax[2], p[2]);
	}

	double totalArea = 0.0;
	double totalEdge = 0.0;
	size_t edgeSamples = 0;

	for (size_t i = 0; i + 2 < mesh.index_count; i += 3)
	{
		unsigned int ia = mesh.indices[i + 0], ib = mesh.indices[i + 1], ic = mesh.indices[i + 2];
		if (ia >= mesh.vertex_count || ib >= mesh.vertex_count || ic >= mesh.vertex_count)
			continue;

		const float* a = &mesh.vertex_positions[ia * positionStride];
		const float* b = &mesh.vertex_positions[ib * positionStride];
		const float* c = &mesh.vertex_positions[ic * positionStride];

		float ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
		float ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
		float bc[3] = {c[0] - b[0], c[1] - b[1], c[2] - b[2]};
		float n[3]  = {ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]};
		float nlen  = learnedSafeSqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
		float area  = nlen * 0.5f;

		if (nlen > 0.0f)
		{
			n[0] /= nlen; n[1] /= nlen; n[2] /= nlen;
		}

		float lab = learnedSafeSqrt(ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]);
		float lac = learnedSafeSqrt(ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2]);
		float lbc = learnedSafeSqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]);

		unsigned int ids[3] = {ia, ib, ic};
		float localEdges[3][2] = {{lab, lac}, {lab, lbc}, {lac, lbc}};

		for (size_t k = 0; k < 3; ++k)
		{
			unsigned int v = ids[k];
			faceCount[v]++;
			normalX[v] += n[0]; normalY[v] += n[1]; normalZ[v] += n[2];
			areaSum[v] += area / 3.0f;
			edgeSum[v] += localEdges[k][0] + localEdges[k][1];
			minEdge[v] = std::min(minEdge[v], std::min(localEdges[k][0], localEdges[k][1]));
			maxEdge[v] = std::max(maxEdge[v], std::max(localEdges[k][0], localEdges[k][1]));
		}

		totalArea += area;
		totalEdge += double(lab + lac + lbc);
		edgeSamples += 3;
	}

	const size_t edgeCount = mesh.index_count;
	if (config.learned_importance_topology_edge_limit == 0 || edgeCount <= size_t(config.learned_importance_topology_edge_limit))
	{
		std::vector<uint64_t> edges;
		edges.reserve(edgeCount);

		for (size_t i = 0; i + 2 < mesh.index_count; i += 3)
		{
			unsigned int ia = mesh.indices[i + 0], ib = mesh.indices[i + 1], ic = mesh.indices[i + 2];
			if (ia >= mesh.vertex_count || ib >= mesh.vertex_count || ic >= mesh.vertex_count)
				continue;
			edges.push_back(learnedEdgeKey(ia, ib));
			edges.push_back(learnedEdgeKey(ib, ic));
			edges.push_back(learnedEdgeKey(ic, ia));
		}

		std::sort(edges.begin(), edges.end());

		for (size_t i = 0; i < edges.size();)
		{
			size_t j = i + 1;
			while (j < edges.size() && edges[j] == edges[i])
				++j;

			unsigned int a = learnedEdgeA(edges[i]);
			unsigned int b = learnedEdgeB(edges[i]);
			size_t count = j - i;
			if (a < mesh.vertex_count && b < mesh.vertex_count)
			{
				if (count == 1)
				{
					boundary[a] = 1;
					boundary[b] = 1;
				}
				else if (count > 2)
				{
					nonmanifold[a] = 1;
					nonmanifold[b] = 1;
				}
			}

			i = j;
		}
	}

	float meanArea = float(totalArea / std::max<size_t>(mesh.index_count / 3, 1));
	float meanEdge = float(totalEdge / std::max<size_t>(edgeSamples, 1));
	meanArea = std::max(meanArea, 1e-20f);
	meanEdge = std::max(meanEdge, 1e-10f);

	float center[3] = {(bmin[0] + bmax[0]) * 0.5f, (bmin[1] + bmax[1]) * 0.5f, (bmin[2] + bmax[2]) * 0.5f};
	float extent[3] = {std::max(bmax[0] - bmin[0], 1e-10f), std::max(bmax[1] - bmin[1], 1e-10f), std::max(bmax[2] - bmin[2], 1e-10f)};
	float invDiag = 1.0f / std::max(learnedSafeSqrt(extent[0] * extent[0] + extent[1] * extent[1] + extent[2] * extent[2]), 1e-10f);

	for (size_t i = 0; i < mesh.vertex_count; ++i)
	{
		const float* p = &mesh.vertex_positions[i * positionStride];
		float faces = float(std::max<uint32_t>(faceCount[i], 1));
		float nlen = learnedSafeSqrt(normalX[i] * normalX[i] + normalY[i] * normalY[i] + normalZ[i] * normalZ[i]);
		float normalVariation = 1.0f - learnedClamp(nlen / faces, 0.0f, 1.0f);
		float localMinEdge = minEdge[i] == FLT_MAX ? meanEdge : minEdge[i];
		float localMaxEdge = maxEdge[i] > 0.0f ? maxEdge[i] : meanEdge;
		float localMeanEdge = edgeSum[i] / std::max(2.0f * faces, 1.0f);
		float edgeContrast = learnedClamp((localMaxEdge - localMinEdge) / std::max(localMaxEdge, meanEdge), 0.0f, 1.0f);
		float areaRatio = learnedClamp(areaSum[i] / (meanArea * faces), 0.0f, 4.0f) * 0.25f;
		float shortEdge = learnedClamp(1.0f - localMinEdge / meanEdge, 0.0f, 1.0f);
		float longEdge = learnedClamp(localMaxEdge / meanEdge - 1.0f, 0.0f, 3.0f) / 3.0f;
		float valenceNorm = learnedClamp(faces / 12.0f, 0.0f, 1.0f);
		float lowValence = learnedClamp((3.0f - faces) / 3.0f, 0.0f, 1.0f);
		float anisotropy = learnedClamp(localMeanEdge > 0.0f ? (localMaxEdge / localMeanEdge - 1.0f) : 0.0f, 0.0f, 3.0f) / 3.0f;
		float dx = (p[0] - center[0]) * invDiag;
		float dy = (p[1] - center[1]) * invDiag;
		float dz = (p[2] - center[2]) * invDiag;
		float radial = learnedClamp(learnedSafeSqrt(dx * dx + dy * dy + dz * dz) * 2.0f, 0.0f, 1.0f);
		float locked = mesh.vertex_lock && (mesh.vertex_lock[i] & meshopt_SimplifyVertex_Protect) ? 1.0f : 0.0f;

		float descriptor[12] = {
		    valenceNorm,
		    normalVariation,
		    areaRatio,
		    shortEdge,
		    longEdge,
		    edgeContrast,
		    boundary[i] ? 1.0f : 0.0f,
		    nonmanifold[i] ? 1.0f : 0.0f,
		    anisotropy,
		    lowValence,
		    radial,
		    locked,
		};

		importance[i] = learnedClamp(learnedImportanceMlp(descriptor) * invStrength, 0.0f, 1.0f);
	}

	return importance;
}

}
