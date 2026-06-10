//==============================================================================
// 文件：src/meshlod/meshlod_simplify.h
// 模块定位：网格简化实现，处理特征边、曲率、感知误差、轮廓保护和回退简化策略。
// 数据流：输入是当前 LOD 网格和权重配置；输出是更低复杂度的网格以及用于遍历的误差估计。
// 方法说明：简化阶段不仅最小化几何误差，还引入法线、纹理坐标、切线和轮廓相关权重，以提高视觉一致性。
// 正确性约束：回退路径必须保证算法收敛；误差度量应随简化增加保持非负并可用于屏幕空间阈值比较。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#pragma once


// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include "meshlod_impl.h"

namespace clod
{


// 结构：FeatureEdge。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct FeatureEdge
{
	unsigned int a;
	unsigned int b;
	unsigned int v0;
	unsigned int v1;
	unsigned int face;
	float length;
};

struct BoundarySegment
{
	unsigned int v0;
	unsigned int v1;
	float length;
};


// 函数：clamp01。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
static float clamp01(float v)
{
	return std::min(1.f, std::max(0.f, v));
}


// 函数：positionSub。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
static void positionSub(float r[3], const float* a, const float* b)
{
	r[0] = a[0] - b[0];
	r[1] = a[1] - b[1];
	r[2] = a[2] - b[2];
}


// 函数：dot3。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
static float dot3(const float* a, const float* b)
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}


// 函数：normalize3。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
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


// 函数：length3。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
static float length3(const float* v)
{
	return sqrtf(dot3(v, v));
}

static void detectFunctionalBoundaryLoops(const clodMesh& mesh, size_t stride, const std::vector<float>& mean_normals, const std::vector<BoundarySegment>& boundary_segments, float model_scale, std::vector<unsigned int>& boundary_degree, std::vector<float>& circular_hole, std::vector<float>& functional_boundary, uint64_t* component_count = nullptr, uint64_t* circular_loop_count = nullptr)
{
	if (boundary_segments.empty() || mesh.vertex_count == 0)
		return;

	std::vector<unsigned int> degree(mesh.vertex_count, 0);
	for (const BoundarySegment& segment : boundary_segments)
	{
		if (segment.v0 < mesh.vertex_count && segment.v1 < mesh.vertex_count)
		{
			degree[segment.v0]++;
			degree[segment.v1]++;
		}
	}

	std::vector<unsigned int> offsets(mesh.vertex_count + 1, 0);
	for (size_t v = 0; v < mesh.vertex_count; ++v)
		offsets[v + 1] = offsets[v] + degree[v];

	std::vector<unsigned int> cursor(offsets);
	std::vector<unsigned int> adjacency(offsets.back());
	for (const BoundarySegment& segment : boundary_segments)
	{
		if (segment.v0 < mesh.vertex_count && segment.v1 < mesh.vertex_count)
		{
			adjacency[cursor[segment.v0]++] = segment.v1;
			adjacency[cursor[segment.v1]++] = segment.v0;
		}
	}

	std::vector<unsigned char> visited(mesh.vertex_count, 0);
	std::vector<unsigned int> component;
	std::vector<unsigned int> queue;
	component.reserve(64);
	queue.reserve(64);

	for (const BoundarySegment& seed_segment : boundary_segments)
	{
		unsigned int seeds[2] = {seed_segment.v0, seed_segment.v1};
		for (unsigned int seed : seeds)
		{
			if (seed >= mesh.vertex_count || visited[seed])
				continue;

			component.clear();
			queue.clear();
			visited[seed] = 1;
			queue.push_back(seed);

			for (size_t head = 0; head < queue.size(); ++head)
			{
				unsigned int v = queue[head];
				component.push_back(v);

				for (unsigned int e = offsets[v]; e < offsets[v + 1]; ++e)
				{
					unsigned int next = adjacency[e];
					if (next < mesh.vertex_count && !visited[next])
					{
						visited[next] = 1;
						queue.push_back(next);
					}
				}
			}

			if (component.size() < 3)
				continue;

			if (component_count)
				(*component_count)++;

			float center[3] = {};
			float normal[3] = {};
			unsigned int degree2 = 0;
			for (unsigned int v : component)
			{
				const float* p = &mesh.vertex_positions[v * stride];
				center[0] += p[0];
				center[1] += p[1];
				center[2] += p[2];

				normal[0] += mean_normals[v * 3 + 0];
				normal[1] += mean_normals[v * 3 + 1];
				normal[2] += mean_normals[v * 3 + 2];

				if (degree[v] == 2)
					degree2++;
			}

			float inv_count = 1.f / float(component.size());
			center[0] *= inv_count;
			center[1] *= inv_count;
			center[2] *= inv_count;
			float normal_len = normalize3(normal);

			float mean_radius = 0.f;
			float mean_plane_deviation = 0.f;
			for (unsigned int v : component)
			{
				const float* p = &mesh.vertex_positions[v * stride];
				float d[3] = {p[0] - center[0], p[1] - center[1], p[2] - center[2]};
				mean_radius += length3(d);
				if (normal_len > 1e-8f)
					mean_plane_deviation += fabsf(dot3(d, normal));
			}
			mean_radius *= inv_count;
			mean_plane_deviation *= inv_count;

			float radius_variance = 0.f;
			for (unsigned int v : component)
			{
				const float* p = &mesh.vertex_positions[v * stride];
				float d[3] = {p[0] - center[0], p[1] - center[1], p[2] - center[2]};
				float radius = length3(d);
				float delta = radius - mean_radius;
				radius_variance += delta * delta;
			}
			radius_variance *= inv_count;

			float relative_radius_sigma = mean_radius > 1e-8f ? sqrtf(radius_variance) / mean_radius : 1.f;
			float relative_planarity = mean_radius > 1e-8f ? mean_plane_deviation / mean_radius : 1.f;
			float degree2_ratio = float(degree2) * inv_count;
			float size_score = clamp01((float(component.size()) - 5.f) / 12.f);
			float roundness_score = clamp01((0.28f - relative_radius_sigma) / 0.28f);
			float planarity_score = normal_len > 1e-8f ? clamp01((0.16f - relative_planarity) / 0.16f) : 0.5f;
			float loop_score = size_score * roundness_score * planarity_score * clamp01((degree2_ratio - 0.5f) / 0.5f);
			float boundary_score = clamp01(0.35f + degree2_ratio * 0.35f);

			if (mean_radius > model_scale * 0.35f)
				loop_score *= 0.65f;

			if (loop_score > 0.1f && circular_loop_count)
				(*circular_loop_count)++;

			for (unsigned int v : component)
			{
				functional_boundary[v] = std::max(functional_boundary[v], boundary_score);
				boundary_degree[v] = std::max(boundary_degree[v], degree[v]);
				if (loop_score > 0.1f)
					circular_hole[v] = std::max(circular_hole[v], clamp01(0.45f + loop_score * 0.55f));
			}
		}
	}
}


// 函数：computeFeatureImportance。计算派生值，供后续剔除、LOD、统计或资源规划使用。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：计算结果通常参与阈值比较或内存规划，数值稳定性和边界条件需要特别注意。
std::vector<float> computeFeatureImportance(const clodConfig& config, const clodMesh& mesh, const std::vector<unsigned int>& remap)
{


	// 函数：importance。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<float> importance(mesh.vertex_count, 0.f);
	if (!mesh.indices || mesh.index_count < 3 || !mesh.vertex_positions)
		return importance;

	const size_t stride = mesh.vertex_positions_stride / sizeof(float);
	const size_t face_count = mesh.index_count / 3;
	const float sharp_threshold = config.feature_edge_threshold > 0.f ? config.feature_edge_threshold : 0.5f;

	const float cos_threshold = cosf(sharp_threshold);


	// 函数：face_normals。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<float> face_normals(face_count * 3, 0.f);


	// 函数：mean_normals。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<float> mean_normals(mesh.vertex_count * 3, 0.f);


	// 函数：normal_variation。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<float> normal_variation(mesh.vertex_count, 0.f);


	// 函数：thin_wall。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<float> thin_wall(mesh.vertex_count, 0.f);
	std::vector<float> circular_hole(mesh.vertex_count, 0.f);
	std::vector<float> cylindrical_patch(mesh.vertex_count, 0.f);
	std::vector<float> functional_boundary(mesh.vertex_count, 0.f);
	std::vector<float> curved_edge_sum(mesh.vertex_count, 0.f);
	std::vector<unsigned int> curved_edge_count(mesh.vertex_count, 0);


	// 函数：boundary。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<float> boundary(mesh.vertex_count, 0.f);


	// 函数：non_manifold。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<float> non_manifold(mesh.vertex_count, 0.f);


	// 函数：boundary_degree。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<unsigned int> boundary_degree(mesh.vertex_count, 0);


	// 函数：local_min_edge。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<float> local_min_edge(mesh.vertex_count, FLT_MAX);


	// 函数：local_max_edge。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<float> local_max_edge(mesh.vertex_count, 0.f);


	// 函数：local_edge_sum。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<float> local_edge_sum(mesh.vertex_count, 0.f);


	// 函数：local_edge_count。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<unsigned int> local_edge_count(mesh.vertex_count, 0);
	std::vector<FeatureEdge> edges;
	std::vector<BoundarySegment> boundary_segments;
	std::vector<BoundarySegment> sharp_loop_segments;
	uint64_t boundary_loop_components = 0;
	uint64_t sharp_ring_components = 0;
	uint64_t circular_boundary_loops = 0;
	uint64_t circular_sharp_rings = 0;

	edges.reserve(mesh.index_count);
	boundary_segments.reserve(mesh.index_count / 6);
	sharp_loop_segments.reserve(mesh.index_count / 8);

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
			edge_score = 1.f;
			boundary[edges[begin].v0] = 1.f;
			boundary[edges[begin].v1] = 1.f;
			boundary_degree[edges[begin].v0]++;
			boundary_degree[edges[begin].v1]++;
			boundary_segments.push_back({edges[begin].v0, edges[begin].v1, edges[begin].length});
		}
		else if (end - begin > 2)
		{
			edge_score = 1.f;
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
			{
				edge_score = clamp01((cos_threshold - d) / std::max(cos_threshold + 1.f, 1e-5f));
				if (edge_score > 0.18f)
				{
					for (size_t i = begin; i < end; ++i)
						sharp_loop_segments.push_back({edges[i].v0, edges[i].v1, edges[i].length});
				}
			}
			else
			{
				float curved_score = clamp01((1.f - d) / 0.08f);
				if (curved_score > 0.02f)
				{
					for (size_t i = begin; i < end; ++i)
					{
						curved_edge_sum[edges[i].v0] += curved_score;
						curved_edge_sum[edges[i].v1] += curved_score;
						curved_edge_count[edges[i].v0]++;
						curved_edge_count[edges[i].v1]++;
					}
				}
			}
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

	detectFunctionalBoundaryLoops(mesh, stride, mean_normals, boundary_segments, model_scale, boundary_degree, circular_hole, functional_boundary, &boundary_loop_components, &circular_boundary_loops);
	detectFunctionalBoundaryLoops(mesh, stride, mean_normals, sharp_loop_segments, model_scale, boundary_degree, circular_hole, functional_boundary, &sharp_ring_components, &circular_sharp_rings);

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
		float max_edge = local_max_edge[v];
		float slender_score = (avg_edge > 0.f && min_edge > 0.f) ? clamp01((avg_edge / min_edge - 2.f) / 6.f) : 0.f;
		float anisotropy_score = (max_edge > 0.f && min_edge > 0.f) ? clamp01((max_edge / min_edge - 3.f) / 10.f) : 0.f;
		float small_scale_score = avg_edge > 0.f ? clamp01((model_scale * 0.015f - avg_edge) / (model_scale * 0.015f)) : 0.f;
		float hole_loop_score = std::max(circular_hole[v], boundary_degree[v] >= 2 ? 0.65f : boundary[v]);
		float curved_score = curved_edge_count[v] ? clamp01(curved_edge_sum[v] / float(curved_edge_count[v]) * 1.4f) : 0.f;
		float cylinder_score = clamp01(curved_score * 0.55f + normal_variation[v] * 0.65f);
		cylinder_score *= 1.f - boundary[v] * 0.35f;
		cylindrical_patch[v] = std::max(cylindrical_patch[v], cylinder_score);


		thin_wall[v] = std::max(thin_wall[v], slender_score * 0.45f + anisotropy_score * 0.25f + small_scale_score * 0.2f + functional_boundary[v] * 0.1f);

		float weighted =
			boundary[v] * 0.18f +
			functional_boundary[v] * 0.18f +
			hole_loop_score * 0.24f +
			cylindrical_patch[v] * 0.18f +
			non_manifold[v] * 0.20f +
			importance[v] * 0.18f +
			normal_variation[v] * 0.12f +
			thin_wall[v] * 0.18f;

		importance[v] = std::max(importance[v], clamp01(weighted));

		if (circular_hole[v] > 0.35f)
			importance[v] = std::max(importance[v], clamp01(0.88f + circular_hole[v] * 0.12f));
		else if (functional_boundary[v] > 0.55f)
			importance[v] = std::max(importance[v], clamp01(0.68f + functional_boundary[v] * 0.18f));

		if (cylindrical_patch[v] > 0.55f)
			importance[v] = std::max(importance[v], clamp01(0.58f + cylindrical_patch[v] * 0.22f));
	}


	// 函数：propagated。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<float> propagated(importance);
	for (size_t i = 0; i < edges.size(); ++i)
	{

		float line_strength = std::max(importance[edges[i].v0], importance[edges[i].v1]);
		if (line_strength > 0.45f)
		{
			float semantic_strength = std::max(circular_hole[edges[i].v0], circular_hole[edges[i].v1]);
			semantic_strength = std::max(semantic_strength, std::max(cylindrical_patch[edges[i].v0], cylindrical_patch[edges[i].v1]));
			float attenuated = line_strength * (semantic_strength > 0.55f ? 0.72f : 0.55f);

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


		// 函数：representative。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
		// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
		// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
		std::vector<float> representative(mesh.vertex_count, 0.f);
		for (size_t v = 0; v < mesh.vertex_count; ++v)

			representative[remap[v]] = std::max(representative[remap[v]], importance[v]);
		for (size_t v = 0; v < mesh.vertex_count; ++v)

			importance[v] = std::max(importance[v], representative[remap[v]]);
	}

	if (mesh.feature_metrics)
	{
		clodFeatureMetrics metrics = {};
		metrics.input_vertices = mesh.vertex_count;
		metrics.input_triangles = mesh.index_count / 3;
		metrics.boundary_loop_components = boundary_loop_components;
		metrics.sharp_ring_components = sharp_ring_components;
		metrics.circular_hole_loops = circular_boundary_loops + circular_sharp_rings;

		for (size_t v = 0; v < mesh.vertex_count; ++v)
		{
			const float feature = clamp01(importance[v]);
			const uint64_t feature_ppm = uint64_t(feature * 1000000.f + 0.5f);
			metrics.feature_importance_sum_ppm += feature_ppm;
			metrics.feature_importance_max_ppm = std::max(metrics.feature_importance_max_ppm, feature_ppm);

			if (boundary[v] > 0.5f)
				metrics.boundary_vertices++;
			if (non_manifold[v] > 0.5f)
				metrics.non_manifold_vertices++;
			if (circular_hole[v] > 0.35f)
				metrics.circular_hole_vertices++;
			if (functional_boundary[v] > 0.55f)
				metrics.functional_boundary_vertices++;
			if (cylindrical_patch[v] > 0.55f)
				metrics.cylindrical_patch_vertices++;
			if (thin_wall[v] > 0.5f)
				metrics.thin_wall_vertices++;
			if (feature >= 0.78f)
				metrics.protected_feature_vertices++;
			if (feature >= 0.92f)
				metrics.critical_feature_vertices++;
		}

		std::vector<unsigned char> sharp_seen(mesh.vertex_count, 0);
		for (const BoundarySegment& segment : sharp_loop_segments)
		{
			if (segment.v0 < mesh.vertex_count)
				sharp_seen[segment.v0] = 1;
			if (segment.v1 < mesh.vertex_count)
				sharp_seen[segment.v1] = 1;
		}
		for (size_t v = 0; v < mesh.vertex_count; ++v)
		{
			if (sharp_seen[v])
				metrics.sharp_feature_vertices++;
		}

		*mesh.feature_metrics = metrics;
	}

	return importance;
}


// 函数：featureStats。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
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


// 函数：featureAdaptiveTarget。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
static size_t featureAdaptiveTarget(const clodConfig& config, const std::vector<unsigned int>& indices, const std::vector<float>& feature_importance, size_t target_count)
{
	float avg_feature = 0.f;
	float max_feature = 0.f;
	float critical_count = 0.f;
	float protected_count = 0.f;

	featureStats(indices, feature_importance, avg_feature, max_feature);

	if (!feature_importance.empty())
	{
		for (size_t i = 0; i < indices.size(); ++i)
		{
			unsigned int v = indices[i];
			if (v < feature_importance.size())
			{
				if (feature_importance[v] >= 0.92f)
					critical_count += 1.f;
				else if (feature_importance[v] >= 0.78f)
					protected_count += 1.f;
			}
		}
	}

	float critical_ratio = indices.empty() ? 0.f : critical_count / float(indices.size());
	float protected_ratio = indices.empty() ? 0.f : protected_count / float(indices.size());


	float pressure = clamp01(avg_feature * 0.45f + max_feature * 0.25f + critical_ratio * 0.22f + protected_ratio * 0.08f);

	float strength = clamp01(config.curvature_adaptive_strength + config.silhouette_preservation);
	float preserve = 1.f - config.simplify_ratio;
	size_t relaxed = target_count + size_t(float(indices.size() - target_count) * preserve * strength * pressure);
	if (critical_count > 0.f)
	{
		size_t critical_floor = target_count + size_t(float(indices.size() - target_count) * clamp01(0.35f + strength * (0.25f + critical_ratio)));
		relaxed = std::max(relaxed, critical_floor);
	}
	relaxed = (relaxed / 3) * 3;

	return std::max(target_count, std::min(indices.size(), relaxed));
}


// 函数：applyFeatureLocks。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
static void applyFeatureLocks(const clodConfig& config, const std::vector<unsigned int>& indices, const std::vector<float>& feature_importance, std::vector<unsigned char>& locks)
{
	if (feature_importance.empty() || config.silhouette_preservation <= 0.f)
		return;


	float lock_threshold = 0.9f - 0.25f * clamp01(config.silhouette_preservation);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		unsigned int v = indices[i];
		if (v < feature_importance.size() && feature_importance[v] >= lock_threshold)
			locks[v] |= meshopt_SimplifyVertex_Protect;
	}
}


// 函数：perceptualError。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
float perceptualError(float geometric_error, float vertex_count, float original_count)
{
	return geometric_error * powf(vertex_count / (original_count > 0 ? original_count : 1), 0.3f);
}


// 函数：simplifyFallback。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void simplifyFallback(std::vector<unsigned int>& lod, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, size_t target_count, float* error)
{


	// 函数：subset。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<SloppyVertex> subset(indices.size());


	// 函数：subset_locks。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
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


// 函数：simplify。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
std::vector<unsigned int> simplify(const clodConfig& config, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, const std::vector<float>& feature_importance, size_t target_count, float* error)
{
	if (target_count > indices.size())
		return indices;

	size_t positions_stride = mesh.vertex_positions_stride / sizeof(float);

	size_t adaptive_target = featureAdaptiveTarget(config, indices, feature_importance, target_count);

	float avg_feature = 0.f;
	float max_feature = 0.f;

	featureStats(indices, feature_importance, avg_feature, max_feature);


	// 函数：enhanced_locks。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<unsigned char> enhanced_locks(locks);

	applyFeatureLocks(config, indices, feature_importance, enhanced_locks);


	// 函数：lod。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
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


		// 函数：vertex_used。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
		// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
		// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
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
