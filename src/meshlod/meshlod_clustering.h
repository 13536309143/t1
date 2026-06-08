//==============================================================================
// 文件：src/meshlod/meshlod_clustering.h
// 模块定位：簇 分组和边界锁定逻辑，组织 meshoptimizer 结果并保护需要保留的拓扑边界。
// 数据流：输入是简化前后的网格与邻接信息；输出是 簇 列表、组 划分和锁定约束。
// 方法说明：聚类阶段在局部渲染批次大小和拓扑连续性之间折中，边界锁定用于避免简化破坏轮廓或裂缝。
// 正确性约束：组 内 簇 必须共享同一次简化语义；边界锁定不能导致后续简化无法满足基本三角形预算。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#pragma once


// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include "meshlod_impl.h"

namespace clod
{


// 函数：clusterFeatureImportance。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
static float clusterFeatureImportance(const std::vector<unsigned int>& indices, const std::vector<float>* feature_importance)
{
	if (!feature_importance || feature_importance->empty())
		return 0.f;

	float sum = 0.f;
	float max_feature = 0.f;
	float strong_count = 0.f;
	size_t count = 0;

	for (size_t i = 0; i < indices.size(); ++i)
	{
		unsigned int v = indices[i];
		if (v < feature_importance->size())
		{
			float f = (*feature_importance)[v];
			sum += f;

			max_feature = std::max(max_feature, f);
			if (f > 0.65f)
				strong_count += 1.f;
			count++;
		}
	}

	float avg = count ? sum / float(count) : 0.f;
	float strong_ratio = count ? strong_count / float(count) : 0.f;
	return std::min(1.f, avg * 0.5f + max_feature * 0.3f + strong_ratio * 0.2f);
}


// 函数：clusterize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
std::vector<Cluster> clusterize(const clodConfig& config, const clodMesh& mesh, const unsigned int* indices, size_t index_count, const std::vector<float>* feature_importance)
{

	size_t max_meshlets = meshopt_buildMeshletsBound(index_count, config.max_vertices, config.min_triangles);


	// 函数：meshlets。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<meshopt_Meshlet> meshlets(max_meshlets);


	// 函数：meshlet_vertices。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<unsigned int> meshlet_vertices(index_count);

#if MESHOPTIMIZER_VERSION < 1000


	// 函数：meshlet_triangles。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<unsigned char> meshlet_triangles(index_count + max_meshlets * 3);
#else


	// 函数：meshlet_triangles。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<unsigned char> meshlet_triangles(index_count);
#endif

	if (config.cluster_spatial)
		meshlets.resize(meshopt_buildMeshletsSpatial(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices, index_count,
		    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
		    config.max_vertices, config.min_triangles, config.max_triangles, config.cluster_fill_weight));
	else
		meshlets.resize(meshopt_buildMeshletsFlex(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices, index_count,
		    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
		    config.max_vertices, config.min_triangles, config.max_triangles, 0.f, config.cluster_split_factor));


	// 函数：clusters。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<Cluster> clusters(meshlets.size());

	for (size_t i = 0; i < meshlets.size(); ++i)
	{
		const meshopt_Meshlet& meshlet = meshlets[i];
		if (config.optimize_clusters)

			meshopt_optimizeMeshlet(&meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset], meshlet.triangle_count, meshlet.vertex_count);
		clusters[i].vertices = meshlet.vertex_count;

		clusters[i].indices.resize(meshlet.triangle_count * 3);
		for (size_t j = 0; j < meshlet.triangle_count * 3; ++j)
			clusters[i].indices[j] = meshlet_vertices[meshlet.vertex_offset + meshlet_triangles[meshlet.triangle_offset + j]];
		clusters[i].group = -1;
		clusters[i].refined = -1;

		clusters[i].feature_importance = clusterFeatureImportance(clusters[i].indices, feature_importance);
	}

	return clusters;
}


// 函数：partition。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
std::vector<std::vector<int> > partition(const clodConfig& config, const clodMesh& mesh, const std::vector<Cluster>& clusters, const std::vector<int>& pending, const std::vector<unsigned int>& remap)
{
	if (pending.size() <= config.partition_size)
		return {pending};

	std::vector<unsigned int> cluster_indices;


	// 函数：cluster_counts。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<unsigned int> cluster_counts(pending.size());
	size_t total_index_count = 0;
	for (size_t i = 0; i < pending.size(); ++i)

		total_index_count += clusters[pending[i]].indices.size();


	cluster_indices.reserve(total_index_count);
	for (size_t i = 0; i < pending.size(); ++i)
	{
		const Cluster& cluster = clusters[pending[i]];
		cluster_counts[i] = unsigned(cluster.indices.size());
		for (size_t j = 0; j < cluster.indices.size(); ++j)

			cluster_indices.push_back(remap[cluster.indices[j]]);
	}


	// 函数：cluster_part。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<unsigned int> cluster_part(pending.size());
	size_t partition_count = meshopt_partitionClusters(&cluster_part[0], &cluster_indices[0], cluster_indices.size(), &cluster_counts[0], cluster_counts.size(),
	    config.partition_spatial ? mesh.vertex_positions : NULL, remap.size(), mesh.vertex_positions_stride, config.partition_size);


	// 函数：partitions。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
	// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
	// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
	std::vector<std::vector<int> > partitions(partition_count);
	for (size_t i = 0; i < partition_count; ++i)

		partitions[i].reserve(config.partition_size + config.partition_size / 3);

	std::vector<unsigned int> partition_remap;
	if (config.partition_sort)
	{


		// 函数：partition_point。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
		// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
		// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
		std::vector<float> partition_point(partition_count * 3);
		for (size_t i = 0; i < pending.size(); ++i)
			memcpy(&partition_point[cluster_part[i] * 3], clusters[pending[i]].bounds.center, sizeof(float) * 3);

		partition_remap.resize(partition_count);
		meshopt_spatialSortRemap(partition_remap.data(), partition_point.data(), partition_count, sizeof(float) * 3);
	}

	for (size_t i = 0; i < pending.size(); ++i)

		partitions[partition_remap.empty() ? cluster_part[i] : partition_remap[cluster_part[i]]].push_back(pending[i]);

	if (config.curvature_adaptive_strength <= 0.f && config.silhouette_preservation <= 0.f)
		return partitions;

	std::vector<std::vector<int> > feature_partitions;
	feature_partitions.reserve(partitions.size());

	for (size_t i = 0; i < partitions.size(); ++i)
	{
		const std::vector<int>& part = partitions[i];
		float max_feature = 0.f;
		float avg_feature = 0.f;
		float strong_feature_count = 0.f;

		for (size_t j = 0; j < part.size(); ++j)
		{
			float feature = clusters[part[j]].feature_importance;
			avg_feature += feature;

			max_feature = std::max(max_feature, feature);
			if (feature > 0.65f)
				strong_feature_count += 1.f;
		}

		avg_feature = part.empty() ? 0.f : avg_feature / float(part.size());
		float strong_ratio = part.empty() ? 0.f : strong_feature_count / float(part.size());

		float feature_pressure = std::min(1.f, avg_feature * 0.45f + max_feature * 0.35f + strong_ratio * 0.2f);
		size_t feature_limit = config.partition_size;

		if (feature_pressure > 0.35f)
		{

			float strength = std::min(1.f, config.curvature_adaptive_strength + config.silhouette_preservation);
			feature_limit = std::max<size_t>(4, size_t(float(config.partition_size) * (1.f - 0.45f * strength * feature_pressure)));
		}

		for (size_t begin = 0; begin < part.size(); begin += feature_limit)
		{
			size_t end = std::min(begin + feature_limit, part.size());
			feature_partitions.emplace_back(part.begin() + begin, part.begin() + end);
		}
	}

	return feature_partitions;
}


// 函数：lockBoundary。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void lockBoundary(std::vector<unsigned char>& locks, const std::vector<std::vector<int> >& groups, const std::vector<Cluster>& clusters, const std::vector<unsigned int>& remap, const unsigned char* vertex_lock)
{
	for (size_t i = 0; i < locks.size(); ++i)
		locks[i] &= ~((1 << 0) | (1 << 7));

	for (size_t i = 0; i < groups.size(); ++i)
	{
		for (size_t j = 0; j < groups[i].size(); ++j)
		{
			const Cluster& cluster = clusters[groups[i][j]];

			for (size_t k = 0; k < cluster.indices.size(); ++k)
			{
				unsigned int v = cluster.indices[k];
				unsigned int r = remap[v];
				locks[r] |= locks[r] >> 7;
			}
		}

		for (size_t j = 0; j < groups[i].size(); ++j)
		{
			const Cluster& cluster = clusters[groups[i][j]];

			for (size_t k = 0; k < cluster.indices.size(); ++k)
			{
				unsigned int v = cluster.indices[k];
				unsigned int r = remap[v];

				locks[r] |= 1 << 7;
			}
		}
	}

	for (size_t i = 0; i < locks.size(); ++i)
	{
		unsigned int r = remap[i];
		locks[i] = (locks[r] & 1) | (locks[i] & meshopt_SimplifyVertex_Protect);
		if (vertex_lock)
			locks[i] |= vertex_lock[i];
	}
}

}
