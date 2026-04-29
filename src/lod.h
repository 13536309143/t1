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
#ifdef __cplusplus
extern "C"
{
#endif
clodConfig clodDefaultConfig(size_t max_triangles);
size_t clodBuild(clodConfig config, clodMesh mesh, void* output_context, clodOutput output_callback, clodIteration iteration_callback);
void clodBuild_iterationTask(void* iteration_context, void* output_context, size_t task_index, unsigned int thread_index);
size_t clodLocalIndices(unsigned int* vertices, unsigned char* triangles, const unsigned int* indices, size_t index_count);
#ifdef __cplusplus
} 
template <typename Output>
size_t clodBuild(clodConfig config, clodMesh mesh, Output output)
{
	struct Call
	{
		static int output(void* output_context, clodGroup group, const clodCluster* clusters, size_t cluster_count)
		{
			return (*static_cast<Output*>(output_context))(group, clusters, cluster_count);
		}
	};

	return clodBuild(config, mesh, &output, &Call::output, nullptr);
}
#endif
#ifdef CLUSTERLOD

#include <float.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <atomic>

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

static clodBounds boundsCompute(const clodMesh& mesh, const std::vector<unsigned int>& indices, float error)
{
	meshopt_Bounds bounds = meshopt_computeClusterBounds(&indices[0], indices.size(), mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride);
	clodBounds result;
	result.center[0] = bounds.center[0];
	result.center[1] = bounds.center[1];
	result.center[2] = bounds.center[2];
	result.radius = bounds.radius;
	result.error = error;
	return result;
}

static clodBounds boundsMerge(const std::vector<Cluster>& clusters, const std::vector<int>& group)
{
	// 优化：避免不必要的内存分配，直接使用固定大小数组
	clodBounds result = {};
	
	// 如果只有一个元素，直接返回
	if (group.size() == 1) {
		result = clusters[group[0]].bounds;
		return result;
	}
	
	// 优化：使用临时缓冲区合并包围球
	float centers[64 * 3]; // 假设最大64个簇
	float radii[64];
	size_t count = std::min(group.size(), size_t(64));
	
	for (size_t j = 0; j < count; ++j) {
		const clodBounds& b = clusters[group[j]].bounds;
		centers[j * 3 + 0] = b.center[0];
		centers[j * 3 + 1] = b.center[1];
		centers[j * 3 + 2] = b.center[2];
		radii[j] = b.radius;
		result.error = std::max(result.error, b.error);
	}
	
	meshopt_Bounds merged = meshopt_computeSphereBounds(centers, count, sizeof(float) * 3, radii, sizeof(float));
	result.center[0] = merged.center[0];
	result.center[1] = merged.center[1];
	result.center[2] = merged.center[2];
	result.radius = merged.radius;

	return result;
}
static std::vector<Cluster> clusterize(const clodConfig& config, const clodMesh& mesh, const unsigned int* indices, size_t index_count)
{
	// 1. 预估：根据提供的最大顶点/三角形数量，计算理论上最多会生成多少个集群(Meshlets)
	size_t max_meshlets = meshopt_buildMeshletsBound(index_count, config.max_vertices, config.min_triangles);
	// 2. 预分配第三方库所需的内存结构
	std::vector<meshopt_Meshlet> meshlets(max_meshlets);// 存放生成的集群元数据
	std::vector<unsigned int> meshlet_vertices(index_count);// 存放去重后的局部顶点索引

#if MESHOPTIMIZER_VERSION < 1000//旧版本 meshoptimizer 的 API 对三角形数组大小的不同要求为了兼容旧版本，暂时多分配一些内存
	std::vector<unsigned char> meshlet_triangles(index_count + max_meshlets * 3); 
#else
	std::vector<unsigned char> meshlet_triangles(index_count);
#endif
	// 3. 核心切分逻辑
	if (config.cluster_spatial)
		// 采用基于“空间(Spatial)”的切分：根据顶点的 3D 物理位置就近切分。包围盒更紧凑，适合视锥剔除。
		meshlets.resize(meshopt_buildMeshletsSpatial(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices, index_count,
		    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
		    config.max_vertices, config.min_triangles, config.max_triangles, config.cluster_fill_weight));
	else
		// 采用基于“拓扑(Flex)”的切分：根据三角形的邻接关系切分。集群内部连通性更好，减面时不易产生破绽。
		meshlets.resize(meshopt_buildMeshletsFlex(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices, index_count,
		    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
		    config.max_vertices, config.min_triangles, config.max_triangles, 0.f, config.cluster_split_factor));
	// 4. 数据转换：将第三方库的数据格式转换为引擎自定义的 Cluster 结构
	// 优化：使用reserve预分配，减少内存重新分配
	std::vector<Cluster> clusters;
	clusters.reserve(meshlets.size());

	for (size_t i = 0; i < meshlets.size(); ++i)
	{
		const meshopt_Meshlet& meshlet = meshlets[i];
		Cluster cluster;
		// 如果开启了优化，对生成的 Cluster 内部顶点和三角形进行重排序，以最大化 GPU 顶点缓存(V-Cache)的命中率
		if (config.optimize_clusters)
			meshopt_optimizeMeshlet(&meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset], meshlet.triangle_count, meshlet.vertex_count);
		// 记录当前集群包含的不重复的顶点数量
		cluster.vertices = meshlet.vertex_count;
		// 优化：一次性分配所需的索引内存
		cluster.indices.resize(meshlet.triangle_count * 3);
		// 遍历当前集群的所有局部三角形索引，结合 meshlet_vertices 查表，还原成整个大网格的全局索引
		for (size_t j = 0; j < meshlet.triangle_count * 3; ++j)
			cluster.indices[j] = meshlet_vertices[meshlet.vertex_offset + meshlet_triangles[meshlet.triangle_offset + j]];
		// 初始化层级关系状态：-1 表示当前还是游离状态，尚未归属任何父节点（Group）
		cluster.group = -1;
		cluster.refined = -1;
		clusters.push_back(std::move(cluster));
	}

	return clusters;// 返回切分好的一批微网格集群
}

static std::vector<std::vector<int> > partition(const clodConfig& config, const clodMesh& mesh, const std::vector<Cluster>& clusters, const std::vector<int>& pending, const std::vector<unsigned int>& remap)
{
	// 如果待处理的集群数 已经少于等于 一个组的容量上限，就不需要复杂算法了，直接全塞进一个组返回
	if (pending.size() <= config.partition_size)
		return {pending};
	// 预分配：准备把所有 pending 集群内部包含的三角形索引全部平铺展开，交给聚类算法
	std::vector<unsigned int> cluster_indices;
	std::vector<unsigned int> cluster_counts(pending.size());// 记录每个集群包含多少个索引
	size_t total_index_count = 0;
	for (size_t i = 0; i < pending.size(); ++i)
		total_index_count += clusters[pending[i]].indices.size();

	cluster_indices.reserve(total_index_count);
	// 遍历待处理队列
	for (size_t i = 0; i < pending.size(); ++i)
	{
		const Cluster& cluster = clusters[pending[i]];
		cluster_counts[i] = unsigned(cluster.indices.size());
		// 这里的 remap 是位置去重表（空间同一点但法线UV不同），打包时把它们当作相同的拓扑点处理，有利于更好地找出集群间的接缝
		for (size_t j = 0; j < cluster.indices.size(); ++j)
			cluster_indices.push_back(remap[cluster.indices[j]]);
	}
	// cluster_part 数组用于接收聚类结果，记录第 i 个 pending 集群被分配到了哪个组ID
	std::vector<unsigned int> cluster_part(pending.size());
	// 调用核心聚类算法：将多个小集群打包成更少的组（比如把 64 个集群聚类成 4 个组）
	size_t partition_count = meshopt_partitionClusters(&cluster_part[0], &cluster_indices[0], cluster_indices.size(), &cluster_counts[0], cluster_counts.size(),
	    config.partition_spatial ? mesh.vertex_positions : NULL, remap.size(), mesh.vertex_positions_stride, config.partition_size);
	// 准备最终输出的二维数组：外层是组(Group)，内层是组成员(Cluster ID)
	std::vector<std::vector<int> > partitions(partition_count);
	for (size_t i = 0; i < partition_count; ++i)
		partitions[i].reserve(config.partition_size + config.partition_size / 3);

	std::vector<unsigned int> partition_remap;
	// 组的空间排序优化：为了提高 GPU 渲染时的空间局部性，根据组的中心点进行空间排序(如 Z-Curve)
	if (config.partition_sort)
	{
		std::vector<float> partition_point(partition_count * 3);// 收集每个组对应的集群的中心点
		for (size_t i = 0; i < pending.size(); ++i)
			memcpy(&partition_point[cluster_part[i] * 3], clusters[pending[i]].bounds.center, sizeof(float) * 3);
		partition_remap.resize(partition_count);
		// 生成排序后的重映射表
		meshopt_spatialSortRemap(partition_remap.data(), partition_point.data(), partition_count, sizeof(float) * 3);
	}
	// 将 pending 中的集群，根据前面算出的分配 ID (并考虑排序映射)，正式塞入 partitions 二维数组中
	for (size_t i = 0; i < pending.size(); ++i)
		partitions[partition_remap.empty() ? cluster_part[i] : partition_remap[cluster_part[i]]].push_back(pending[i]);

	return partitions;// 返回打包好的组
}

static void lockBoundary(std::vector<unsigned char>& locks, const std::vector<std::vector<int> >& groups, const std::vector<Cluster>& clusters, const std::vector<unsigned int>& remap, const unsigned char* vertex_lock)
{
	// 先清除所有顶点旧的组边界锁状态 (第0位和第7位是锁定位)
	for (size_t i = 0; i < locks.size(); ++i)
		locks[i] &= ~((1 << 0) | (1 << 7));
	// 遍历所有的组以及组内的所有集群
	for (size_t i = 0; i < groups.size(); ++i)
	{
		// 第一次遍历：利用位运算巧妙地探测顶点是否在多处被引用
		for (size_t j = 0; j < groups[i].size(); ++j)
		{
			const Cluster& cluster = clusters[groups[i][j]];

			for (size_t k = 0; k < cluster.indices.size(); ++k)
			{
				unsigned int v = cluster.indices[k];
				unsigned int r = remap[v]; // 映射到唯一的物理位置ID
				// 如果这个顶点在其他组也出现了（因为第7位已经被别的组设为1了），这里就会把它的最低位(第0位)置为1，表示发生了跨组共享(即处于边界)
				locks[r] |= locks[r] >> 7;
			}
		}
		// 第二次遍历：把当前组涉及的顶点位置的第7位置为 1，作为“已访问”的标记留给下一个组探测
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
	// 最后将这些跨组边界的临时标记，转换为 meshoptimizer 库识别的 Protect (保护) 状态
	for (size_t i = 0; i < locks.size(); ++i)
	{
		unsigned int r = remap[i];
		// 提取刚才算出的第0位(边界标志)，或者保留用户原本指定的保护状态
		locks[i] = (locks[r] & 1) | (locks[i] & meshopt_SimplifyVertex_Protect);
		// 如果用户本身传入了额外的顶点锁（比如不想修改UV边缘），叠加进去
		if (vertex_lock)
			locks[i] |= vertex_lock[i];
	}
}
struct SloppyVertex
{
	float x, y, z;
	unsigned int id;// 记录原始顶点 ID，方便减面后还原回全局索引
};

static void simplifyFallback(std::vector<unsigned int>& lod, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, size_t target_count, float* error)
{
	std::vector<SloppyVertex> subset(indices.size());
	std::vector<unsigned char> subset_locks(indices.size());
	lod.resize(indices.size());
	// 1. 提取要减面的局部顶点数据
	size_t positions_stride = mesh.vertex_positions_stride / sizeof(float);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		unsigned int v = indices[i];
		assert(v < mesh.vertex_count);
		// 将全局顶点数据拷贝到局部结构体中
		subset[i].x = mesh.vertex_positions[v * positions_stride + 0];
		subset[i].y = mesh.vertex_positions[v * positions_stride + 1];
		subset[i].z = mesh.vertex_positions[v * positions_stride + 2];
		subset[i].id = v;

		subset_locks[i] = locks[v];// 拷贝该顶点的锁定状态
		lod[i] = unsigned(i);// 初始化索引为 0, 1, 2...
	}
	// 2. 调用 meshoptimizer 的 Sloppy 减面算法
	// 它基于空间聚类而非边坍缩(Edge Collapse)，所以不保证拓扑连续，但能强行减到目标面数
	lod.resize(meshopt_simplifySloppy(&lod[0], &lod[0], lod.size(), &subset[0].x, subset.size(), sizeof(SloppyVertex), subset_locks.data(), target_count, FLT_MAX, error));
	// 调整误差比例，因为 Sloppy 算法返回的误差可能需要根据模型缩放重新计算
	*error *= meshopt_simplifyScale(&subset[0].x, subset.size(), sizeof(SloppyVertex));
	// 3. 将局部索引重新映射回大网格的全局顶点 ID
	for (size_t i = 0; i < lod.size(); ++i)
		lod[i] = subset[lod[i]].id;
}

static std::vector<unsigned int> simplify(const clodConfig& config, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, size_t target_count, float* error)
{
	// 如果目标面数比当前还多，不需要减面，直接返回
	if (target_count > indices.size())
		return indices;

	std::vector<unsigned int> lod(indices.size());
	// 设定减面选项：允许产生非流形(Sparse)，输出绝对误差，根据配置开启宽容模式或正则化
	unsigned int options = meshopt_SimplifySparse | meshopt_SimplifyErrorAbsolute | (config.simplify_permissive ? meshopt_SimplifyPermissive : 0) | (config.simplify_regularize ? meshopt_SimplifyRegularize : 0);
	// 尝试一：带属性保护的高质量减面 (考虑 UV/法线 等 attributes)
	lod.resize(meshopt_simplifyWithAttributes(&lod[0], &indices[0], indices.size(),
	    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
	    mesh.vertex_attributes, mesh.vertex_attributes_stride, mesh.attribute_weights, mesh.attribute_count,
	    &locks[0], target_count, FLT_MAX, options, error));
	// 尝试二：如果尝试一失败了（比如面数没降下来），且配置允许 fallback，开启 Permissive 模式再试一次
	// Permissive 模式允许破坏部分拓扑以换取更高的减面率
	if (lod.size() > target_count && config.simplify_fallback_permissive && !config.simplify_permissive)
		lod.resize(meshopt_simplifyWithAttributes(&lod[0], &indices[0], indices.size(),
		    mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride,
		    mesh.vertex_attributes, mesh.vertex_attributes_stride, mesh.attribute_weights, mesh.attribute_count,
		    &locks[0], target_count, FLT_MAX, options | meshopt_SimplifyPermissive, error));
	// 尝试三：如果前面还是失败，启用最后的杀手锏 —— Sloppy 算法
	if (lod.size() > target_count && config.simplify_fallback_sloppy)
	{
		simplifyFallback(lod, mesh, indices, locks, target_count, error);
		// 惩罚 Sloppy 算法产生的误差，乘以一个系数（比如 2.0），促使渲染器更早地切换回精细模型
		*error *= config.simplify_error_factor_sloppy; 
	}
	// 误差修正：限制最大误差不能超过最长边的一定比例，防止极长细三角形导致的计算异常
	if (config.simplify_error_edge_limit > 0)
	{
		float max_edge_sq = 0;
		// 遍历所有保留下来的三角形，找出最长的边
		for (size_t i = 0; i < indices.size(); i += 3)
		{
			unsigned int a = indices[i + 0], b = indices[i + 1], c = indices[i + 2];
			assert(a < mesh.vertex_count && b < mesh.vertex_count && c < mesh.vertex_count);

			const float* va = &mesh.vertex_positions[a * (mesh.vertex_positions_stride / sizeof(float))];
			const float* vb = &mesh.vertex_positions[b * (mesh.vertex_positions_stride / sizeof(float))];
			const float* vc = &mesh.vertex_positions[c * (mesh.vertex_positions_stride / sizeof(float))];
			float eab = (va[0] - vb[0]) * (va[0] - vb[0]) + (va[1] - vb[1]) * (va[1] - vb[1]) + (va[2] - vb[2]) * (va[2] - vb[2]);
			float eac = (va[0] - vc[0]) * (va[0] - vc[0]) + (va[1] - vc[1]) * (va[1] - vc[1]) + (va[2] - vc[2]) * (va[2] - vc[2]);
			float ebc = (vb[0] - vc[0]) * (vb[0] - vc[0]) + (vb[1] - vc[1]) * (vb[1] - vc[1]) + (vb[2] - vc[2]) * (vb[2] - vc[2]);

			float emax = std::max(std::max(eab, eac), ebc);
			float emin = std::min(std::min(eab, eac), ebc);
			max_edge_sq = std::max(max_edge_sq, std::max(emin, emax / 4));
		}
		// 将最终误差卡在这个最大限制之内
		*error = std::min(*error, sqrtf(max_edge_sq) * config.simplify_error_edge_limit);
	}

	return lod;
}

static int outputGroup(const clodConfig& config, const clodMesh& mesh, const std::vector<Cluster>& clusters, const std::vector<int>& group, const clodBounds& simplified, int depth, void* output_context, clodOutput output_callback, size_t task_index, unsigned int thread_index)
{
	// 准备传递给外部的集群数组结构
	std::vector<clodCluster> group_clusters(group.size());

	for (size_t i = 0; i < group.size(); ++i)
	{
		const Cluster& cluster = clusters[group[i]];
		clodCluster& result = group_clusters[i];
		// 记录父节点 ID
		result.refined = cluster.refined;
		// 如果要求优化包围盒，且该节点不是第一层叶子节点，则根据内部索引重新精确计算包围球，否则直接复用
		result.bounds = (config.optimize_bounds && cluster.refined != -1) ? boundsCompute(mesh, cluster.indices, cluster.bounds.error) : cluster.bounds;
		result.indices = cluster.indices.data();
		result.index_count = cluster.indices.size();
		result.vertex_count = cluster.vertices;
	}
	// 触发用户提供的回调函数（例如在 Scene::storeGroup 中接收），将这些数据写入 GPU 显存或磁盘
	return output_callback ? output_callback(output_context, {depth, simplified}, group_clusters.data(), group_clusters.size(), task_index, thread_index) : -1;
}
// 迭代上下文，在多线程环境下承载所有阶段的数据
struct IterationContext
{
	clodConfig config;
	clodMesh   mesh;
	clodOutput output_callback = nullptr;
	std::vector<unsigned char> locks;// 顶点锁
	std::vector<unsigned int>  remap;// 顶点位置重映射

	int depth = 0;// 当前处理的 LOD 层级（从 0 开始）
	std::vector<Cluster> clusters;// 所有的微集群数据
	std::atomic<size_t>  next_cluster = {}; // 线程安全的集群分配计数器
	std::vector<std::vector<int>> groups;// 当前层级的所有组

	std::vector<int>    pending;// 待处理（下一轮）的集群队列
	std::atomic<size_t> next_pending = {}; // 线程安全的 pending 分配计数器
};
}
// ==========================================
// 默认光栅化/Mesh Shader 的配置参数
// ==========================================
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
	config.partition_spatial = true;// 分组时基于空间位置
	config.partition_size = 16;// 一个组大约包含 16 个 Cluster

	config.cluster_spatial = false;// 切分 Cluster 时基于拓扑连接
	config.cluster_split_factor = 2.0f;

	config.optimize_clusters = true;// 开启 Vertex Cache 优化
	//config.simplify_ratio = 0.15f;
	//config.simplify_threshold = 0.15f;
	config.simplify_ratio = 0.5f;// 每往上一层，期望面数减半
	config.simplify_threshold = 0.85f;// 如果减面后多边形数仍大于 85%，视为减面失败，直接设为叶子

	config.simplify_error_merge_previous = 1.0f; // 继承子节点误差的权重
	config.simplify_error_factor_sloppy = 2.0f;  // sloppy 算法误差的惩罚因子
	config.simplify_permissive = true; // 允许破坏非关键拓扑以换取减面率
	config.simplify_fallback_permissive = true;
	config.simplify_fallback_sloppy = true; // 允许使用后备 sloppy 算法

	return config;
}

// ==============================================================================
// 1. 单个迭代任务：负责对一个“组(Group)”内的集群进行合并、减面、并重新切分
// ==============================================================================
void clodBuild_iterationTask(void* iteration_context, void* output_context, size_t i, unsigned int thread_index)
{
	using namespace clod;

	// 获取迭代上下文，这里面存了所有的全局状态和数据
	IterationContext& context = *(IterationContext*)iteration_context;
	// groups: 当前层级划分出的所有“组”，groups[i] 表示当前线程要处理的第 i 个组
	const std::vector<std::vector<int>>& groups = context.groups;
	// clusters: 全局的集群列表
	std::vector<Cluster>& clusters = context.clusters;
	// locks: 顶点锁状态，用于在减面时保护边界不被破坏
	const std::vector<unsigned char>& locks = context.locks;
	// mesh: 原始网格信息（顶点位置、属性等）
	const clodMesh& mesh = context.mesh;
	// config: LOD构建的配置参数（最大顶点数、减面率等）
	const clodConfig& config = context.config;
	// depth: 当前在 LOD DAG 树中的深度（层级）
	int                                  depth = context.depth;

	// merged: 准备存放当前组内所有集群合并后的三角形索引
	std::vector<unsigned int> merged;
	// 预分配内存：组内集群数 * 每个集群最大三角形数 * 3(顶点数)
	merged.reserve(groups[i].size() * config.max_triangles * 3);

	// 遍历当前组内的所有集群，把它们的索引全部拼接到 merged 数组中
	for (size_t j = 0; j < groups[i].size(); ++j)
		merged.insert(merged.end(), clusters[groups[i][j]].indices.begin(), clusters[groups[i][j]].indices.end());

	// 计算减面后的目标索引数量：当前总索引数 / 3 (得到三角形数) * 减面率(如0.5) * 3
	size_t target_size = size_t((merged.size() / 3) * config.simplify_ratio) * 3;

	// 计算当前组合并后的整体包围球和误差边界
	clodBounds bounds = boundsMerge(clusters, groups[i]);

	float error = 0.f; // 用于接收本次减面产生的几何误差

	// 核心减面操作：使用 meshoptimizer 库对合并后的网格进行简化，输出简化后的索引
	std::vector<unsigned int> simplified = simplify(config, mesh, merged, locks, target_size, &error);

	// 如果简化后的网格大小 仍然大于 设定的阈值（比如减面失败或被锁定的顶点太多）
	if (simplified.size() > merged.size() * config.simplify_threshold)
	{
		// 将该节点的误差设为最大值，意味着它无法再被简化，必须作为最终的根节点或叶子保留
		bounds.error = FLT_MAX;
		// 强行输出当前组（不再继续向上一级生成LOD），结束该分支的迭代
		outputGroup(config, mesh, clusters, groups[i], bounds, depth, output_context, context.output_callback, i, thread_index);
		return;
	}

	// 累加误差：当前节点的误差 = max(子节点误差 * 乘数, 本次减面误差) + 本次减面误差 * 累加系数
	bounds.error = std::max(bounds.error * config.simplify_error_merge_previous, error) + error * config.simplify_error_merge_additive;

	// 将当前（减面前的）精细网格组通过回调输出到外部存储，并返回一个 refined ID（父节点引用）
	int refined = outputGroup(config, mesh, clusters, groups[i], bounds, depth, output_context, context.output_callback, i, thread_index);

	// 清空当前组内旧集群的索引数据，因为它们已经被合并输出，不需要留在内存中了
	for (size_t j = 0; j < groups[i].size(); ++j)
		clusters[groups[i][j]].indices = std::vector<unsigned int>();

	// 将减面后（简化过）的网格数据，重新切分为若干个新的（更粗糙的）Cluster
	std::vector<Cluster> split = clusterize(config, mesh, simplified.data(), simplified.size());

	// 原子操作：在全局集群列表和待处理列表中为新拆分出的集群分配索引位置
	size_t cluster_index = context.next_cluster.fetch_add(split.size());
	size_t pending_index = context.next_pending.fetch_add(split.size());

	// 遍历新生成的粗糙集群
	for (Cluster& cluster : split)
	{
		// 记录它们是由哪个精细组简化而来的（建立 DAG 的父子拓扑边）
		cluster.refined = refined;

		// 继承刚才计算出的误差和包围球
		cluster.bounds = bounds;

		assert(pending_index < context.pending.size());
		assert(cluster_index < context.clusters.size());

		// 将新集群的索引放入 pending 队列，供下一轮（更粗一层）迭代使用
		context.pending[pending_index++] = int(cluster_index);
		// 存入全局集群数组
		context.clusters[cluster_index++] = std::move(cluster);
	}
}


// ==============================================================================
// 2. 核心构建主循环：负责整个 LOD DAG 树的初始化和逐层向上构建
// ==============================================================================
size_t clodBuild(clodConfig config, clodMesh mesh, void* output_context, clodOutput output_callback, clodIteration iteration_callback)
{
	using namespace clod;

	// 基础断言：检查顶点属性步长和保护掩码的合法性
	assert(mesh.vertex_attributes_stride % sizeof(float) == 0);
	assert(mesh.attribute_count * sizeof(float) <= mesh.vertex_attributes_stride);
	assert(mesh.attribute_protect_mask < (1u << (mesh.vertex_attributes_stride / sizeof(float))));

	IterationContext context; // 创建迭代上下文
	context.config = config;
	context.mesh = mesh;
	context.output_callback = output_callback;
	context.locks.resize(mesh.vertex_count); // 初始化顶点锁数组
	context.remap.resize(mesh.vertex_count); // 初始化顶点重映射数组

	// 生成顶点位置重映射表（合并空间位置完全相同的重复顶点）
	meshopt_generatePositionRemap(&context.remap[0], mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride);

	// 如果配置了属性保护掩码（例如保护 UV 边界或法线硬边不被错误减面）
	if (mesh.attribute_protect_mask)
	{
		size_t max_attributes = mesh.vertex_attributes_stride / sizeof(float);

		for (size_t i = 0; i < mesh.vertex_count; ++i)
		{
			unsigned int r = context.remap[i];
			for (size_t j = 0; j < max_attributes; ++j)
				// 如果两个顶点空间位置相同，但受保护的属性（如UV）不同，则锁定该顶点（标记为边界）
				if (r != i && (mesh.attribute_protect_mask & (1u << j)) && mesh.vertex_attributes[i * max_attributes + j] != mesh.vertex_attributes[r * max_attributes + j])
					context.locks[i] |= meshopt_SimplifyVertex_Protect;
		}
	}

	// 第 1 步：初始网格切分，将庞大的原始网格切分成第一批（LOD 0）的叶子集群
	context.clusters = clusterize(config, mesh, mesh.indices, mesh.index_count);
	context.next_cluster = context.clusters.size();

	// 为每个初始集群计算基础包围球，初始误差为 0
	for (Cluster& cluster : context.clusters)
		cluster.bounds = boundsCompute(mesh, cluster.indices, 0.f);

	// 初始化待处理队列（pending），把所有初始集群的索引放进去
	context.pending.resize(context.clusters.size());
	for (size_t i = 0; i < context.clusters.size(); ++i)
		context.pending[i] = int(i);

	// 第 2 步：自底向上的 DAG 构建主循环（只要待处理的集群数 > 1，就继续合并减面）
	while (context.pending.size() > 1)
	{
		// 2.1 将当前层级的离散集群打包成“组（Group）”
		context.groups = partition(config, mesh, context.clusters, context.pending, context.remap);

		// 2.2 为即将生成的新集群和新 pending 元素扩容，避免动态分配
		context.clusters.resize(context.clusters.size() + context.pending.size() + context.groups.size());
		context.pending.resize(context.pending.size() + context.groups.size());
		context.next_pending = 0;

		// 2.3 锁定“组与组”之间的边界顶点，防止独立减面后产生网格裂缝（Cracks）
		lockBoundary(context.locks, context.groups, context.clusters, context.remap, context.mesh.vertex_lock);

		// 2.4 执行组的迭代任务（多线程或单线程）
		if (iteration_callback)
		{
			// 如果提供了外部的并行调度回调，则交由外部多线程执行 clodBuild_iterationTask
			iteration_callback(&context, output_context, context.depth, context.groups.size());
		}
		else
		{
			// 否则在当前线程单线程同步执行所有组的减面和再切分
			for (size_t i = 0; i < context.groups.size(); ++i)
			{
				clodBuild_iterationTask(&context, output_context, i, 0);
			}
		}

		// 2.5 裁剪掉多余的预分配空间，更新当前 pending 队列为新生成的更粗糙的集群
		context.pending.resize(context.next_pending);
		context.clusters.resize(context.next_cluster);

		context.depth++; // LOD 层级加 1，进入下一轮循环（更粗的 LOD）
	}

	// 第 3 步：处理根节点
	if (context.pending.size())
	{
        /////////////////////////////////////////////////////////////////////////
        // 移除了最高级别LOD必须只有一个cluster的强制要求		
		// assert(context.pending.size() == 1); // 最终应该只剩下 1 个最粗糙的集群
		// const Cluster& cluster = context.clusters[context.pending[0]];
		// clodBounds bounds = cluster.bounds;
		//////////////////////////////////////////////////////////////////////////
		// 不再强制要求只有一个cluster，允许最高级别LOD有多个clusters
		clodBounds bounds = boundsMerge(context.clusters, context.pending);
		// 强制将根节点的误差设为无穷大，确保相机极远时也能渲染它
		bounds.error = FLT_MAX;
		// 输出最后的根节点
		outputGroup(config, mesh, context.clusters, context.pending, bounds, context.depth, output_context, output_callback, 0, 0);
	}

	return context.clusters.size(); // 返回生成的集群总数
}


// ==============================================================================
// 3. 提取局部索引 (Local Indices)：将全局大网格索引转换为 Cluster 内部的局部索引
// 配合 Mesh Shader 使用，因为每个 Cluster 最多只有 64/128 个顶点
// ==============================================================================
size_t clodLocalIndices(unsigned int* vertices, unsigned char* triangles, const unsigned int* indices, size_t index_count)
{
	size_t unique = 0; // 记录当前 Cluster 中唯一顶点的数量

	// 优化：使用更大的哈希缓存，减少冲突
	static constexpr size_t CACHE_SIZE = 4096;
	// 使用链表解决冲突的哈希缓存结构
	struct CacheEntry {
		unsigned int vertex_id;
		unsigned short local_index;
		CacheEntry* next;
	};
	
	// 预分配缓存池，避免动态内存分配
	static CacheEntry cachePool[CACHE_SIZE * 2];
	static CacheEntry* cache[CACHE_SIZE];
	static bool cacheInitialized = false;
	
	if (!cacheInitialized) {
		// 初始化缓存头指针
		memset(cache, 0, sizeof(cache));
		cacheInitialized = true;
	}
	
	// 遍历输入的全局索引（通常是一个 Cluster 的全部三角形索引）
	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int v = indices[i]; // 当前全局顶点 ID
		
		// 优化的哈希函数，使用乘法和取模减少冲突
		unsigned int key = (v * 2654435761u) & (CACHE_SIZE - 1);
		
		// 在哈希链表中查找顶点
		CacheEntry* entry = cache[key];
		while (entry != nullptr) {
			if (entry->vertex_id == v) {
				triangles[i] = (unsigned char)entry->local_index;
				goto found;
			}
			entry = entry->next;
		}
		
		// 未找到，为新顶点分配局部索引
		entry = &cachePool[unique];
		entry->vertex_id = v;
		entry->local_index = (unsigned short)unique;
		entry->next = cache[key];
		cache[key] = entry;
		
		triangles[i] = (unsigned char)unique;
		vertices[unique++] = v;
		
found:
		continue;
	}

	assert(unique <= 256); // 保证局部顶点数不超过 256（为了能用 unsigned char 存储三角形索引）
	return unique; // 返回该 Cluster 的实际顶点去重后的数量
}
#endif