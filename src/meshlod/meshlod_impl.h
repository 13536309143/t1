//==============================================================================
// 文件：src/meshlod/meshlod_impl.h
// 模块定位：meshlod 内部类型与函数声明，承载构建上下文、临时 簇 表示和算法辅助函数。
// 数据流：输入来自公共 API 的 mesh/config；输出供 build、簇ing 和 simplify 头文件共享内部状态。
// 方法说明：内部实现层把公共 ABI 与算法细节隔离，便于调整简化策略而不改变外部调用契约。
// 正确性约束：内部类型不应泄露到公共 API；跨头文件共享的字段语义必须保持一致。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#pragma once


// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include "meshlod_types.h"
#include <float.h>
#include <math.h>
#include <string.h>
#include <cassert>
#include <algorithm>
#include <vector>
#include <atomic>
#include <unordered_map>
#include <queue>
#include <array>
#include <meshoptimizer.h>

namespace clod
{


// 结构：Cluster。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct Cluster
{
	size_t vertices;
	std::vector<unsigned int> indices;
	int group;
	int refined;
	clodBounds bounds;
};


// 结构：SloppyVertex。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct SloppyVertex
{
	float x, y, z;
	unsigned int id;
};


// 结构：IterationContext。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct IterationContext
{
	clodConfig config;
	clodMesh   mesh;
	clodOutput output_callback = nullptr;
	std::vector<unsigned char> locks;
	std::vector<unsigned int>  remap;
	std::vector<float> feature_importance;
	std::vector<unsigned char> feature_locks;
	std::vector<float> feature_attributes;
	std::vector<float> feature_attribute_weights;

	int depth = 0;
	std::vector<Cluster> clusters;
	std::atomic<size_t>  next_cluster = {};
	std::vector<std::vector<int>> groups;

	std::vector<int>    pending;
	std::atomic<size_t> next_pending = {};
};


// 函数：boundsCompute。计算派生值，供后续剔除、LOD、统计或资源规划使用。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：计算结果通常参与阈值比较或内存规划，数值稳定性和边界条件需要特别注意。
clodBounds boundsCompute(const clodMesh& mesh, const std::vector<unsigned int>& indices, float error);


// 函数：boundsMerge。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
clodBounds boundsMerge(const std::vector<Cluster>& clusters, const std::vector<int>& group);


// 函数：clusterize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
std::vector<Cluster> clusterize(const clodConfig& config, const clodMesh& mesh, const unsigned int* indices, size_t index_count);


// 函数：partition。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
std::vector<std::vector<int> > partition(const clodConfig& config, const clodMesh& mesh, const std::vector<Cluster>& clusters, const std::vector<int>& pending, const std::vector<unsigned int>& remap);


// 函数：lockBoundary。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void lockBoundary(std::vector<unsigned char>& locks, const std::vector<std::vector<int> >& groups, const std::vector<Cluster>& clusters, const std::vector<unsigned int>& remap, const unsigned char* vertex_lock);


// 函数：simplifyFallback。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void simplifyFallback(std::vector<unsigned int>& lod, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, size_t target_count, float* error);


// 函数：simplify。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
std::vector<unsigned int> simplify(const clodConfig& config, const clodMesh& mesh, const std::vector<unsigned int>& indices, const std::vector<unsigned char>& locks, size_t target_count, float* error);


// 函数：outputGroup。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
int outputGroup(const clodConfig& config, const clodMesh& mesh, const std::vector<Cluster>& clusters, const std::vector<int>& group, const clodBounds& simplified, int depth, void* output_context, clodOutput output_callback, size_t task_index, unsigned int thread_index);

}
