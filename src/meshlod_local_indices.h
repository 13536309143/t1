#pragma once
#include "meshlod_types.h"
#include <cassert>
#include <cstring>

size_t clodLocalIndices(unsigned int* vertices, unsigned char* triangles, const unsigned int* indices, size_t index_count)
{
	size_t unique = 0;

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

	assert(unique <= 256);
	return unique;
}
