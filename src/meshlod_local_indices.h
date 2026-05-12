#pragma once
#include "meshlod_types.h"
#include <cassert>
#include <cstring>

size_t clodLocalIndices(unsigned int* vertices, unsigned char* triangles, const unsigned int* indices, size_t index_count)
{
	size_t unique = 0;

	static constexpr size_t CACHE_SIZE = 4096;
	unsigned int cacheValue[CACHE_SIZE];
	unsigned char cacheIndex[CACHE_SIZE];
	memset(cacheValue, 0xff, sizeof(cacheValue));
	
	// 遍历输入的全局索引（通常是一个 Cluster 的全部三角形索引）
	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int v = indices[i]; // 当前全局顶点 ID
		
		unsigned int key = (v * 2654435761u) & (CACHE_SIZE - 1);
		if(cacheValue[key] == v)
		{
			triangles[i] = cacheIndex[key];
			continue;
		}

		for(size_t j = 0; j < unique; ++j)
		{
			if(vertices[j] == v)
			{
				cacheValue[key] = v;
				cacheIndex[key] = (unsigned char)j;
				triangles[i] = (unsigned char)j;
				goto found;
			}
		}

		cacheValue[key] = v;
		cacheIndex[key] = (unsigned char)unique;
		triangles[i] = (unsigned char)unique;
		vertices[unique++] = v;

found:;
	}

	assert(unique <= 256);
	return unique;
}
