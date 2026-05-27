#pragma once
#include "meshlod_types.h"
#include <cassert>

size_t clodLocalIndices(unsigned int* vertices, unsigned char* triangles, const unsigned int* indices, size_t index_count)
{
	size_t unique = 0;

	for (size_t i = 0; i < index_count; ++i)
	{
		unsigned int v = indices[i];
		size_t local = 0;

		for (; local < unique; ++local)
			if (vertices[local] == v)
				break;

		if (local == unique)
			vertices[unique++] = v;

		assert(local < 256);
		triangles[i] = (unsigned char)local;
	}

	return unique;
}
