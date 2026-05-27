/*
 * File: src/lod/mesh/meshlod_config.h
 * Purpose: Mesh LOD configuration defaults and tuning values.
 */
#pragma once
#include "meshlod_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

clodConfig clodDefaultConfig(size_t max_triangles);

#ifdef __cplusplus
}
#endif
