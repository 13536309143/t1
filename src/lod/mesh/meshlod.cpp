/*
 * File: src/lod/mesh/meshlod.cpp
 * Purpose: Entry point for mesh LOD generation using bounds, clustering, simplification, build, and local-index helpers.
 */
#include <meshoptimizer.h>
#include "meshlod.h"
#include "meshlod_bounds.h"
#include "meshlod_clustering.h"
#include "meshlod_simplify.h"
#include "meshlod_build.h"
#include "meshlod_local_indices.h"


// #include <meshoptimizer.h>
// #define CLUSTERLOD 1
// #include "lod.h"