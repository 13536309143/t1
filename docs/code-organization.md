# Code Organization

This project is organized by runtime responsibility rather than by file type.
The CMake target collects source and shader files recursively, so new files can be
added inside the existing module folders without changing the top-level build file.

## C++ Modules

- `src/application`: application entry point, UI, and top-level `LodClusters` orchestration.
- `src/io`: glTF import, processed-scene cache, and binary serialization helpers.
- `src/scene`: scene data model, geometry storage, instances, materials, bounds, and scene-wide statistics.
- `src/scene/processing`: scene processing algorithms that transform imported meshes into compressed clustered LOD data.
- `src/lod/mesh`: standalone mesh LOD generation implementation and supporting headers.
- `src/gpu`: Vulkan infrastructure, resource allocation, shader compilation, Hi-Z, and GPU scene upload helpers.
- `src/rendering`: renderer-facing abstractions, shared draw setup, and the cluster LOD render path.
- `src/streaming`: streaming scene upload, resident data, request/update tasks, and allocator utilities.
- `src/third_party`: single-file third-party integration units that are compiled with the target.

## Shader Modules

- `shaders/shared/interface`: shared CPU/GPU layout headers and shader interface definitions.
- `shaders/shared/common`: reusable shader helpers, culling code, shading helpers, and attribute encoding.
- `shaders/passes/cluster`: primary cluster rendering shaders and software-raster cluster pass.
- `shaders/passes/debug`: cluster and instance bounding-box visualization passes.
- `shaders/passes/fullscreen`: full-screen background, resolve, and Hi-Z generation passes.
- `shaders/compute/streaming`: streaming request, setup, and scene update compute shaders.
- `shaders/compute/traversal`: LOD traversal and traversal pre-sort compute shaders.
- `shaders/compute/build`: indirect draw/dispatch setup compute shaders.

## Build Notes

- `CMakeLists.txt` uses `GLOB_RECURSE` with `CONFIGURE_DEPENDS` for `src` and `shaders`.
- C++ include directories are module-level, so includes stay short and independent of file locations.
- Shader compilation paths include the shader module folder, for example `passes/cluster/cluster_mesh.mesh.glsl`.
- Runtime shader include search paths include every shader subdirectory for both source-tree and install-tree layouts.

## Implementation Split

- Application code is split into constructor/defaults, lifecycle/file loading, scene/render setup, runtime config changes, frame rendering, and UI.
- GPU resource code is split into general resources/commands/shader compilation and framebuffer/Hi-Z render-target management.
- Streaming task code is split into request/residency bookkeeping and allocator/update/transfer task execution.
