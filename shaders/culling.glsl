/*

GPU 端的计算着色器，用于遍历 LOD 树，执行视锥体剔除和遮挡剔除，决定哪些集群最终需要被画出来

主文件，包含所有 culling 模块

*/

// 包含所有 culling 模块
#include "culling_constants.inc"
#include "culling_frustum.inc"
#include "culling_hiz.inc"
#include "culling_raster.inc"
