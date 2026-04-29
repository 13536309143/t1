//核心算法部分。负责将原始网格（Mesh）切分成微小的集群，并利用第三方库 meshoptimizer 简化网格，构建出一套连续的 LOD 层级树。
#include <glm/gtc/constants.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parallel_work.hpp>
#include <meshoptimizer.h>
#include "scene.hpp"
#include "../shaders/attribute_encoding.h"

namespace lodclusters {
   // 辅助模板函数：用于在连续内存分配时，填充数组之间的空隙（补零对齐）
   template <typename T0, typename T1>
   void padZeroes(std::span<T0>& previous, T1* next)
   {
     {
       // 计算前一个数据块末尾到下一个数据块起始指针之间的字节差
       size_t padSize = size_t(next) - size_t(previous.data() + previous.size());
       if(padSize)// 如果有间隙
       {
           // 用 0 填充这段内存间隙
         memset(previous.data() + previous.size(), 0, padSize);
       }
     }
   }
   // 核心函数：将单个簇组（Group，包含多个Cluster）的数据打包并存入全局内存
   uint32_t Scene::storeGroup(TempContext* context,        // 临时上下文（包含多线程状态等）
                              uint32_t           threadIndex,    // 当前执行的线程索引
                              uint32_t           groupIndex,     // 当前组的全局索引
                              const clodGroup&   group,          // meshopt 生成的组数据（包含误差、包围球等）
                              uint32_t           clusterCount,   // 该组包含的集群（Cluster）数量
                              const clodCluster* clusters)       // 集群数组指针
   {
       ProcessingInfo& processing = context->processingInfo; // 获取处理进度和统计信息
       GeometryStorage& geometry = context->geometry;       // 获取全局的几何数据存储对象
       Scene::GroupInfo groupInfo = {};                      // 初始化当前组的元数据信息
     uint32_t level = uint32_t(group.depth);// 获取当前组所在的 LOD 层级深度
     // 获取当前线程专属的临时内存块（避免多线程内存分配冲突）
     uint8_t* groupTempData = &context->threadGroupDatas[context->threadGroupSize * threadIndex];
     Scene::GroupInfo groupTempInfo = context->threadGroupInfo;
     GroupStorage     groupTempStorage(groupTempData, groupTempInfo);// 映射临时内存为 GroupStorage 结构
     // 设置用于顶点去重的快速缓存（大小为 256）
     std::span<uint32_t> vertexCacheEarlyValue((uint32_t*)(groupTempData + context->threadGroupStorageSize), 256);
     std::span<uint32_t> vertexCacheEarlyPos((uint32_t*)vertexCacheEarlyValue.data() + 256, 256);
     // 本地顶点缓存，容量为 组内最大集群数 * 每个集群最大顶点数
     std::span<uint32_t> vertexCacheLocal(vertexCacheEarlyPos.data() + 256, m_config.clusterGroupSize * m_config.clusterVertices);
     uint32_t       clusterMaxVerticesCount = 0; // 记录本组内包含顶点最多的那个集群的顶点数
     uint32_t       clusterMaxTrianglesCount = 0; // 记录本组内包含三角形最多的那个集群的三角形数
     // 初始化组的全局包围盒（初始为反向极值）
     shaderio::BBox groupBbox                = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0, 0};
     {
         // 以下是统计当前组内总数据的偏移量游标
       uint32_t triangleOffset = 0;
       uint32_t vertexOffset   = 0;
       uint32_t vertexDataOffset = 0;// 运行时顶点数据的字节偏移量
       // 用于统计的字节数记录
       size_t vertexPosBytes      = 0;
       size_t vertexNrmBytes      = 0;
       size_t vertexTexCoordBytes = 0;
       // 计算原始全局顶点属性数组的步长
       uint32_t attributeStride = uint32_t(geometry.vertexAttributes.size() / geometry.vertexPositions.size());
       // 遍历当前组内的每一个集群（Cluster）
       for(uint32_t c = 0; c < clusterCount; c++)
       {
         uint32_t* localVertices = &vertexCacheLocal[vertexOffset];// 当前集群的本地顶点指针
         const clodCluster& tempCluster = clusters[c];// 获取原始集群数据
         shaderio::Cluster& groupCluster  = groupTempStorage.clusters[c];// 目标集群结构
         uint32_t           triangleCount = uint32_t(tempCluster.index_count / 3);// 计算三角形数量
         uint32_t           vertexCount   = 0;// 当前集群的独立顶点数统计
         groupCluster.vertices = vertexDataOffset;// 记录该集群顶点数据在组内的起始偏移
         groupCluster.indices  = triangleOffset * 3;// 记录该集群索引数据在组内的起始偏移
         // 初始化快速缓存为全 1（-1）
         memset(vertexCacheEarlyValue.data(), ~0, vertexCacheEarlyValue.size_bytes());
         // 遍历集群的所有索引，进行顶点去重（Deduplication）
         for(uint32_t i = 0; i < tempCluster.index_count; i++)
         {
           uint32_t vertexIndex = tempCluster.indices[i];// 全局顶点索引
           uint32_t cacheIndex  = ~0;
           // 尝试从快速哈希缓存中查找该顶点是否已存在
           uint32_t cacheEarlyValue = vertexCacheEarlyValue[vertexIndex & 0xFF];
           if(cacheEarlyValue == vertexIndex)
           {
             cacheIndex = vertexCacheEarlyPos[vertexIndex & 0xFF];// 命中缓存
           }
           else
           {
               // 未命中，进行线性遍历查找
             for(uint32_t v = 0; v < vertexCount; v++)
             {
               if(localVertices[v] == vertexIndex)
               {
                 cacheIndex = v;// 找到了
               }
             }
           }
           // 如果是个全新的顶点
           if(cacheIndex == ~0)
           {
               cacheIndex = vertexCount++; // 分配新的本地索引
               localVertices[cacheIndex] = vertexIndex;   // 存入本地顶点表
               vertexCacheEarlyValue[vertexIndex & 0xFF] = vertexIndex;   // 更新快速缓存
               vertexCacheEarlyPos[vertexIndex & 0xFF] = cacheIndex;// 更新快速缓存
           }
           // 将去重后的本地索引存入临时存储中
           groupTempStorage.indices[i + triangleOffset * 3] = uint8_t(cacheIndex);
         }
         // 初始化当前集群的包围盒
         shaderio::BBox bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, FLT_MAX, -FLT_MAX};
         // 处理顶点位置
         {
           // in compression case we pack the attributes later
           if(m_config.useCompressedData)// 如果开启了顶点压缩
           {
             for(uint32_t v = 0; v < vertexCount; v++)
             {
               glm::vec3 pos = geometry.vertexPositions[localVertices[v]];
               // 仅计算包围盒，压缩打包将在稍后统一步骤进行
               bbox.lo = glm::min(bbox.lo, glm::vec3(pos));
               bbox.hi = glm::max(bbox.hi, glm::vec3(pos));
             }
           }
           else// 如果不压缩
           {
             for(uint32_t v = 0; v < vertexCount; v++)
             {
               // copy position
               // 直接将位置数据拷贝到临时存储中
               glm::vec3 pos = geometry.vertexPositions[localVertices[v]];
               *(glm::vec3*)&groupTempStorage.vertices[vertexDataOffset + v * 3] = pos;
               // 更新包围盒
               bbox.lo = glm::min(bbox.lo, glm::vec3(pos));
               bbox.hi = glm::max(bbox.hi, glm::vec3(pos));
             }
           }
   
           vertexDataOffset += vertexCount * 3;// 更新数据偏移 (3 个 float)
           vertexPosBytes += sizeof(float) * 3 * vertexCount;
         }
   
         if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL)
         {
           if(!m_config.useCompressedData)
           {
             if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT)
             {
               for(uint32_t v = 0; v < vertexCount; v++)
               {
                 // 处理法线和切线数据
                 glm::vec3 normal = *(const glm::vec3*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeNormalOffset]);
                 glm::vec4 tangent = *(const glm::vec4*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeTangentOffset]);
                 // 将法线和切线打包进一个 32位 整数中（节省显存带宽）
                 uint32_t encoded = shaderio::normal_pack(normal);
                 encoded |= shaderio::tangent_pack(normal, tangent) << ATTRENC_NORMAL_BITS;
                 *(uint32_t*)&groupTempStorage.vertices[vertexDataOffset + v] = encoded;
               }
             }
             else// 仅有法线
             {
               for(uint32_t v = 0; v < vertexCount; v++)
               {
                 glm::vec3 tmp =
                     *(const glm::vec3*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeNormalOffset]);
                 uint32_t encoded= shaderio::normal_pack(tmp);// 仅打包法线
                 *(uint32_t*)&groupTempStorage.vertices[vertexDataOffset + v] = encoded;
               }
             }
           }
           vertexDataOffset += vertexCount;
           vertexNrmBytes += sizeof(uint32_t) * vertexCount;// 更新数据偏移 (1 个 uint32)
         }
         // 处理纹理坐标 (UV0 和 UV1)
         for(uint32_t t = 0; t < 2; t++)
         {
           shaderio::ClusterAttributeBits usedBit =
               t == 0 ? shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0 : shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1;
           uint32_t attributeTexOffset = t == 0 ? geometry.attributeTex0offset : geometry.attributeTex1offset;
           if(geometry.attributeBits & usedBit)
           {
             // 内存对齐到 vec2 的边界
             vertexDataOffset = (vertexDataOffset + 1) & ~1;
             if(!m_config.useCompressedData)
             {
               for(uint32_t v = 0; v < vertexCount; v++)
               {
                 glm::vec2 tmp =*(const glm::vec2*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + attributeTexOffset]);
                 // 存入 UV 数据
                 *(glm::vec2*)&groupTempStorage.vertices[vertexDataOffset + v * 2] = tmp;
               }
             }
             vertexDataOffset += vertexCount * 2;// 更新偏移 (2 个 float)
             vertexTexCoordBytes += sizeof(float) * 2 * vertexCount;
           }
         }
         // 计算该集群三角形中的最短边和最长边（用于后续的 LOD 误差判定或剔除）
         for(uint32_t t = 0; t < triangleCount; t++)
         {
           glm::vec3 trianglePositions[3];
           for(uint32_t v = 0; v < 3; v++)
           {
             // 提取三角形三个顶点的位置
             trianglePositions[v] =geometry.vertexPositions[localVertices[groupTempStorage.indices[(triangleOffset + t) * 3 + v]]];
           }
           for(uint32_t e = 0; e < 3; e++)
           {
             // 计算三条边的长度并更新极值
             float distance    = glm::distance(trianglePositions[e], trianglePositions[(e + 1) % 3]);
             bbox.shortestEdge = std::min(bbox.shortestEdge, distance);
             bbox.longestEdge  = std::max(bbox.longestEdge, distance);
           }
         }
         // 将集群的包围盒合并到所在组（Group）的包围盒中
         groupBbox.lo = glm::min(groupBbox.lo, bbox.lo);
         groupBbox.hi = glm::max(groupBbox.hi, bbox.hi);
         // 保存该集群的属性和元数据
         groupTempStorage.clusterBboxes[c]           = bbox;
         groupTempStorage.clusterGeneratingGroups[c] = uint32_t(tempCluster.refined); // 记录它是从哪个高精度组简化而来的（对于连续LOD的依赖图 DAG 至关重要）
         groupCluster.triangleCountMinusOne = uint8_t(triangleCount - 1); // GPU端习惯存 -1 的值来支持 0~256 映射在 8bit 里
         groupCluster.vertexCountMinusOne   = uint8_t(vertexCount - 1);
         groupCluster.lodLevel              = uint8_t(level);
         groupCluster.groupChildIndex       = uint8_t(c);
         groupCluster.attributeBits         = uint8_t(geometry.attributeBits);
         groupCluster.localMaterialID       = uint8_t(0);
         groupCluster.reserved              = 0;
         // 更新极值统计
         clusterMaxTrianglesCount = std::max(clusterMaxTrianglesCount, triangleCount);
         clusterMaxVerticesCount  = std::max(clusterMaxVerticesCount, vertexCount);
         // 原子的更新应用程序的全局直方图（用于 UI 显示统计）
         ((std::atomic_uint32_t&)m_histograms.clusterTriangles[triangleCount])++;
         ((std::atomic_uint32_t&)m_histograms.clusterVertices[vertexCount])++;
         vertexOffset += vertexCount;
         triangleOffset += triangleCount;
       }// -- 集群遍历结束
       // 填充当前组的头部信息
       groupInfo.offsetBytes                 = 0;
       groupInfo.reserved1                   = 0;
       groupInfo.clusterCount                = uint8_t(clusterCount);
       groupInfo.triangleCount               = uint16_t(triangleOffset);
       groupInfo.vertexCount                 = uint16_t(vertexOffset);
       groupInfo.lodLevel                    = uint8_t(level);
       groupInfo.attributeBits               = uint8_t(geometry.attributeBits);
       groupInfo.vertexDataCount             = vertexDataOffset;
       groupInfo.uncompressedVertexDataCount = 0;
       groupInfo.uncompressedSizeBytes       = 0;
       groupInfo.sizeBytes                   = groupInfo.computeSize();// 计算总共占用的字节数
       // 统计处理数据
       {
         processing.stats.groups++;
         processing.stats.clusters += clusterCount;
         processing.stats.vertices += vertexOffset;
         processing.stats.groupHeaderBytes += sizeof(shaderio::Group);
         processing.stats.clusterHeaderBytes += sizeof(shaderio::Cluster) * clusterCount;
         processing.stats.clusterBboxBytes += sizeof(shaderio::BBox) * clusterCount;
         processing.stats.clusterGenBytes += sizeof(uint32_t) * clusterCount;
         processing.stats.triangleIndexBytes += sizeof(uint8_t) * triangleOffset * 3;
         processing.stats.vertexPosBytes += vertexPosBytes;
         processing.stats.vertexNrmBytes += vertexNrmBytes;
         processing.stats.vertexTexCoordBytes += vertexTexCoordBytes;
         ((std::atomic_uint32_t&)m_histograms.groupClusters[clusterCount])++;
       }
       // 如果开启了极致压缩，在此处对整个组的数据进行有损或无损压缩
       if(m_config.useCompressedData)
       {
         compressGroup(context, groupTempStorage, groupInfo, vertexCacheLocal.data());
       }
     }
     // ============== 以下阶段将临时数据写入全局最终的容器中 ==============
     // 检查是否需要按特定顺序进入锁（保证输出文件的确定性）
     bool useOrderedLock = groupIndex != ~0 && context->innerThreadingActive;
     if(useOrderedLock)
     {
       // 自旋等待，直到轮到当前组的索引（用于保持多线程下输出顺序与单线程一致）
       while(true)
       {
         if(context->groupIndexOrdered.load() == groupIndex)
         {
           // 占据属于当前组的全局字节偏移量
           groupInfo.offsetBytes = context->groupDataOrdered.fetch_add(groupInfo.sizeBytes);
           // 允许下一个组进入
           context->groupIndexOrdered.store(groupIndex + 1);
           break;
         }
         else
         {
           std::this_thread::yield();// 让出 CPU 等待
         }
       }
     }
     {
       std::lock_guard lock(context->groupMutex);// 全局互斥锁，保护共享数据的写入
       // 扩展全局场景包围盒
       geometry.bbox.lo = glm::min(groupBbox.lo, geometry.bbox.lo);
       geometry.bbox.hi = glm::max(groupBbox.hi, geometry.bbox.hi);
       // 如果发现了新的 LOD 层级，则初始化新层级的元数据
       if(context->lodLevel != uint32_t(group.depth))
       {
         context->lodLevel                  = uint32_t(group.depth);
         const shaderio::LodLevel* previous = group.depth ? &geometry.lodLevels[group.depth - 1] : nullptr;
         shaderio::LodLevel initLevel{};
         // 记录新层级在全局数组中的偏移量
         initLevel.clusterOffset = previous ? previous->clusterOffset + previous->clusterCount : 0;
         initLevel.groupOffset   = previous ? previous->groupOffset + previous->groupCount : 0;
         initLevel.minBoundingSphereRadius = FLT_MAX;
         initLevel.minMaxQuadricError      = FLT_MAX;
         geometry.lodLevels.push_back(initLevel);
       }
       // 累加当前层级的总集群数和组数
       geometry.lodLevels[level].clusterCount += groupInfo.clusterCount;
       geometry.lodLevels[level].groupCount++;
       // 更新此层级下的最小包围球半径和最小二次误差（剔除优化）
       geometry.lodLevels[level].minBoundingSphereRadius =std::min(geometry.lodLevels[level].minBoundingSphereRadius, group.simplified.radius);
       geometry.lodLevels[level].minMaxQuadricError =std::min(geometry.lodLevels[level].minMaxQuadricError, group.simplified.error);
       // 更新全局极值统计
       geometry.clusterMaxTrianglesCount = std::max(clusterMaxTrianglesCount, geometry.clusterMaxTrianglesCount);
       geometry.clusterMaxVerticesCount  = std::max(clusterMaxVerticesCount, geometry.clusterMaxVerticesCount);
       // 若为最高精度层级(level==0)，单独记录
       if(level == 0)
       {
         geometry.hiClustersCount += groupInfo.clusterCount;
         geometry.hiTriangleCount += groupInfo.triangleCount;
         geometry.hiVerticesCount += groupInfo.vertexCount;
       }
       // 记录所有层级总数据
       geometry.totalClustersCount += groupInfo.clusterCount;
       geometry.totalTriangleCount += groupInfo.triangleCount;
       geometry.totalVerticesCount += groupInfo.vertexCount;
       // 分配全局存储空间
       if(useOrderedLock)
       {
         if(geometry.groupData.size()<groupInfo.offsetBytes + groupInfo.sizeBytes)
         {
           geometry.groupData.resize(groupInfo.offsetBytes + groupInfo.sizeBytes);
         }
       }
       else
       {
         // 若没有使用有序锁，直接向后追加
         groupInfo.offsetBytes = geometry.groupData.size();
         geometry.groupData.resize(groupInfo.offsetBytes + groupInfo.sizeBytes);
         if(groupIndex == ~0)
         {
           groupIndex = uint32_t(geometry.groupInfos.size());
           geometry.groupInfos.resize(groupIndex + 1);
         }
       }
       geometry.groupInfos[groupIndex] = groupInfo;// 保存元信息
       {
         // 最终的数据内存拷贝阶段（从临时内存拷贝到几何体的全局向量 groupData 中）
         GroupStorage groupStorage(&geometry.groupData[groupInfo.offsetBytes], groupInfo);
         size_t startAddress = size_t(groupStorage.group);
         groupStorage.group->residentID        = 0;      // GPU 驻留ID，上传到显存时才会被修补
         groupStorage.group->clusterResidentID = 0;
         // 填入用于 GPU 遍历判断的组元数据
         groupStorage.group->lodLevel                             = level;
         groupStorage.group->clusterCount                         = groupInfo.clusterCount;
         groupStorage.group->traversalMetric.boundingSphereX      = group.simplified.center[0];
         groupStorage.group->traversalMetric.boundingSphereY      = group.simplified.center[1];
         groupStorage.group->traversalMetric.boundingSphereZ      = group.simplified.center[2];
         groupStorage.group->traversalMetric.boundingSphereRadius = group.simplified.radius;
         groupStorage.group->traversalMetric.maxQuadricError      = group.simplified.error;
         // 拷贝集群头部数组
         memcpy(groupStorage.clusters.data(), groupTempStorage.clusters.data(), groupStorage.clusters.size_bytes());
         // 修补内部偏移指针（将其转换为相对于起点的局部偏移）
         for(uint32_t c = 0; c < clusterCount; c++)
         {
           shaderio::Cluster& groupCluster = groupStorage.clusters[c];
           if(groupInfo.uncompressedSizeBytes)
           {
             groupCluster.vertices = groupStorage.getClusterLocalOffset(c, groupStorage.vertices.data() + groupCluster.vertices,groupInfo.uncompressedSizeBytes);
             groupCluster.indices = groupStorage.getClusterLocalOffset(c, groupStorage.vertices.data() + groupCluster.indices);
           }
           else
           {
             groupCluster.vertices = groupStorage.getClusterLocalOffset(c, groupStorage.vertices.data() + groupCluster.vertices);
             groupCluster.indices = groupStorage.getClusterLocalOffset(c, groupStorage.indices.data() + groupCluster.indices);
           }
         }
         // 拷贝依赖图信息（该集群由哪些组生成）
         memcpy(groupStorage.clusterGeneratingGroups.data(), groupTempStorage.clusterGeneratingGroups.data(),groupStorage.clusterGeneratingGroups.size_bytes());
         padZeroes(groupStorage.clusterGeneratingGroups, groupStorage.clusterBboxes.data());
         // 拷贝包围盒、索引和顶点数据
         memcpy(groupStorage.clusterBboxes.data(), groupTempStorage.clusterBboxes.data(), groupStorage.clusterBboxes.size_bytes());
         memcpy(groupStorage.indices.data(), groupTempStorage.indices.data(), groupStorage.indices.size_bytes());
         padZeroes(groupStorage.indices, groupStorage.vertices.data());
         memcpy(groupStorage.vertices.data(), groupTempStorage.vertices.data(), groupStorage.vertices.size_bytes());
         padZeroes(groupStorage.vertices, (uint32_t*)(groupStorage.raw + groupInfo.sizeBytes));
       }
     }
     return groupIndex;
   }
   //当 meshoptimizer 在进行并行 LOD 生成时，会通过这两个回调函数触发上述存储逻辑。
   //在进行同层级多线程处理时，meshoptimizer 调用的回调函数。
   //任务数(task_count)代表当前 LOD 深度需要处理的组数。
   void Scene::clodIterationMeshoptimizer(void* intermediate_context, void* output_context, int depth, size_t task_count)
   {
     TempContext*     context  = reinterpret_cast<TempContext*>(output_context);
     GeometryStorage& geometry = context->geometry;
     // 预分配当前层级所需的所有组信息的空间
     context->levelGroupOffset      = geometry.groupInfos.size();
     context->levelGroupOffsetValid = true;
     geometry.groupInfos.resize(context->levelGroupOffset + task_count);
   
     // 利用 nvutils 的并行库，调度多线程任务执行单次迭代的构建
     nvutils::parallel_batches_pooled<1>(task_count,[&](uint64_t idx, uint32_t threadInnerIdx) 
         {
           clodBuild_iterationTask(intermediate_context, output_context, idx, threadInnerIdx);
         },
         context->processingInfo.numInnerThreads);
         context->levelGroupOffsetValid = false;
   }
   // 当 meshoptimizer 完成了一个组的简化和聚类后，调用此函数返回结果。
   int Scene::clodGroupMeshoptimizer(void* output_context, clodGroup group, const clodCluster* clusters, size_t cluster_count, size_t task_index, uint32_t thread_index)
   {
     TempContext*     context  = reinterpret_cast<TempContext*>(output_context);
     GeometryStorage& geometry = context->geometry;
     // 计算该组在全局数组中的预期索引
     uint32_t groupIndex =context->innerThreadingActive && context->levelGroupOffsetValid ? uint32_t(context->levelGroupOffset + task_index) : ~0u;
     // 调用上面的 storeGroup 将生成的简化组打包保存
     return context->scene.storeGroup(context, thread_index, groupIndex, group, uint32_t(cluster_count), clusters);
   }
   //此函数负责设置 meshoptimizer 的相关参数，并触发对输入网格的切分和网格简化（Decimation）流程。
   void Scene::buildGeometryLod(ProcessingInfo& processingInfo, GeometryStorage& geometry)
   {
     // 获取基于当前配置（目标三角形数）的默认 clod 配置
     clodConfig clodInfo = clodDefaultConfig(m_config.clusterTriangles);
     // 覆盖用户自定义配置
     clodInfo.cluster_fill_weight = m_config.meshoptFillWeight;    // 聚类填充权重
     clodInfo.cluster_split_factor = m_config.meshoptSplitFactor;   // 聚类分裂因子
     clodInfo.max_vertices = m_config.clusterVertices;      // 单个集群最大顶点数（例如 64）
     clodInfo.partition_size = m_config.clusterGroupSize;     // 将多少个集群组合成一个"组" (Group，例如 8 个)
     clodInfo.partition_spatial = true;                          // 启用空间划分
     clodInfo.partition_sort = true;                          // 划分后排序
     clodInfo.optimize_clusters = true;                             // 在集群内部对三角形进行顶点缓存优化排序
     // 针对 meshopt_partitionClusters 的特定最坏情况，微调 partition_size，防止越界
     while((clodInfo.partition_size + clodInfo.partition_size / 3) > m_config.clusterGroupSize)
     {
       clodInfo.partition_size--;
     }
     // 这些参数控制在跨越 LOD 层级时（由于是对已经简化的网格再次简化），二次误差（Quadric Error）如何累积和传递
     clodInfo.simplify_error_merge_previous = m_config.lodErrorMergePrevious;
     clodInfo.simplify_error_merge_additive = m_config.lodErrorMergeAdditive;
     clodInfo.simplify_error_edge_limit = m_config.lodErrorEdgeLimit;
     ///////////////////////////////////////
	   //开启lod优化
     clodInfo.curvature_adaptive_strength = m_config.curvatureAdaptiveStrength;
     clodInfo.curvature_window_radius     = m_config.curvatureWindowRadius;
     clodInfo.feature_edge_threshold      = m_config.featureEdgeThreshold;
     clodInfo.perceptual_weight           = m_config.perceptualWeight;
     clodInfo.silhouette_preservation     = m_config.silhouettePreservation;
     ///////////////////////////////////////
     // 封装输入的几何网格指针
     clodMesh inputMesh                = {};
     inputMesh.vertex_positions        = reinterpret_cast<const float*>(geometry.vertexPositions.data());
     inputMesh.vertex_count            = geometry.vertexPositions.size();
     inputMesh.vertex_positions_stride = sizeof(glm::vec3);
     inputMesh.index_count             = geometry.triangles.size() * 3;
     inputMesh.indices                 = reinterpret_cast<const uint32_t*>(geometry.triangles.data());
     float attributeWeights[9] = {};// 顶点属性权重数组
     // 如果模型存在除了位置以外的其他属性（法线、UV等），设置其在网格简化时的权重
     if(geometry.attributesWithWeights)
     {
       // 法线权重，防止简化时法线突变
       if(m_config.simplifyNormalWeight > 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL))
       {
         attributeWeights[geometry.attributeNormalOffset + 0] = m_config.simplifyNormalWeight;
         attributeWeights[geometry.attributeNormalOffset + 1] = m_config.simplifyNormalWeight;
         attributeWeights[geometry.attributeNormalOffset + 2] = m_config.simplifyNormalWeight;
       }
       // UV 权重，防止简化导致贴图严重拉伸
       if(m_config.simplifyTexCoordWeight > 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0))
       {
         attributeWeights[geometry.attributeTex0offset + 0] = m_config.simplifyTexCoordWeight;
         attributeWeights[geometry.attributeTex0offset + 1] = m_config.simplifyTexCoordWeight;
       }
       if(m_config.simplifyTexCoordWeight > 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1))
       {
         attributeWeights[geometry.attributeTex1offset + 0] = m_config.simplifyTexCoordWeight;
         attributeWeights[geometry.attributeTex1offset + 1] = m_config.simplifyTexCoordWeight;
       }
       // 切线权重
       if(m_config.simplifyTangentWeight > 0 && m_config.simplifyTangentSignWeight > 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT))
       {
         attributeWeights[geometry.attributeTangentOffset + 0] = m_config.simplifyTangentWeight;
         attributeWeights[geometry.attributeTangentOffset + 1] = m_config.simplifyTangentWeight;
         attributeWeights[geometry.attributeTangentOffset + 2] = m_config.simplifyTangentWeight;
         attributeWeights[geometry.attributeTangentOffset + 3] = m_config.simplifyTangentSignWeight;
       }
       inputMesh.attribute_count          = geometry.attributesWithWeights;
       inputMesh.vertex_attributes        = geometry.vertexAttributes.data();
       inputMesh.vertex_attributes_stride = sizeof(float) * inputMesh.attribute_count;
       inputMesh.attribute_weights        = attributeWeights;
     }
     TempContext context = {processingInfo, geometry, *this};
     // 估算最坏情况下的组元信息大小，用于分配临时内存
     GroupInfo worstGroup       = {};
     worstGroup.clusterCount    = uint8_t(m_config.clusterGroupSize);
     worstGroup.vertexCount     = uint16_t(m_config.clusterGroupSize * m_config.clusterVertices);
     worstGroup.triangleCount   = uint16_t(m_config.clusterGroupSize * m_config.clusterTriangles);
     worstGroup.attributeBits   = geometry.attributeBits;
     worstGroup.vertexDataCount = worstGroup.estimateVertexDataCount();
     worstGroup.sizeBytes       = worstGroup.computeSize();
     // 根据线程数分配临时内存，保证线程安全
     context.innerThreadingActive   = processingInfo.numInnerThreads > 1;
     context.threadGroupInfo        = worstGroup;
     context.threadGroupStorageSize = uint32_t(worstGroup.computeSize());
     context.threadGroupSize        = nvutils::align_up(context.threadGroupStorageSize, 4) + sizeof(uint32_t) * 256 * 2 + sizeof(uint32_t) * m_config.clusterGroupSize * m_config.clusterVertices;
     context.threadGroupDatas.resize(context.threadGroupSize * processingInfo.numInnerThreads);
     // 根据启发式算法保留足够的 vector 空间，避免动态扩容带来的拷贝开销
     size_t reservedClusters  = (geometry.triangles.size() + m_config.clusterTriangles - 1) / m_config.clusterTriangles;
     size_t reservedGroups    = (reservedClusters + m_config.clusterGroupSize - 1) / m_config.clusterGroupSize;
     size_t reservedTriangles = geometry.triangles.size();
     reservedClusters  = size_t(double(reservedClusters) * 2.0);
     reservedGroups    = size_t(double(reservedGroups) * 3.0);
     reservedTriangles = size_t(double(reservedTriangles) * 2.0);
     size_t reservedData = 0;
     reservedData += sizeof(shaderio::Group) * reservedGroups;
     reservedData += sizeof(shaderio::Cluster) * reservedClusters;
     reservedData += sizeof(shaderio::BBox) * reservedClusters;
     reservedData += sizeof(uint32_t) * reservedClusters;
     reservedData += sizeof(uint8_t) * reservedTriangles;
     reservedData += sizeof(glm::vec3) * reservedClusters * m_config.clusterVertices;
     geometry.groupData.reserve(reservedData);
     geometry.groupInfos.reserve(reservedGroups);
     geometry.lodLevels.reserve(32);
     // 【此处根据用户注释，修改为了单线程】
     // 强制关闭多线程并发迭代，保证 QEM (Quadric Error Metrics) 简化结果绝对一致
     // clodBuild 会进行递归的“生成集群->合并->简化->生成更粗糙的集群”的过程  // 强制关闭多线程并发迭代，保证 QEM 简化结果绝对一致//////////////////////////////////////这里可以控制每次简化的是否一样
     //clodBuild(clodInfo, inputMesh, &context, clodGroupMeshoptimizer,processingInfo.numInnerThreads > 1 ? clodIterationMeshoptimizer : nullptr);
     clodBuild(clodInfo, inputMesh, &context, clodGroupMeshoptimizer, nullptr);//单线程///////////////////////////////////
     // 原始网格数据已经被处理完了，清空以节省系统内存
     geometry.triangles        = {};
     geometry.vertexPositions  = {};
     geometry.vertexAttributes = {};
     geometry.lodLevelsCount = uint32_t(geometry.lodLevels.size());
    // 校验构建结果：最后一级 LOD（最远、最模糊的层级）必须只剩下一个 Group 和一个 Cluster 
    /////////////////////////////////////////////////////////////////////////
    // 移除了最高级别LOD必须只有一个cluster的强制要求
    //  if(geometry.lodLevelsCount)
    //  {
    //    shaderio::LodLevel& lastLodLevel = geometry.lodLevels.back();
    //    if(lastLodLevel.groupCount != 1 || lastLodLevel.clusterCount != 1)
    //    {
    //      assert(0);
    //      LOGE("clodBuild failed: last lod level has more than one cluster\n");
    //      std::exit(-1);
    //    }
    //  }
     ////////////////////////////////////////////////////////////////////////
     // 校验构建结果：不再强制要求最后一级 LOD 只有一个 Group 和一个 Cluster
     // 移除强制要求，允许最后一级 LOD 有多个 groups 和 clusters
     // 这样可以处理那些无法简化为单个 cluster 的复杂模型

     // 压缩多余的 vector 预留内存
     geometry.groupInfos.shrink_to_fit();
     geometry.groupData.shrink_to_fit();
     geometry.lodLevels.shrink_to_fit();
     // 构建用于 GPU 剔除和遍历的 LOD 空间层次包围树
     buildHierarchy(processingInfo, geometry);
     // 计算树节点的最终包围盒边界
     geometry.lodNodeBboxes.resize(geometry.lodNodes.size());
     computeLodBboxes_recursive(geometry, 0);
     ((std::atomic_uint32_t&)m_histograms.lodLevels[geometry.lodLevelsCount])++;
   }
   //buildHierarchy 彻底抛弃了网格的拓扑连接关系，转而只看空间距离，构建了一棵严格的空间包围体树（BVH Tree）
   //buildHierarchy 空间层次结构树构建，生成了所有的组（Groups）后，这个函数会在上面建立一棵基于包围球的树（类似 R - Tree），这样 GPU Compute Shader 在遍历时可以一次性剔除一整片不需要渲染的区域。
   //这棵树构建完成后，将作为一个连续的 1D 数组发送给 GPU。GPU 的 Compute Shader 在渲染时，会从根节点开始遍历这棵树，利用包围球进行视锥体剔除（Frustum Culling）和遮挡剔除（Occlusion Culling），从而一次性剔除大片不可见的几何体，极大降低渲染开销。
   void Scene::buildHierarchy(ProcessingInfo& processingInfo, GeometryStorage& geometry)
   {
     // 1. 预先计算每一层级（LOD Level）需要多少个空间树节点
     // 内存布局规划：[最顶层Root(1个)] -> [各LOD级Root(lodLevelCount个)] -> [各种内部节点与叶子节点]
     uint32_t lodLevelCount = geometry.lodLevelsCount;// 偏移跨过所有 Root 节点
     std::vector<Range> lodNodeRanges(lodLevelCount);
     {
       uint32_t nodeOffset = 1 + lodLevelCount;
       for(uint32_t lodLevel = 0; lodLevel < lodLevelCount; lodLevel++)
       {
         const shaderio::LodLevel& lodLevelInfo = geometry.lodLevels[lodLevel];
         uint32_t nodeCount = lodLevelInfo.groupCount;// 初始节点数 = 该层的组(Group)数，作为叶子
         uint32_t iterationCount = nodeCount;
         // 模拟每次向上聚合（每 preferredNodeWidth 个节点聚成一个父节点），统计总节点数
         while(iterationCount > 1)
         {
           iterationCount = (iterationCount + m_config.preferredNodeWidth - 1) / m_config.preferredNodeWidth;
           nodeCount += iterationCount;
         }
         nodeCount--;// 减去当前层级的根节点（因为已经算在偏移起始区了）
         lodNodeRanges[lodLevel].offset = nodeOffset;
         lodNodeRanges[lodLevel].count  = nodeCount;
         nodeOffset += nodeCount;// 移动游标
       }
       // 根据算出的总数开辟空间，为后续的无锁并行写入打下了基础。
       geometry.lodNodes.resize(nodeOffset);
     }
     // 2. 并行为每一个 LOD 级别构建独立的树
     nvutils::parallel_batches_pooled<1>(
lodLevelCount,[&](uint64_t idx, uint32_t threadInnerIdx) 
     {
       uint32_t                  lodLevel     = uint32_t(idx);
       const shaderio::LodLevel& lodLevelInfo = geometry.lodLevels[lodLevel];
       const Range&              lodNodeRange = lodNodeRanges[lodLevel];
       uint32_t nodeCount      = lodLevelInfo.groupCount;
       uint32_t nodeOffset     = lodNodeRange.offset;
       uint32_t lastNodeOffset = nodeOffset;
       // 阶段 A：先将所有的簇组（Groups）转化为树的最底层叶子节点
       for(uint32_t g = 0; g < nodeCount; g++)
       {
         uint32_t         groupID   = g + lodLevelInfo.groupOffset;
         const GroupInfo& groupInfo = geometry.groupInfos[groupID];
         GroupView        groupView(geometry.groupData, groupInfo);
         // 如果该层只有一个组，它直接作为该层的根节点
         shaderio::Node& node = nodeCount == 1 ? geometry.lodNodes[1 + lodLevel] : geometry.lodNodes[nodeOffset++];
         node                                      = {};
         node.groupRange.isGroup                   = 1;// 标记这是指向真实几何 Group 的叶子节点，这是一个标志位，告诉 GPU 遍历到这里时，已经到达叶子，下面是真实的渲染数据了。
         node.groupRange.groupIndex                = groupID;
         node.groupRange.groupClusterCountMinusOne = groupInfo.clusterCount - 1;
         node.traversalMetric                      = groupView.group->traversalMetric;
       }
       if(nodeCount == 1)// 特例：已写入根区域
       {
         nodeOffset++;
       }
       // 阶段 B：自底向上，逐层构建树的父节点
       uint32_t depth = 0;
       uint32_t iterationCount = nodeCount;// 当前需要聚合的子节点数量
       std::vector<uint32_t> partitionedIndices;
       std::vector<shaderio::Node> oldNodes;
       while(iterationCount > 1)
       {
         uint32_t lastNodeCount = iterationCount;
         shaderio::Node* lastNodes = &geometry.lodNodes[lastNodeOffset];
         // 使用 meshoptimizer 将空间相近的子节点聚类在一起，返回排序索引
         partitionedIndices.resize(lastNodeCount);
         meshopt_spatialClusterPoints(partitionedIndices.data(), &lastNodes->traversalMetric.boundingSphereX,lastNodeCount, sizeof(shaderio::Node), m_config.preferredNodeWidth);//根据节点包围球的中心点坐标进行空间聚类
         {
           // 根据聚类算法返回的索引，对节点数组进行重新排序
           oldNodes.clear();
           oldNodes.insert(oldNodes.end(), lastNodes, lastNodes + lastNodeCount);
           for(uint32_t n = 0; n < lastNodeCount; n++)
           {
             lastNodes[n] = oldNodes[partitionedIndices[n]];
           }
         }
         // 计算新一层父节点的数量
         iterationCount = (lastNodeCount + m_config.preferredNodeWidth - 1) / m_config.preferredNodeWidth;
         // 决定新的父节点写在哪（如果是最后一个父节点，写在各 LOD 级的根区）
         shaderio::Node* newNodes = iterationCount == 1 ? &geometry.lodNodes[1 + lodLevel] : &geometry.lodNodes[nodeOffset];
         // 创建父节点
         for(uint32_t n = 0; n < iterationCount; n++)
         {
           shaderio::Node& node          = newNodes[n];
           shaderio::Node* childrenNodes = &lastNodes[n * m_config.preferredNodeWidth];
           // 处理边界情况下的子节点个数
           uint32_t childCount = std::min((n + 1) * m_config.preferredNodeWidth, lastNodeCount) - n * m_config.preferredNodeWidth;
           node                                 = {};
           node.nodeRange.isGroup               = 0;// 标记这是一个内部节点（包含多个子节点）
           node.nodeRange.childCountMinusOne    = childCount - 1;
           node.nodeRange.childOffset           = lastNodeOffset + n * m_config.preferredNodeWidth;// 指向刚才排序好的子节点数组
           node.traversalMetric.maxQuadricError = 0;
           // 父节点的最大二次误差应等于其所有子节点中最大的那个误差，以确保遍历时的保守性。
           for(uint32_t c = 0; c < childCount; c++)
           {
             node.traversalMetric.maxQuadricError =std::max(node.traversalMetric.maxQuadricError, childrenNodes[c].traversalMetric.maxQuadricError);
           }
           // 计算能够包围所有子节点的超级包围球
           meshopt_Bounds merged = meshopt_computeSphereBounds(&childrenNodes[0].traversalMetric.boundingSphereX, childCount, sizeof(shaderio::Node),&childrenNodes[0].traversalMetric.boundingSphereRadius, sizeof(shaderio::Node));
           node.traversalMetric.boundingSphereX      = merged.center[0];
           node.traversalMetric.boundingSphereY      = merged.center[1];
           node.traversalMetric.boundingSphereZ      = merged.center[2];
           node.traversalMetric.boundingSphereRadius = merged.radius;
         }
         lastNodeOffset = nodeOffset;
         nodeOffset += iterationCount;// 游标推入下一层
         depth++;
       }
       nodeOffset--;
       assert(lodNodeRange.offset + lodNodeRange.count == nodeOffset); // 健全性检查
     },processingInfo.numInnerThreads);
     //重复上述分区和合并过程，直到该 LOD 级别只剩下一个根节点（Lod Root）。

     // 3. 构建最高层级的超级根节点（Top Root）
     // 这个节点包含着所有的 LOD 级别根节点。遍历时从这里出发，能快速跳过整个不被需要的模型。
     {
       // 合并所有 LOD 层级根节点的包围球
       meshopt_Bounds merged =meshopt_computeSphereBounds(&geometry.lodNodes[1].traversalMetric.boundingSphereX, lodLevelCount, sizeof(shaderio::Node),&geometry.lodNodes[1].traversalMetric.boundingSphereRadius, sizeof(shaderio::Node));
       shaderio::Node& node          = geometry.lodNodes[0];
       shaderio::Node* childrenNodes = &geometry.lodNodes[1];// 所有 LOD 的根节点连续排在索引 1 到 lodLevelCount 处
       node                                      = {};
       node.nodeRange.isGroup                    = 0;
       node.nodeRange.childCountMinusOne         = lodLevelCount - 1;
       node.nodeRange.childOffset                = 1;
       node.traversalMetric.boundingSphereX      = merged.center[0];
       node.traversalMetric.boundingSphereY      = merged.center[1];
       node.traversalMetric.boundingSphereZ      = merged.center[2];
       node.traversalMetric.boundingSphereRadius = merged.radius;
       node.traversalMetric.maxQuadricError= 0;
       for(uint32_t c = 0; c < lodLevelCount; c++)
       {
         node.traversalMetric.maxQuadricError =std::max(node.traversalMetric.maxQuadricError, childrenNodes[c].traversalMetric.maxQuadricError);
       }
     }
   }
}
