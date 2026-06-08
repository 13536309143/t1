//==============================================================================
// 文件：src/scene/scene_cluster_compression.cpp
// 模块定位：簇组 压缩与解压实现，在 CPU 缓存体积、系统内存占用和 GPU 上传带宽之间取得平衡。
// 数据流：输入是 组 内局部顶点、属性和索引；输出是紧凑存储或解压后的 着色器 运行时布局。
// 方法说明：压缩器利用组内局部性和属性量化，把位置、法线、切线、纹理坐标编码成更小的比特流，同时保留解码可验证性。
// 正确性约束：压缩后若不节省空间必须回退到未压缩布局；簇.indices 的临时复用必须在解压阶段恢复真实索引偏移。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <bit>

#include <algorithm>
#include <meshoptimizer.h>
#include "scene.hpp"

#include "attribute_encoding.h"


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace compression {


// 类型：OutputBitStream。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class OutputBitStream
{
public:
  OutputBitStream() {}

  OutputBitStream(size_t byteSize, uint32_t* data) { init(byteSize, data); }


  // 函数：init。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void init(size_t byteSize, uint32_t* data)
  {

    assert(byteSize % sizeof(uint32_t) == 0);
    m_data = data;
    m_bitsSize = byteSize * 8;
    m_bitsPos = 0;
  }

  size_t getWrittenBitsCount() const { return m_bitsPos; }


  // 函数：write。把当前状态写入缓存、缓冲、文件或着色器可消费的数据布局。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：写入路径应明确字节对齐、所有权和可见性，避免后续读取端解释错误。
  void write(uint32_t val, uint32_t bitCount)
  {

    assert(bitCount <= 32);

    assert(m_bitsPos + bitCount <= m_bitsSize);

    val &= bitCount == 32 ? ~0u : ((1u << bitCount) - 1);

    size_t   idxLo = m_bitsPos / 32;
    size_t   idxHi = (m_bitsPos + bitCount - 1) / 32;

    uint32_t shift = uint32_t(m_bitsPos % 32);

    if(shift == 0)
    {
      m_data[idxLo] = val;
    }
    else
    {
      m_data[idxLo] |= val << shift;
    }

    if(shift + bitCount > 32)
    {
      m_data[idxHi] = val >> (32 - shift);
    }
    m_bitsPos += bitCount;
  }

  template <typename T>


  // 函数：write。把当前状态写入缓存、缓冲、文件或着色器可消费的数据布局。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：写入路径应明确字节对齐、所有权和可见性，避免后续读取端解释错误。
  void write(const T& tValue)
  {
    static_assert(sizeof(T) <= sizeof(uint32_t));
    union
    {
      uint32_t u32;
      T        t;
    };
    u32 = 0;
    t   = tValue;
    write(u32, sizeof(T) * 8);
  }

private:
    uint32_t* m_data = nullptr;
    size_t    m_bitsSize = 0;
    size_t    m_bitsPos = 0;
};


// 类型：InputBitStream。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class InputBitStream
{
public:
  InputBitStream() {}
  InputBitStream(size_t byteSize, const uint32_t* data) { init(byteSize, data); }


  // 函数：init。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void init(size_t byteSize, const uint32_t* data)
  {
    assert(byteSize % sizeof(uint32_t) == 0);
    m_data     = data;
    m_bitsPos  = 0;
    m_bitsSize = byteSize * 8;
  }


  // 函数：read。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
  void read(uint32_t* value, uint32_t bitCount)
  {

    assert(bitCount <= 32);

    assert(m_bitsPos + bitCount <= m_bitsSize);
    size_t   idxLo = m_bitsPos / 32;
    size_t   idxHi = (m_bitsPos + bitCount - 1) / 32;

    uint32_t shift = uint32_t(m_bitsPos % 32);

    union
    {
      uint64_t u64;
      uint32_t u32[2];
    };

    u32[0] = m_data[idxLo];
    u32[1] = m_data[idxHi];


    value[0] = uint32_t(u64 >> shift);

    value[0] &= bitCount == 32 ? ~0u : ((1u << bitCount) - 1);
    m_bitsPos += bitCount;
  }

  template <typename T>


  // 函数：read。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
  void read(T& value)
  {
    static_assert(sizeof(T) <= sizeof(uint32_t));
    union
    {
      uint32_t u32;
      T        tValue;
    };
    read(&u32, sizeof(T) * 8);
    value = tValue;
  }

  size_t getBytesRead() const { return sizeof(uint32_t) * ((m_bitsPos + 31) / 32); }
  size_t getElementsRead() const { return ((m_bitsPos + 31) / 32); }
private:
  const uint32_t* m_data     = nullptr;
  size_t          m_bitsSize = 0;
  size_t          m_bitsPos  = 0;
};


template <class T, uint32_t DIM>


// 类型：ArithmeticDeCompressor。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class ArithmeticDeCompressor
{
public:


  // 函数：init。初始化本模块所需状态、资源或 GPU 侧绑定。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：初始化过程建立后续阶段假定存在的不变量，例如句柄有效、缓冲大小足够、描述符已绑定。
  void init(size_t byteSize, const uint32_t* data)
  {

    m_input.init(byteSize, data);
    uint16_t outShifts;
    uint16_t outPrecs;

    m_input.read(outShifts);

    m_input.read(outPrecs);
    for(uint32_t d = 0; d < DIM; d++)
    {

      m_shifts[d]     = (outShifts >> (d * 5)) & 31;
      m_precisions[d] = ((outPrecs >> (d * 5)) & 31) + 1;

      m_input.read(m_lo[d]);
    }
  }


  // 函数：readVertices。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
  size_t readVertices(size_t count, T* output, size_t strideInElements)
  {
    for(size_t v = 0; v < count; v++)
    {
        T* vec = output + v * strideInElements;
        for (uint32_t d = 0; d < DIM; d++)
      {
        uint32_t deltaBits = 0;

        m_input.read(&deltaBits, m_precisions[d]);

        vec[d] = m_lo[d] + (deltaBits << m_shifts[d]);
      }
    }
    return m_input.getBytesRead();
  }
public:
    T   m_lo[DIM];
    int m_shifts[DIM] = {};
    int m_precisions[DIM] = {};
  InputBitStream m_input;
};


template <class T, uint32_t DIM>


// 类型：ArithmeticCompressor。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class ArithmeticCompressor
{
public:

  ArithmeticCompressor()
  {

    for(uint32_t d = 0; d < DIM; d++)
    {

      m_lo[d]    = std::numeric_limits<T>::max();

      m_hi[d]    = std::numeric_limits<T>::min();
      m_masks[d] = 0;
    }
  }

  template <typename Tindices>


  // 函数：registerVertices。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void registerVertices(size_t count, const Tindices* indices, size_t vecSize, const T* vecBuffer, size_t vecStrideInElements)
  {
    m_count = count;

    for(size_t i = 0; i < count; i++)
    {
      size_t index = indices[i];

      assert(index < vecSize);
      const T* vec = &vecBuffer[index * vecStrideInElements];
      for(uint32_t d = 0; d < DIM; d++)
      {

        m_lo[d] = std::min(m_lo[d], vec[d]);

        m_hi[d] = std::max(m_hi[d], vec[d]);
      }
    }

    for(size_t i = 0; i < count; i++)
    {
      size_t   index = indices[i];
      const T* vec   = &vecBuffer[index * vecStrideInElements];
      for(uint32_t d = 0; d < DIM; d++)
      {
          uint32_t dv = vec[d] - m_lo[d];
          m_masks[d] |= dv;
      }
    }

    computeVertexSize();
  }


  // 函数：getOutputByteSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  size_t getOutputByteSize() const
  {

    size_t numDeltaBits = 0;
    for(uint32_t d = 0; d < DIM; d++)
    {
      numDeltaBits += m_precisions[d];
    }
    numDeltaBits *= m_count;


    return sizeof(uint32_t) * ((16 + 16 + 32 * 3 + numDeltaBits + 31) / 32);
  }


  // 函数：beginOutput。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void beginOutput(size_t byteSize, uint32_t* out)
  {
    assert(byteSize <= getOutputByteSize());

    outBits.init(byteSize, out);

    uint16_t outShifts = m_shifts[0];
    uint16_t outPrec   = m_precisions[0] - 1;

    for(uint32_t d = 1; d < DIM; d++)
    {
      outShifts |= m_shifts[d] << (d * 5);
      outPrec |= (m_precisions[d] - 1) << (d * 5);
    }

    outBits.write(outShifts);

    outBits.write(outPrec);
    for(uint32_t d = 0; d < DIM; d++)
    {

      outBits.write(m_lo[d]);
    }
  }

  template <typename Tindices>


  // 函数：outputVertices。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  void outputVertices(size_t count, const Tindices* indices, size_t vecSize, const T* vecBuffer, size_t vecStrideInElements)
  {
    for(size_t i = 0; i < count; i++)
    {
      size_t index = indices[i];

      assert(index < vecSize);
      const T* vec = &vecBuffer[index * vecStrideInElements];
      for(uint32_t d = 0; d < DIM; d++)
      {

        outBits.write((vec[d] - m_lo[d]) >> m_shifts[d], m_precisions[d]);
      }
    }
  }

public:
  T      m_lo[DIM];
  T      m_hi[DIM];
  T      m_masks[DIM];
  size_t m_count           = 0;
  int    m_shifts[DIM]     = {};
  int    m_precisions[DIM] = {};
  OutputBitStream outBits;


  // 函数：computeVertexSize。计算派生值，供后续剔除、LOD、统计或资源规划使用。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：计算结果通常参与阈值比较或内存规划，数值稳定性和边界条件需要特别注意。
  void computeVertexSize()
  {
    for(uint32_t d = 0; d < DIM; ++d)
    {
      if(m_masks[d] == 0)
      {
        m_shifts[d]     = 31;
        m_precisions[d] = 1;
      }
      else
      {


        m_shifts[d] = std::countr_zero(m_masks[d]);
        const uint32_t value_range = m_hi[d] - m_lo[d];


        int            bits        = std::bit_width(value_range >> m_shifts[d]);
        m_precisions[d]            = std::max(bits, int(1));
      }
    }
  }
};
}


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace lodclusters {


// 函数：Scene::compressGroup。执行压缩或解压流程，在体积和运行时访问格式之间做转换。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：压缩必须保留可验证的重建语义；当压缩收益不足或超出约束时应回退到未压缩表示。
void Scene::compressGroup(TempContext* context, GroupStorage& groupTempStorage, GroupInfo& groupInfo, uint32_t* vertexCacheLocal)
{
  GeometryStorage& geometry = context->geometry;

  size_t attributeStride = geometry.vertexAttributes.size() / geometry.vertexPositions.size();

  uint32_t vertexOffset = 0;
  uint32_t vertexDataOffset = 0;
  for(uint32_t c = 0; c < groupInfo.clusterCount; c++)
  {
    const uint32_t* localVertices = vertexCacheLocal + vertexOffset;
    shaderio::Cluster& cluster     = groupTempStorage.clusters[c];
    uint32_t           vertexCount = cluster.vertexCountMinusOne + 1;

    cluster.indices = vertexDataOffset;

    {
      compression::ArithmeticCompressor<uint32_t, 3> compressor;

      compressor.registerVertices(vertexCount, localVertices, geometry.vertexPositions.size(),(const uint32_t*)geometry.vertexPositions.data(), 3);

      size_t compressedSize = compressor.getOutputByteSize();

      if(compressedSize >= sizeof(glm::vec3) * vertexCount)
      {

        for(uint32_t v = 0; v < vertexCount; v++)
        {
          memcpy(&groupTempStorage.vertices[vertexDataOffset + v * 3], &geometry.vertexPositions[localVertices[v]],sizeof(glm::vec3));
        }
        vertexDataOffset += 3 * vertexCount;
      }
      else
      {

        cluster.attributeBits |= shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_POS;

        compressor.beginOutput(compressedSize, (uint32_t*)&groupTempStorage.vertices[vertexDataOffset]);
        compressor.outputVertices(vertexCount, localVertices, geometry.vertexPositions.size(),(const uint32_t*)geometry.vertexPositions.data(), 3);


        vertexDataOffset += uint32_t(compressedSize / sizeof(uint32_t));
      }
    }

    if(geometry.attributeNormalOffset != ~0)
    {
      if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT)
      {
        for(uint32_t v = 0; v < vertexCount; v++)
        {

          glm::vec3 normal = *(const glm::vec3*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeNormalOffset]);
          glm::vec4 tangent = *(const glm::vec4*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeTangentOffset]);


          uint32_t encoded = shaderio::normal_pack(normal);

          encoded |= shaderio::tangent_pack(normal, tangent) << ATTRENC_NORMAL_BITS;

          *(uint32_t*)&groupTempStorage.vertices[vertexDataOffset + v] = encoded;
        }
      }
      else
      {
        for(uint32_t v = 0; v < vertexCount; v++)
        {
          glm::vec3 tmp = *(const glm::vec3*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeNormalOffset]);

          uint32_t encoded = shaderio::normal_pack(tmp);
          *(uint32_t*)&groupTempStorage.vertices[vertexDataOffset + v] = encoded;
        }
      }
      vertexDataOffset += vertexCount;
    }

    for(uint32_t t = 0; t < 2; t++)
    {

      shaderio::ClusterAttributeBits usedBit = t == 0 ? shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0 : shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1;
      shaderio::ClusterAttributeBits compressedBit = t == 0 ? shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_TEX_0 :shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_TEX_1;
      uint32_t attributeTexOffset = t == 0 ? geometry.attributeTex0offset : geometry.attributeTex1offset;
      if(geometry.attributeBits & usedBit)
      {
        compression::ArithmeticCompressor<uint32_t, 2> compressor;

        compressor.registerVertices(vertexCount, localVertices, geometry.vertexPositions.size(),(const uint32_t*)(geometry.vertexAttributes.data() + attributeTexOffset), attributeStride);

        size_t compressedSize = compressor.getOutputByteSize();

        if(compressedSize >= sizeof(glm::vec2) * vertexCount)
        {

          for(uint32_t v = 0; v < vertexCount; v++)
          {
            const glm::vec2* attribute =(const glm::vec2*)&geometry.vertexAttributes[localVertices[v] * attributeStride + attributeTexOffset];
            memcpy(&groupTempStorage.vertices[vertexDataOffset + v * 2], attribute, sizeof(glm::vec2));
          }
          vertexDataOffset += 2 * vertexCount;
        }
        else
        {
          cluster.attributeBits |= compressedBit;
          compressor.beginOutput(compressedSize, (uint32_t*)&groupTempStorage.vertices[vertexDataOffset]);

          compressor.outputVertices(vertexCount, localVertices, geometry.vertexPositions.size(),(const uint32_t*)(geometry.vertexAttributes.data() + attributeTexOffset), attributeStride);
          vertexDataOffset += uint32_t(compressedSize / sizeof(uint32_t));
        }
      }
    }
    vertexOffset += vertexCount;
  }

  context->processingInfo.stats.vertexCompressedBytes += sizeof(uint32_t) * vertexDataOffset;


  groupInfo.uncompressedSizeBytes       = groupInfo.sizeBytes;
  groupInfo.uncompressedVertexDataCount = groupInfo.vertexDataCount;
  groupInfo.vertexDataCount             = vertexDataOffset;

  groupInfo.sizeBytes                   = groupInfo.computeSize();
}


// 函数：Scene::decompressGroup。执行压缩或解压流程，在体积和运行时访问格式之间做转换。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：压缩必须保留可验证的重建语义；当压缩收益不足或超出约束时应回退到未压缩表示。
void Scene::decompressGroup(const GroupInfo& info, const GroupView& groupSrc, void* dstWriteOnly, size_t dstSize)
{


  GroupInfo uncompressedInfo       = info;
  uncompressedInfo.sizeBytes       = info.uncompressedSizeBytes;
  uncompressedInfo.vertexDataCount = info.uncompressedVertexDataCount;


  // 函数：groupDstWriteOnly。把当前状态写入缓存、缓冲、文件或着色器可消费的数据布局。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：写入路径应明确字节对齐、所有权和可见性，避免后续读取端解释错误。
  GroupStorage groupDstWriteOnly(dstWriteOnly, uncompressedInfo);

  memcpy(dstWriteOnly, groupSrc.raw, info.computeUncompressedSectionSize());
  uint32_t indicesOffset = 0;
  for(uint32_t c = 0; c < info.clusterCount; c++)
  {
    shaderio::Cluster&       clusterDstWriteOnly = groupDstWriteOnly.clusters[c];
    const shaderio::Cluster& clusterSrc          = groupSrc.clusters[c];
    uint32_t                 triangleCount       = clusterSrc.triangleCountMinusOne + 1;
    uint32_t                 vertexCount         = clusterSrc.vertexCountMinusOne + 1;
    uint32_t                 vertexDataOffset    = clusterSrc.vertices;


    uint32_t* dstData = groupDstWriteOnly.getClusterLocalData(c, clusterSrc.vertices);


    const uint32_t* srcData = (const uint32_t*)groupSrc.getClusterIndices(c);

    clusterDstWriteOnly.indices = groupDstWriteOnly.getClusterLocalOffset(c, groupDstWriteOnly.indices.data() + indicesOffset);
    indicesOffset += triangleCount * 3;
    uint32_t dstOffset = 0;

    if(clusterSrc.attributeBits & shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_POS)
    {


      ptrdiff_t srcSize = ptrdiff_t(groupSrc.vertices.data() + groupSrc.vertices.size()) - ptrdiff_t(srcData);

      assert(srcSize >= 0);

      compression::ArithmeticDeCompressor<uint32_t, 3> decompressor;
      decompressor.init(size_t(srcSize), srcData);

      srcData += decompressor.readVertices(vertexCount, dstData + dstOffset, 3) / sizeof(uint32_t);
      dstOffset += 3 * vertexCount;
    }
    else
    {

      memcpy(dstData, srcData, sizeof(glm::vec3) * vertexCount);
      srcData += 3 * vertexCount;
      dstOffset += 3 * vertexCount;
    }


    if(clusterSrc.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL)
    {
      memcpy(dstData + dstOffset, srcData, sizeof(uint32_t) * vertexCount);
      srcData += vertexCount;
      dstOffset += vertexCount;
    }

    for(uint32_t t = 0; t < 2; t++)
    {
      shaderio::ClusterAttributeBits usedBit = t == 0 ? shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0 : shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1;
      shaderio::ClusterAttributeBits compressedBit = t == 0 ? shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_TEX_0 : shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_TEX_1;

      if((clusterSrc.attributeBits & (usedBit | compressedBit)) == (usedBit | compressedBit))
      {

        ptrdiff_t srcSize = ptrdiff_t(groupSrc.vertices.data() + groupSrc.vertices.size()) - ptrdiff_t(srcData);

        assert(srcSize >= 0);

        compression::ArithmeticDeCompressor<uint32_t, 2> decompressor;
        decompressor.init(size_t(srcSize), srcData);
        srcData += decompressor.readVertices(vertexCount, dstData + dstOffset, 2) / sizeof(uint32_t);
        dstOffset += 2 * vertexCount;
      }
      else if(clusterSrc.attributeBits & usedBit)
      {


        dstOffset = (dstOffset + 1) & ~1;
        memcpy(dstData + dstOffset, srcData, sizeof(glm::vec2) * vertexCount);
        srcData += 2 * vertexCount;
        dstOffset += 2 * vertexCount;
      }
    }

    assert(size_t(dstData + dstOffset) <= size_t(dstWriteOnly) + dstSize);
  }
}
}
