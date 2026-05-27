//集群数据进行压缩
#include <bit>
// 引入算法库，提供 std::min, std::max 等函数
#include <algorithm>
#include <meshoptimizer.h>
#include "scene.hpp"
// 引入 Shader 与 C++ 共享的顶点属性编码算法（如法线/切线的八面体映射压缩 normal_pack）
#include "../shaders/attribute_encoding.h"
// 命名空间：专用于基础的数据压缩算法
namespace compression {
//=============================================================================
// 类：OutputBitStream (输出比特流)
// 作用：允许我们将任意位（bit）长度的数据紧密打包进一个 uint32_t 数组中，突破字节对齐限制
//=============================================================================
class OutputBitStream
{
public:
  OutputBitStream() {}
  // 构造并初始化，传入最大字节数和目标数组指针
  OutputBitStream(size_t byteSize, uint32_t* data) { init(byteSize, data); }
  void init(size_t byteSize, uint32_t* data)
  {
    // 确保分配的内存大小是 4字节 (32位) 的整数倍
    assert(byteSize % sizeof(uint32_t) == 0);
    m_data = data;           // 目标数据写入的指针
    m_bitsSize = byteSize * 8;   // 将最大容量转换为 "位(bit)" 的数量
    m_bitsPos = 0;              // 当前写入位置的位游标（从第0位开始）
  }
  // 获取已经写入的比特总数
  size_t getWrittenBitsCount() const { return m_bitsPos; }
  // 核心写入函数：将 val 的低 bitCount 位，写入到比特流中
  void write(uint32_t val, uint32_t bitCount)
  {
    assert(bitCount <= 32);// 一次最多写32位
    assert(m_bitsPos + bitCount <= m_bitsSize);// 防止越界写入内存
    // 掩码操作：把 val 高位无用的数据清零，只保留我们要的 bitCount 位
    val &= bitCount == 32 ? ~0u : ((1u << bitCount) - 1);
    // 计算当前要写入的比特对应 m_data 数组里的哪几个 uint32_t 元素
    size_t   idxLo = m_bitsPos / 32;                  // 数据落入的第一个 32位 整数的索引
    size_t   idxHi = (m_bitsPos + bitCount - 1) / 32; // 数据末尾落入的 32位 整数的索引 (可能跨越了 uint32 的边界)
    uint32_t shift = uint32_t(m_bitsPos % 32);        // 在第一个 32位 整数内部的位偏移量
    // 处理低位部分（写入 idxLo）
    if(shift == 0)
    {
      m_data[idxLo] = val;// 如果刚好对齐到 32 位的开头，直接赋值
    }
    else
    {
      m_data[idxLo] |= val << shift;// 如果不对齐，将 val 左移后，通过按位或(|)追加到现有数据后面
    }
    // 如果这次写入跨越了当前的 uint32_t，需要把剩下的高位部分写到下一个 uint32_t (idxHi) 里
    if(shift + bitCount > 32)
    {
      m_data[idxHi] = val >> (32 - shift);// 把刚才没存完的高位部分右移拿出来，存入下一个数组元素
    }
    m_bitsPos += bitCount;// 推进位游标
  }
  // 模板写入函数：方便直接写入 float、int 等任意类型（只要大小不超过32位）
  template <typename T>
  void write(const T& tValue)
  {
    static_assert(sizeof(T) <= sizeof(uint32_t));// 编译期检查大小
    union
    {
      uint32_t u32;
      T        t;
    };// 利用 union 绕过严格别名规则，把 T 类型直接当作 uint32_t 的位模式来读取
    u32 = 0;
    t   = tValue;
    write(u32, sizeof(T) * 8);// 将整个类型的位数完整写入
  }

private:
    uint32_t* m_data = nullptr; // 底层数据数组
    size_t    m_bitsSize = 0;       // 总容量 (bits)
    size_t    m_bitsPos = 0;       // 当前写入进度 (bits)
};
//=============================================================================
// 类：InputBitStream (输入比特流)
// 作用：OutputBitStream 的逆过程，从紧密打包的数组中按指定的 bit 数读出数据
//=============================================================================
class InputBitStream
{
public:
  InputBitStream() {}
  InputBitStream(size_t byteSize, const uint32_t* data) { init(byteSize, data); }
  void init(size_t byteSize, const uint32_t* data)
  {
    assert(byteSize % sizeof(uint32_t) == 0);
    m_data     = data;
    m_bitsPos  = 0;
    m_bitsSize = byteSize * 8;
  }
  // 核心读取函数：从流中读取 bitCount 个位，存入 value
  void read(uint32_t* value, uint32_t bitCount)
  {
    assert(bitCount <= 32);
    assert(m_bitsPos + bitCount <= m_bitsSize);
    size_t   idxLo = m_bitsPos / 32;
    size_t   idxHi = (m_bitsPos + bitCount - 1) / 32;
    uint32_t shift = uint32_t(m_bitsPos % 32);
    // 巧妙利用 uint64_t 一次性读取可能跨越两个 uint32_t 边界的数据
    union
    {
      uint64_t u64;
      uint32_t u32[2];
    };

    u32[0] = m_data[idxLo]; // 读入低位 32bit
    u32[1] = m_data[idxHi]; // 读入高位 32bit (即使 idxLo == idxHi 也没关系，只是多读了一次)

    value[0] = uint32_t(u64 >> shift); // 整体右移掉之前已经读过的 bit，把想要的数据对齐到开头
    // 掩码清空高位多余的数据，只保留 bitCount 长度的有效值
    value[0] &= bitCount == 32 ? ~0u : ((1u << bitCount) - 1);
    m_bitsPos += bitCount;// 推进读取游标
  }
  // 模板读取函数：直接读出特定类型（如 float）
  template <typename T>
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
  // 获取已读取的字节数（向上取整到 uint32 的边界）
  size_t getBytesRead() const { return sizeof(uint32_t) * ((m_bitsPos + 31) / 32); }
  size_t getElementsRead() const { return ((m_bitsPos + 31) / 32); }
private:
  const uint32_t* m_data     = nullptr;
  size_t          m_bitsSize = 0;
  size_t          m_bitsPos  = 0;
};
//=============================================================================
// 模板类：ArithmeticDeCompressor (算术解压器)
// 作用：将使用包围盒与位移量压缩的 N 维数据（如 3维的XYZ坐标）还原为原始浮点数
// T: 数据类型(如uint32_t)，DIM: 维度(如3代表3D坐标)
//=============================================================================
template <class T, uint32_t DIM>
class ArithmeticDeCompressor
{
public:
  void init(size_t byteSize, const uint32_t* data)
  {
    m_input.init(byteSize, data);// 初始化输入比特流
    uint16_t outShifts;// 压缩时每个维度被截断了多少个低位0
    uint16_t outPrecs;// 压缩时每个维度保留了多少位有效精度
    m_input.read(outShifts);// 读出打包好的 位移量
    m_input.read(outPrecs);// 读出打包好的 精度大小
    for(uint32_t d = 0; d < DIM; d++)
    {
      // 从 16位 整数中拆解出每个维度的 位移 和 精度 (每个占 5 bits，支持最大 32 的值)
      m_shifts[d]     = (outShifts >> (d * 5)) & 31;
      m_precisions[d] = ((outPrecs >> (d * 5)) & 31) + 1;// 精度存的时候减了1，读出来要加1
      m_input.read(m_lo[d]);// 读取该维度包围盒的最小值（Base Value 基准点）
    }
  }
  // 读取 count 个顶点，并将解压后的数据写入 output
  size_t readVertices(size_t count, T* output, size_t strideInElements)
  {
    for(size_t v = 0; v < count; v++)
    {
        T* vec = output + v * strideInElements; // 找到当前顶点写入的内存地址
        for (uint32_t d = 0; d < DIM; d++)       // 遍历每个维度 (X, Y, Z)
      {
        uint32_t deltaBits = 0;
        m_input.read(&deltaBits, m_precisions[d]); // 根据该维度所需精度，读出压缩后的差值
        // 还原公式： 原始值 = 最小值基准 + (压缩差值 << 右移丢弃的位数)
        vec[d] = m_lo[d] + (deltaBits << m_shifts[d]);
      }
    }
    return m_input.getBytesRead();// 返回总共读了多少字节
  }
public:
    T   m_lo[DIM];            // 各维度的最小值(基准点)
    int m_shifts[DIM] = {}; // 各维度的位移量
    int m_precisions[DIM] = {}; // 各维度的精度位数
  InputBitStream m_input;
};
//=============================================================================
// 模板类：ArithmeticCompressor (算术压缩器)
// 作用：分析一组 N 维数据，算出它们的包围盒极值和公共的冗余位，从而自适应地决定最少的存储位数
//=============================================================================
template <class T, uint32_t DIM>
class ArithmeticCompressor
{
public:
  ArithmeticCompressor()
  {
    // 初始化极值为系统最大/最小值，准备找包围盒
    for(uint32_t d = 0; d < DIM; d++)
    {
      m_lo[d]    = std::numeric_limits<T>::max();
      m_hi[d]    = std::numeric_limits<T>::min();
      m_masks[d] = 0;
    }
  }
  // 第一步：注册所有顶点（扫描数据）
  template <typename Tindices>
  void registerVertices(size_t count, const Tindices* indices, size_t vecSize, const T* vecBuffer, size_t vecStrideInElements)
  {
    m_count = count;
    // 扫描一：找到这个集群(Cluster)中所有顶点的包围盒最小值 (m_lo) 和 最大值 (m_hi)
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
    // 扫描二：计算每个坐标偏离最小值的相对差值，并将它们的 bit 叠加起来形成掩码 (m_masks)
    for(size_t i = 0; i < count; i++)
    {
      size_t   index = indices[i];
      const T* vec   = &vecBuffer[index * vecStrideInElements];
      for(uint32_t d = 0; d < DIM; d++)
      {
          uint32_t dv = vec[d] - m_lo[d]; // dv 是相对起点的差值
          m_masks[d] |= dv;               // 把所有顶点的差值的 bit 位混合在一起，看看哪些 bit 被用到了
      }
    }
    computeVertexSize();// 根据掩码计算最终能压缩到多少位
  }
  // 获取压缩后大概需要占用多少字节的内存
  size_t getOutputByteSize() const
  {
    // vertex bits
    size_t numDeltaBits = 0;
    for(uint32_t d = 0; d < DIM; d++)
    {
      numDeltaBits += m_precisions[d];// 算出存一个顶点所需的总 bit 数
    }
    numDeltaBits *= m_count;// 所有顶点数据所需的 bit 数

    // 计算总字节数： 偏移元数据(16) + 精度元数据(16) + 三个维度的基准值(32*3) + 数据本体 + 对齐所需补齐的bit
    return sizeof(uint32_t) * ((16 + 16 + 32 * 3 + numDeltaBits + 31) / 32);
  }
  // 第二步：开始输出写入数据，先写入元数据头（Header）
  void beginOutput(size_t byteSize, uint32_t* out)
  {
    assert(byteSize <= getOutputByteSize());
    outBits.init(byteSize, out);
    // 把各维度的位移量和精度分别打包进两个 16位的整数 里 (每个占 5 bits)
    uint16_t outShifts = m_shifts[0];
    uint16_t outPrec   = m_precisions[0] - 1;
    // 精度至少是1，所以减1存可以省状态
    for(uint32_t d = 1; d < DIM; d++)
    {
      outShifts |= m_shifts[d] << (d * 5);
      outPrec |= (m_precisions[d] - 1) << (d * 5);
    }
    outBits.write(outShifts);// 写入各维度位移量
    outBits.write(outPrec);// 写入各维度精度
    for(uint32_t d = 0; d < DIM; d++)
    {
      outBits.write(m_lo[d]);// 写入各维度的最小值 (基准点)
    }
  }
  // 第三步：输出所有顶点的压缩数据本体
  template <typename Tindices>
  void outputVertices(size_t count, const Tindices* indices, size_t vecSize, const T* vecBuffer, size_t vecStrideInElements)
  {
    for(size_t i = 0; i < count; i++)
    {
      size_t index = indices[i];
      assert(index < vecSize);
      const T* vec = &vecBuffer[index * vecStrideInElements];
      for(uint32_t d = 0; d < DIM; d++)
      {
        // 压缩核心公式：(当前值 - 最小值) >> 位移量。 然后只取 m_precisions 长度的 bit 写入！
        outBits.write((vec[d] - m_lo[d]) >> m_shifts[d], m_precisions[d]);
      }
    }
  }

public:
  T      m_lo[DIM];        // 极小值
  T      m_hi[DIM];        // 极大值
  T      m_masks[DIM];     // bit 掩码
  size_t m_count           = 0;
  int    m_shifts[DIM]     = {};
  int    m_precisions[DIM] = {};
  OutputBitStream outBits;
  // 根据扫描到的掩码，计算最省空间的位移和精度
  void computeVertexSize()
  {
    for(uint32_t d = 0; d < DIM; ++d)
    {
      if(m_masks[d] == 0)// 如果掩码是0，说明这个维度上所有顶点值完全一样！
      {
        m_shifts[d]     = 31;// 最大位移
        m_precisions[d] = 1;// 只需 1个 bit 表示即可 (因为差值全是0)
      }
      else
      {
        // std::countr_zero 计算二进制末尾连续的 0 的个数
        // 例如差值都是 4, 8, 12... 它们的二进制末尾都有 2 个 0。
        // 这说明数据的步长很大，可以直接右移 2 位截断，解压时再左移还原，省下存储空间。
        m_shifts[d] = std::countr_zero(m_masks[d]);
        const uint32_t value_range = m_hi[d] - m_lo[d]; // 最大跨度
        // std::bit_width 计算表示这个跨度所需的最小位数
        // 例如跨度是 100，转二进制需要 7 bits，所以这组数据只要 7 位就能存下！
        int            bits        = std::bit_width(value_range >> m_shifts[d]);
        m_precisions[d]            = std::max(bits, int(1));// 精度至少保留1位
      }
    }
  }
};
}
// 命名空间：与 LOD 簇（微网格）渲染系统相关的逻辑
namespace lodclusters {
//=============================================================================
// 函数：Scene::compressGroup
// 作用：对一个渲染组（Group，包含多个微网格Cluster）的顶点数据进行极致压缩
//=============================================================================
void Scene::compressGroup(TempContext* context, GroupStorage& groupTempStorage, GroupInfo& groupInfo, uint32_t* vertexCacheLocal)
{
  GeometryStorage& geometry = context->geometry;// 获取全局的几何数据
  size_t attributeStride = geometry.vertexAttributes.size() / geometry.vertexPositions.size();// 计算顶点属性的步长
  // per-cluster 循环：逐个微网格集群进行压缩
  uint32_t vertexOffset = 0;  // 原始顶点索引偏移
  uint32_t vertexDataOffset = 0;  // 压缩后写入目标缓冲的偏移量（单位：uint32_t）
  for(uint32_t c = 0; c < groupInfo.clusterCount; c++)
  {
    const uint32_t* localVertices = vertexCacheLocal + vertexOffset;// 获取当前集群的顶点索引列表
    shaderio::Cluster& cluster     = groupTempStorage.clusters[c];
    uint32_t           vertexCount = cluster.vertexCountMinusOne + 1; // 顶点数（为了节省1bit，存储时减了1）
    // Hack技巧：因为集群的索引数据会被另外处理，这里借用 cluster.indices 字段来记录该集群压缩后的顶点数据的起始偏移量
    cluster.indices = vertexDataOffset;
    // --------- 1. 压缩顶点位置 (Positions, 3维) ---------
    {
      compression::ArithmeticCompressor<uint32_t, 3> compressor;
      // 注册分析当前集群的 XYZ 坐标
      compressor.registerVertices(vertexCount, localVertices, geometry.vertexPositions.size(),(const uint32_t*)geometry.vertexPositions.data(), 3);
      size_t compressedSize = compressor.getOutputByteSize();
      // 防劣化机制：如果压缩后反而比不压缩（3*32bit=12字节）还大，就不压缩了
      if(compressedSize >= sizeof(glm::vec3) * vertexCount)
      {
        // output uncompressed: 直接原样拷贝坐标数据
        for(uint32_t v = 0; v < vertexCount; v++)
        {
          memcpy(&groupTempStorage.vertices[vertexDataOffset + v * 3], &geometry.vertexPositions[localVertices[v]],sizeof(glm::vec3));
        }
        vertexDataOffset += 3 * vertexCount;
      }
      else
      {
        // 成功压缩：打上标志位，告诉后续 Shader/CPU 这是一个被算术压缩过的集群
        cluster.attributeBits |= shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_POS;
        // 执行写入打包
        compressor.beginOutput(compressedSize, (uint32_t*)&groupTempStorage.vertices[vertexDataOffset]);
        compressor.outputVertices(vertexCount, localVertices, geometry.vertexPositions.size(),(const uint32_t*)geometry.vertexPositions.data(), 3);
        // #if 0 屏蔽掉的自检代码，用来测试压缩后再解压是否和原数据完全一致
//#if 0
//        {
//          // validate decompressor
//          compression::ArithmeticDeCompressor<uint32_t, 3> decompressor;
//          decompressor.init(compressedSize, (uint32_t*)&groupTempStorage.vertices[vertexDataOffset]);
//
//          glm::vec3 temp[256];
//          size_t    bytesRead = decompressor.readVertices(vertexCount, (uint32_t*)temp, 3);
//
//          for(uint32_t v = 0; v < vertexCount; v++)
//          {
//            glm::vec3 pos = geometry.vertexPositions[localVertices[ v]];
//            assert(pos.x == temp[v].x);
//            assert(pos.y == temp[v].y);
//            assert(pos.z == temp[v].z);
//          }
//
//          assert(bytesRead == compressedSize);
//        }
//#endif
        vertexDataOffset += uint32_t(compressedSize / sizeof(uint32_t));// 更新写入游标
      }
    }
    // --------- 2. 压缩顶点法线和切线 (Normals & Tangents) ---------
    if(geometry.attributeNormalOffset != ~0)// 如果模型有法线数据
    {
      if(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT)// 如果有切线数据
      {
        for(uint32_t v = 0; v < vertexCount; v++)
        {
          // 读出原始的浮点法线 (vec3) 和 切线 (vec4)
          glm::vec3 normal = *(const glm::vec3*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeNormalOffset]);
          glm::vec4 tangent = *(const glm::vec4*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeTangentOffset]);
          // 利用八面体映射算法将 3个float 压成一小段整数
          uint32_t encoded = shaderio::normal_pack(normal);
          // 将切线也压成一段整数，并通过左移将其和法线强行塞进同一个 32位 uint 里！极致压缩！
          encoded |= shaderio::tangent_pack(normal, tangent) << ATTRENC_NORMAL_BITS;
          // 写入这 32位的编码
          *(uint32_t*)&groupTempStorage.vertices[vertexDataOffset + v] = encoded;
        }
      }
      else
      {
        for(uint32_t v = 0; v < vertexCount; v++)// 如果只有法线没有切线
        {
          glm::vec3 tmp = *(const glm::vec3*)(&geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeNormalOffset]);
          uint32_t encoded = shaderio::normal_pack(tmp);// 仅压缩法线
          *(uint32_t*)&groupTempStorage.vertices[vertexDataOffset + v] = encoded;
        }
      }
      vertexDataOffset += vertexCount;// 每顶点法线只占了 1 个 uint32_t，游标加 vertexCount
    }
    // --------- 3. 压缩纹理坐标 (TexCoords, 最多2套 UV) ---------
    for(uint32_t t = 0; t < 2; t++)
    {
        // 根据是第0套还是第1套 UV，选择不同的标志位和数据偏移
      shaderio::ClusterAttributeBits usedBit = t == 0 ? shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0 : shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1;
      shaderio::ClusterAttributeBits compressedBit = t == 0 ? shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_TEX_0 :shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_TEX_1;
      uint32_t attributeTexOffset = t == 0 ? geometry.attributeTex0offset : geometry.attributeTex1offset;
      if(geometry.attributeBits & usedBit)// 如果几何体使用了这套 UV
      {
        compression::ArithmeticCompressor<uint32_t, 2> compressor;// 实例化一个2维的压缩器 (因为 UV 是二维数据)
        // 注册 UV 数据分析
        compressor.registerVertices(vertexCount, localVertices, geometry.vertexPositions.size(),(const uint32_t*)(geometry.vertexAttributes.data() + attributeTexOffset), attributeStride);
        size_t compressedSize = compressor.getOutputByteSize();
        // 同样的防劣化机制：比 2个float(8字节) 还大就不压缩
        if(compressedSize >= sizeof(glm::vec2) * vertexCount)
        {
            // 直接拷贝 uncompressed 数据
          for(uint32_t v = 0; v < vertexCount; v++)
          {
            const glm::vec2* attribute =(const glm::vec2*)&geometry.vertexAttributes[localVertices[v] * attributeStride + attributeTexOffset];
            memcpy(&groupTempStorage.vertices[vertexDataOffset + v * 2], attribute, sizeof(glm::vec2));
          }
          vertexDataOffset += 2 * vertexCount;
        }
        else
        {
          cluster.attributeBits |= compressedBit;// 标记该套 UV 已压缩
          compressor.beginOutput(compressedSize, (uint32_t*)&groupTempStorage.vertices[vertexDataOffset]);
          // 打包写入压缩的 UV
          compressor.outputVertices(vertexCount, localVertices, geometry.vertexPositions.size(),(const uint32_t*)(geometry.vertexAttributes.data() + attributeTexOffset), attributeStride);
          vertexDataOffset += uint32_t(compressedSize / sizeof(uint32_t));
        }
      }
    }
    vertexOffset += vertexCount;// 更新下一轮集群要读取的原顶点索引偏移
  }
  // 统计压缩后的总大小，存入上下文便于调试和日志
  context->processingInfo.stats.vertexCompressedBytes += sizeof(uint32_t) * vertexDataOffset;
  // 更新 GroupInfo 结构体，保留未压缩时的理论大小以便将来解压时分配内存，
  // 然后把当前的真实容量更新为压缩后的体积
  groupInfo.uncompressedSizeBytes       = groupInfo.sizeBytes;
  groupInfo.uncompressedVertexDataCount = groupInfo.vertexDataCount;
  groupInfo.vertexDataCount             = vertexDataOffset;
  groupInfo.sizeBytes                   = groupInfo.computeSize();
}
//=============================================================================
// 函数：Scene::decompressGroup
// 作用：对压缩过的集群组进行解压，恢复为标准浮点数格式，通常为了将其上传给显存 (GPU)
//=============================================================================
void Scene::decompressGroup(const GroupInfo& info, const GroupView& groupSrc, void* dstWriteOnly, size_t dstSize)
{
  // 假设目标是一块只写内存 (Write-Only Memory) —— 例如经过 Write-Combined 映射的显卡内存，直接写入最快
  // 1. 根据保存的信息，恢复其未压缩前的大小属性
  GroupInfo uncompressedInfo       = info;
  uncompressedInfo.sizeBytes       = info.uncompressedSizeBytes;
  uncompressedInfo.vertexDataCount = info.uncompressedVertexDataCount;
  // 创建一个目标内存的封装器，方便写数据
  GroupStorage groupDstWriteOnly(dstWriteOnly, uncompressedInfo);
  // 先把除顶点数据之外的固定长度部分（如集群元数据、Bbox等部分）直接拷贝过去
  memcpy(dstWriteOnly, groupSrc.raw, info.computeUncompressedSectionSize());
  uint32_t indicesOffset = 0;
  for(uint32_t c = 0; c < info.clusterCount; c++)
  {
    shaderio::Cluster&       clusterDstWriteOnly = groupDstWriteOnly.clusters[c];
    const shaderio::Cluster& clusterSrc          = groupSrc.clusters[c];
    uint32_t                 triangleCount       = clusterSrc.triangleCountMinusOne + 1;
    uint32_t                 vertexCount         = clusterSrc.vertexCountMinusOne + 1;
    uint32_t                 vertexDataOffset    = clusterSrc.vertices;
    // 获取写入目的地的起始指针 (解压后的数据要放到哪里)
    uint32_t* dstData = groupDstWriteOnly.getClusterLocalData(c, clusterSrc.vertices);
    // 之前压缩时留下的 Hack：cluster.indices 里存的是压缩数据的真实起始指针！
    const uint32_t* srcData = (const uint32_t*)groupSrc.getClusterIndices(c);
    // 既然读出了数据位置，这里就要把 clusterDstWriteOnly.indices 恢复为真正的三角形索引偏移量
    clusterDstWriteOnly.indices = groupDstWriteOnly.getClusterLocalOffset(c, groupDstWriteOnly.indices.data() + indicesOffset);
    indicesOffset += triangleCount * 3;
    uint32_t dstOffset = 0;// 局部写入游标
    // --------- 1. 恢复顶点位置 (Positions) ---------
    if(clusterSrc.attributeBits & shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_POS)
    {
      // 计算剩余的安全可读内存大小，防止越界
      ptrdiff_t srcSize = ptrdiff_t(groupSrc.vertices.data() + groupSrc.vertices.size()) - ptrdiff_t(srcData);
      assert(srcSize >= 0);
      // 实例化解压器，并从比特流中读取数据恢复为 float (即 uint32_t强转的vec3)
      compression::ArithmeticDeCompressor<uint32_t, 3> decompressor;
      decompressor.init(size_t(srcSize), srcData);
      // readVertices 返回读取了多少字节。除以 4 就是多少个 uint32_t
      srcData += decompressor.readVertices(vertexCount, dstData + dstOffset, 3) / sizeof(uint32_t);
      dstOffset += 3 * vertexCount;// 写完了一组 3维 数据
    }
    else
    {
     // 如果没被压缩，直接复制 3*32bit=12字节每顶点
      memcpy(dstData, srcData, sizeof(glm::vec3) * vertexCount);
      srcData += 3 * vertexCount;
      dstOffset += 3 * vertexCount;
    }
    // --------- 2. 恢复顶点法线 (Normals) ---------
    // 注意：这里只是复制！真正将 normal_pack (32位整型) 解码为 float 的逻辑，
    // 通常是在 GPU Shader 里面通过 oct_to_vec 函数高效完成的，CPU这边不需要解包。
    if(clusterSrc.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL)
    {
      memcpy(dstData + dstOffset, srcData, sizeof(uint32_t) * vertexCount);
      srcData += vertexCount;
      dstOffset += vertexCount;
    }
        // --------- 3. 恢复纹理坐标 (TexCoords) ---------
    for(uint32_t t = 0; t < 2; t++)
    {
      shaderio::ClusterAttributeBits usedBit = t == 0 ? shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0 : shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1;
      shaderio::ClusterAttributeBits compressedBit = t == 0 ? shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_TEX_0 : shaderio::CLUSTER_ATTRIBUTE_COMPRESSED_VERTEX_TEX_1;
      // 如果有被使用，且带有压缩标记
      if((clusterSrc.attributeBits & (usedBit | compressedBit)) == (usedBit | compressedBit))
      {
        ptrdiff_t srcSize = ptrdiff_t(groupSrc.vertices.data() + groupSrc.vertices.size()) - ptrdiff_t(srcData);
        assert(srcSize >= 0);
        // 使用2维解压器恢复出 glm::vec2 的 UV 坐标
        compression::ArithmeticDeCompressor<uint32_t, 2> decompressor;
        decompressor.init(size_t(srcSize), srcData);
        srcData += decompressor.readVertices(vertexCount, dstData + dstOffset, 2) / sizeof(uint32_t);
        dstOffset += 2 * vertexCount;
      }
      else if(clusterSrc.attributeBits & usedBit)
      {
          // 如果没压缩，直接拷贝。
          // align (对齐): 确保数据写入位置是对齐到 64位(2个32位) 的边界，防止硬件性能降低
        dstOffset = (dstOffset + 1) & ~1;
        memcpy(dstData + dstOffset, srcData, sizeof(glm::vec2) * vertexCount);
        srcData += 2 * vertexCount;
        dstOffset += 2 * vertexCount;
      }
    }
    // 安全检查：确保我们写出来的解压后数据，没有超过目标内存的分配上限
    assert(size_t(dstData + dstOffset) <= size_t(dstWriteOnly) + dstSize);
  }
}
}