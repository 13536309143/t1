//==============================================================================
// 文件：src/core/serialization.hpp
// 模块定位：二进制缓存序列化辅助，提供 span 数据按固定对齐写入和读取的基础操作。
// 数据流：输入是连续内存视图和游标指针；输出是对齐后的缓存布局或恢复出的只读 span。
// 方法说明：缓存文件使用结构化连续布局保存大量数组，避免逐元素序列化开销，同时通过对齐约束满足后续直接映射访问。
// 正确性约束：写入和读取必须使用相同的对齐规则；span 生命周期依赖底层缓存映射，不能越过映射文件生命周期。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
#pragma once


// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <cstdint>
#include <cstring>
#include <cassert>
#include <span>


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace serialization {

static constexpr uint64_t ALIGNMENT  = 16ULL;
static constexpr uint64_t ALIGN_MASK = ALIGNMENT - 1;
static_assert(ALIGNMENT >= sizeof(uint64_t));

template <typename T>


// 函数：getCachedSize。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
inline uint64_t getCachedSize(const std::span<T>& view)
{

  return ((view.size_bytes() + ALIGN_MASK) & ~ALIGN_MASK) + ALIGNMENT;
}

template <typename T>


// 函数：storeAndAdvance。把当前状态写入缓存、缓冲、文件或着色器可消费的数据布局。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：写入路径应明确字节对齐、所有权和可见性，避免后续读取端解释错误。
inline void storeAndAdvance(bool& isValid, uint64_t& dataAddress, uint64_t dataEnd, const std::span<const T>& view)
{
  assert(static_cast<uint64_t>(dataAddress) % ALIGNMENT == 0);

  if(isValid && dataAddress + getCachedSize(view) <= dataEnd)
  {
    union
    {
      uint64_t count;
      uint8_t  countData[ALIGNMENT];
    };
    memset(countData, 0, sizeof(countData));


    count = view.size();


    memcpy(reinterpret_cast<void*>(dataAddress), countData, ALIGNMENT);
    dataAddress += ALIGNMENT;

    if(view.size())
    {

      memcpy(reinterpret_cast<void*>(dataAddress), view.data(), view.size_bytes());
      dataAddress += (view.size_bytes() + ALIGN_MASK) & ~ALIGN_MASK;
    }
  }
  else
  {
    isValid = false;
  }
}

template <typename T>


// 函数：loadAndAdvance。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
inline void loadAndAdvance(bool& isValid, uint64_t& dataAddress, uint64_t dataEnd, std::span<const T>& view)
{
  union
  {
    const T* basePointer;
    uint64_t baseRaw;
  };
  baseRaw = dataAddress;


  assert(dataAddress % ALIGNMENT == 0);

  uint64_t count = *reinterpret_cast<const uint64_t*>(basePointer);
  baseRaw += ALIGNMENT;

  if(isValid && count && (baseRaw + (sizeof(T) * count) <= dataEnd))
  {

    view = std::span<const T>(basePointer, count);
  }
  else
  {
    view = {};

    isValid = isValid && count == 0;
  }

  baseRaw += sizeof(T) * count;

  baseRaw = (baseRaw + ALIGN_MASK) & ~(ALIGN_MASK);

  dataAddress = baseRaw;
}

}
