//==============================================================================
// 文件：src/scene/scene_gltf.cpp
// 模块定位：glTF 导入实现，通过 cgltf 读取几何、材质、相机、节点层级和 meshopt 压缩 缓冲 view。
// 数据流：输入是 glTF/glb 文件和 SceneLoaderConfig；输出是去重后的 GeometryStorage、材质索引、实例矩阵和相机列表。
// 方法说明：glTF 是语义丰富但渲染布局松散的交换格式，本文件将 accessor、缓冲 view 和 node transform 规约为项目内部统一表示。
// 正确性约束：accessor 类型和 stride 必须严格校验；压缩 view 的生命周期要覆盖读取过程；坐标、法线、切线和 UV 量化需与后续压缩一致。
// 注释风格：使用中文解释 CPU 侧语义；保留必要的 API、类型名和数学缩写以便检索。
//==============================================================================
// 依赖说明：引入本编译单元需要的外部库、项目模块和共享着色器布局。
// 依赖顺序通常反映抽象层次：先外部库，再项目模块，最后与 GPU 共享的接口定义。
#include <float.h>
#include <unordered_map>
#include <string>
#include <array>
#include <glm/gtc/type_ptr.hpp>
#include <cgltf.h>
#include <meshoptimizer.h>
#include <nvutils/logger.hpp>
#include <nvutils/file_mapping.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/parallel_work.hpp>
#include "scene.hpp"

namespace {


// 类型：SpinLock。封装本模块的长期状态、资源所有权和对外操作接口。
// 设计意图：通过成员函数集中维护状态转移，避免调用方直接拼接底层资源生命周期。
// 使用约束：实例初始化、每帧使用和释放应遵守声明顺序对应的依赖关系。
class SpinLock
{
public:

  SpinLock(std::atomic_uint32_t& reference)

      : m_reference(reference)
  {
    while(m_reference.exchange(1, std::memory_order_acquire) == 1)
    {

      while(m_reference.load(std::memory_order_relaxed) == 1)
      {

        std::this_thread::yield();
      }
    }
  }

  ~SpinLock() { m_reference.store(0, std::memory_order_release); }

private:
  std::atomic_uint32_t& m_reference;
};


// 结构：FileMappingList。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
// 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
// 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
struct FileMappingList
{


  // 结构：Entry。组织一组语义相关的数据字段，供 CPU/GPU 流程或模块内部逻辑共享。
  // 设计意图：把同一抽象对象的计数、偏移、地址和配置集中存放，降低跨函数传递时的语义丢失。
  // 使用约束：若该结构被着色器或缓存文件读取，字段顺序、对齐方式和默认值都属于接口契约。
  struct Entry
  {
    nvutils::FileReadMapping mapping;
    int64_t                  refCount = 1;
  };
  std::unordered_map<std::string, Entry>       m_nameToMapping;
  std::unordered_map<const void*, std::string> m_dataToName;
#ifndef NDEBUG
  int64_t m_openBias = 0;
#endif


  // 函数：open。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  bool open(const char* path, size_t* size, void** data)
  {
#ifndef NDEBUG
    m_openBias++;
#endif


    // 函数：pathStr。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    std::string pathStr(path);


    auto it = m_nameToMapping.find(pathStr);
    if(it != m_nameToMapping.end())
    {
      *data = const_cast<void*>(it->second.mapping.data());

      *size = it->second.mapping.size();
      it->second.refCount++;
      return true;
    }

    Entry entry;
    if(entry.mapping.open(path))
    {

      const void* mappingData = entry.mapping.data();
      *data                   = const_cast<void*>(mappingData);

      *size                   = entry.mapping.size();
      m_dataToName.insert({mappingData, pathStr});
      m_nameToMapping.insert({pathStr, std::move(entry)});
      return true;
    }

    return false;
  }


  // 函数：close。释放或回收前面初始化的资源，保持生命周期成对管理。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：释放顺序要遵守资源依赖关系，避免 GPU 仍可能访问的对象被提前销毁。
  void close(void* data)
  {
#ifndef NDEBUG
    m_openBias--;
#endif

    auto itName = m_dataToName.find(data);
    if(itName != m_dataToName.end())
    {

      auto itMapping = m_nameToMapping.find(itName->second);
      if(itMapping != m_nameToMapping.end())
      {
        itMapping->second.refCount--;

        if(!itMapping->second.refCount)
        {

          m_nameToMapping.erase(itMapping);

          m_dataToName.erase(itName);
        }
      }
    }
  }


  ~FileMappingList()
  {
#ifndef NDEBUG

    assert(m_openBias == 0 && "open/close bias wrong");
#endif
    assert(m_nameToMapping.empty() && m_dataToName.empty() && "not all opened files were closed");
  }
};

cgltf_result cgltf_read(const struct cgltf_memory_options* memory_options,
                        const struct cgltf_file_options*   file_options,
                        const char*                        path,
                        cgltf_size*                        size,
                        void**                             data)
{
  FileMappingList* mappings = (FileMappingList*)file_options->user_data;
  if(mappings->open(path, size, data))
  {
    return cgltf_result_success;
  }

  return cgltf_result_io_error;
}


// 函数：cgltf_release。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
void cgltf_release(const struct cgltf_memory_options* memory_options, const struct cgltf_file_options* file_options, void* data)
{
  FileMappingList* mappings = (FileMappingList*)file_options->user_data;

  mappings->close(data);
}
using unique_cgltf_ptr = std::unique_ptr<cgltf_data, decltype(&cgltf_free)>;


// 函数：quantizeFloat。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
inline float quantizeFloat(float value, uint32_t dropBits)
{
  union
  {
    uint32_t u32;
    float    f32;
  } un;

  un.f32      = value;
  uint32_t ui = un.u32;

  const int32_t mask  = (1 << (dropBits)) - 1;
  const int32_t round = (1 << (dropBits)) >> 1;
  int32_t  e   = ui & 0x7f800000;
  uint32_t rui = (ui + round) & ~mask;
  ui = e == 0x7f800000 ? ui : rui;

  ui = e == 0 ? 0 : ui;

  un.u32 = ui;
  return un.f32;
}


// 函数：quantizeFloat。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
inline glm::vec2 quantizeFloat(const glm::vec2& vec, uint32_t dropBits)
{
  glm::vec2 res;

  res.x = quantizeFloat(vec.x, dropBits);

  res.y = quantizeFloat(vec.y, dropBits);
  return res;
}


// 函数：quantizeFloat。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
inline glm::vec3 quantizeFloat(const glm::vec3& vec, uint32_t dropBits)
{
  glm::vec3 res;

  res.x = quantizeFloat(vec.x, dropBits);

  res.y = quantizeFloat(vec.y, dropBits);

  res.z = quantizeFloat(vec.z, dropBits);
  return res;
}


// 函数：quantizeFloat。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
inline glm::vec4 quantizeFloat(const glm::vec4& vec, uint32_t dropBits)
{
  glm::vec4 res;

  res.x = quantizeFloat(vec.x, dropBits);

  res.y = quantizeFloat(vec.y, dropBits);

  res.z = quantizeFloat(vec.z, dropBits);

  res.w = quantizeFloat(vec.w, dropBits);
  return res;
}

struct GeometryFingerprint
{
  uint64_t hash0         = 1469598103934665603ull;
  uint64_t hash1         = 1099511628211ull;
  uint64_t triangleCount = 0;
  uint64_t vertexCount   = 0;
  uint32_t attributeBits = 0;

  bool operator==(const GeometryFingerprint& other) const
  {
    return hash0 == other.hash0 && hash1 == other.hash1 && triangleCount == other.triangleCount
           && vertexCount == other.vertexCount && attributeBits == other.attributeBits;
  }
};

struct GeometryFingerprintHasher
{
  size_t operator()(const GeometryFingerprint& fp) const
  {
    uint64_t h = fp.hash0 ^ (fp.hash1 + 0x9e3779b97f4a7c15ull + (fp.hash0 << 6) + (fp.hash0 >> 2));
    h ^= fp.triangleCount + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= fp.vertexCount + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= fp.attributeBits + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return size_t(h);
  }
};

void fingerprintBytes(GeometryFingerprint& fp, const void* data, size_t size)
{
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
  for(size_t i = 0; i < size; ++i)
  {
    fp.hash0 ^= bytes[i];
    fp.hash0 *= 1099511628211ull;
    fp.hash1 ^= uint64_t(bytes[i]) + 0x9e3779b97f4a7c15ull + (fp.hash1 << 6) + (fp.hash1 >> 2);
  }
}

template <typename T>
void fingerprintValue(GeometryFingerprint& fp, const T& value)
{
  fingerprintBytes(fp, &value, sizeof(T));
}

const cgltf_accessor* findAttributeAccessor(const cgltf_primitive* primitive, const char* name)
{
  for(size_t attribIdx = 0; attribIdx < primitive->attributes_count; attribIdx++)
  {
    const cgltf_attribute& attrib = primitive->attributes[attribIdx];
    if(strcmp(attrib.name, name) == 0)
    {
      return attrib.data;
    }
  }

  return nullptr;
}

uint32_t computeMeshAttributeBits(const cgltf_mesh& mesh, const lodclusters::SceneConfig& config)
{
  uint32_t attributeBits = 0;

  for(size_t primIdx = 0; primIdx < mesh.primitives_count; primIdx++)
  {
    const cgltf_primitive* primitive = &mesh.primitives[primIdx];
    if(primitive->type != cgltf_primitive_type_triangles || primitive->attributes_count == 0)
    {
      continue;
    }

    if((config.enabledAttributes & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL) && findAttributeAccessor(primitive, "NORMAL"))
    {
      attributeBits |= shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL;
    }
    if((config.enabledAttributes & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT) && findAttributeAccessor(primitive, "TANGENT"))
    {
      attributeBits |= shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT;
    }
    if((config.enabledAttributes & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0) && findAttributeAccessor(primitive, "TEXCOORD_0"))
    {
      attributeBits |= shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0;
    }
    if((config.enabledAttributes & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1) && findAttributeAccessor(primitive, "TEXCOORD_1"))
    {
      attributeBits |= shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1;
    }
  }

  if(!(attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL))
  {
    attributeBits &= ~shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT;
  }
  if(!(attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0))
  {
    attributeBits &= ~shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT;
    attributeBits &= ~shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1;
  }

  return attributeBits;
}

void fingerprintMissingAccessor(GeometryFingerprint& fp)
{
  uint32_t missing = 0xffffffffu;
  fingerprintValue(fp, missing);
}

void fingerprintCompressedAccessorStorage(GeometryFingerprint& fp, const cgltf_accessor* accessor)
{
  if(!accessor)
  {
    fingerprintMissingAccessor(fp);
    return;
  }

  fingerprintValue(fp, accessor->component_type);
  fingerprintValue(fp, accessor->type);
  fingerprintValue(fp, accessor->count);
  fingerprintValue(fp, accessor->offset);
  fingerprintValue(fp, accessor->stride);

  const cgltf_buffer_view* view = accessor->buffer_view;
  if(!view)
  {
    return;
  }

  if(view->has_meshopt_compression)
  {
    const cgltf_meshopt_compression& mc = view->meshopt_compression;
    fingerprintValue(fp, mc.mode);
    fingerprintValue(fp, mc.filter);
    fingerprintValue(fp, mc.count);
    fingerprintValue(fp, mc.stride);
    fingerprintValue(fp, mc.offset);
    fingerprintValue(fp, mc.size);
    if(mc.buffer && mc.buffer->data)
    {
      const uint8_t* source = reinterpret_cast<const uint8_t*>(mc.buffer->data) + mc.offset;
      fingerprintBytes(fp, source, mc.size);
    }
  }
}

void fingerprintFloatAccessor(GeometryFingerprint&  fp,
                              const cgltf_accessor* accessor,
                              cgltf_type            expectedType,
                              size_t                componentCount,
                              uint32_t              dropBits)
{
  if(!accessor)
  {
    fingerprintMissingAccessor(fp);
    return;
  }

  fingerprintValue(fp, accessor->count);
  fingerprintValue(fp, expectedType);
  fingerprintValue(fp, componentCount);

  if(!accessor->buffer_view || accessor->buffer_view->has_meshopt_compression)
  {
    fingerprintCompressedAccessorStorage(fp, accessor);
    return;
  }

  for(size_t i = 0; i < accessor->count; i++)
  {
    std::array<float, 4> values = {};
    cgltf_accessor_read_float(accessor, i, values.data(), componentCount);
    for(size_t c = 0; c < componentCount; c++)
    {
      float value = accessor->type == expectedType ? values[c] : 0.0f;
      if(dropBits)
      {
        value = quantizeFloat(value, dropBits);
      }
      fingerprintValue(fp, value);
    }
  }
}

GeometryFingerprint computeGeometryFingerprint(const cgltf_mesh& mesh, const lodclusters::SceneConfig& config)
{
  GeometryFingerprint fp;
  fp.attributeBits = computeMeshAttributeBits(mesh, config);

  uint32_t vertexOffset   = 0;
  uint32_t primitiveCount = 0;

  for(size_t primIdx = 0; primIdx < mesh.primitives_count; primIdx++)
  {
    const cgltf_primitive* primitive = &mesh.primitives[primIdx];
    if(primitive->type != cgltf_primitive_type_triangles || primitive->attributes_count == 0)
    {
      continue;
    }

    const cgltf_accessor* positions = findAttributeAccessor(primitive, "POSITION");
    const cgltf_accessor* normals   = findAttributeAccessor(primitive, "NORMAL");
    const cgltf_accessor* tangents  = findAttributeAccessor(primitive, "TANGENT");
    const cgltf_accessor* tex0      = findAttributeAccessor(primitive, "TEXCOORD_0");
    const cgltf_accessor* tex1      = findAttributeAccessor(primitive, "TEXCOORD_1");
    const cgltf_accessor* indices   = primitive->indices;

    uint32_t primitiveVertexCount = positions ? uint32_t(positions->count) : 0;
    uint32_t primitiveIndexCount  = indices ? uint32_t(indices->count) : 0;

    fingerprintValue(fp, primitiveCount);
    fingerprintValue(fp, primitiveVertexCount);
    fingerprintValue(fp, primitiveIndexCount);

    fp.vertexCount += primitiveVertexCount;
    fp.triangleCount += primitiveIndexCount / 3;

    fingerprintFloatAccessor(fp, positions, cgltf_type_vec3, 3, config.useCompressedData ? config.compressionPosDropBits : 0);
    if(fp.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL)
    {
      fingerprintFloatAccessor(fp, normals, cgltf_type_vec3, 3, 0);
    }
    if(fp.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT)
    {
      fingerprintFloatAccessor(fp, tangents, cgltf_type_vec4, 4, 0);
    }
    if(fp.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0)
    {
      fingerprintFloatAccessor(fp, tex0, cgltf_type_vec2, 2, config.useCompressedData ? config.compressionTexDropBits : 0);
    }
    if(fp.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1)
    {
      fingerprintFloatAccessor(fp, tex1, cgltf_type_vec2, 2, config.useCompressedData ? config.compressionTexDropBits : 0);
    }

    if(!indices)
    {
      fingerprintMissingAccessor(fp);
    }
    else if(indices->buffer_view && indices->buffer_view->has_meshopt_compression)
    {
      fingerprintCompressedAccessorStorage(fp, indices);
    }
    else
    {
      fingerprintValue(fp, indices->count);
      for(size_t i = 0; i < indices->count; i++)
      {
        uint32_t index = uint32_t(cgltf_accessor_read_index(indices, i)) + vertexOffset;
        fingerprintValue(fp, index);
      }
    }

    vertexOffset += primitiveVertexCount;
    primitiveCount++;
  }

  fingerprintValue(fp, primitiveCount);
  return fp;
}
}


// 命名空间说明：限制符号可见范围，并表明这些类型和函数属于同一功能域。
// 该边界有助于区分应用层、渲染层、场景层和算法层的职责。
namespace lodclusters {


// 函数：Scene::loadGLTF。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
Scene::Result Scene::loadGLTF(ProcessingInfo& processingInfo, const std::filesystem::path& filePath)
{

  std::string fileName = nvutils::utf8FromPath(filePath);


  cgltf_options options = {};

  FileMappingList mappings;
  options.file.read      = cgltf_read;
  options.file.release   = cgltf_release;
  options.file.user_data = &mappings;

  cgltf_result     cgltfResult;

  unique_cgltf_ptr gltf = unique_cgltf_ptr(nullptr, &cgltf_free);
  {


    cgltf_data* rawData = nullptr;
    cgltfResult         = cgltf_parse_file(&options, fileName.c_str(), &rawData);

    gltf                = unique_cgltf_ptr(rawData, &cgltf_free);
  }

  if(cgltfResult == cgltf_result_legacy_gltf)
  {
    LOGE(
        "loadGLTF: This glTF file is an unsupported legacy file - probably glTF 1.0, while cgltf only supports glTF "
        "2.0 files. Please load a glTF 2.0 file instead.\n");
    return SCENE_RESULT_ERROR;
  }
  else if((cgltfResult != cgltf_result_success) || (gltf == nullptr))
  {
    LOGE("loadGLTF: cgltf_parse_file failed. Is this a valid glTF file? (cgltf result: %d)\n", cgltfResult);
    return SCENE_RESULT_ERROR;
  }


  cgltfResult = cgltf_validate(gltf.get());
  if(cgltfResult != cgltf_result_success)
  {
    LOGE(
        "loadGLTF: The glTF file could be parsed, but cgltf_validate failed. Consider using the glTF Validator at "
        "https://github.khronos.org/glTF-Validator/ to see if the non-displacement parts of the glTF file are correct. "
        "(cgltf result: %d)\n",
        cgltfResult);
    return SCENE_RESULT_ERROR;
  }


  cgltfResult = cgltf_load_buffers(&options, gltf.get(), fileName.c_str());
  if(cgltfResult != cgltf_result_success)
  {
    LOGE(
        "loadGLTF: The glTF file was valid, but cgltf_load_buffers failed. Are the glTF file's referenced file paths "
        "valid? (cgltf result: %d)\n",
        cgltfResult);
    return SCENE_RESULT_ERROR;
  }


  std::vector<size_t> geometryToMesh;
  std::vector<size_t> geometryTriangleCount;
  std::vector<size_t> taskToGeometry;


  // 函数：meshToGeometry。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  std::vector<size_t> meshToGeometry(gltf->meshes_count, -1);

  uint64_t totalTriangleCount = 0;

  {
    size_t geometryMemoryEstimate = 0;

    std::unordered_map<GeometryFingerprint, size_t, GeometryFingerprintHasher> mapMeshToGeometry;

    for(size_t meshIndex = 0; meshIndex < gltf->meshes_count; meshIndex++)
    {
      const cgltf_mesh gltfMesh = gltf->meshes[meshIndex];

      size_t meshMemoryEstimate = 0;
      size_t meshTriangleCount  = 0;

      for(size_t primIdx = 0; primIdx < gltfMesh.primitives_count; primIdx++)
      {
        cgltf_primitive* gltfPrim = &gltfMesh.primitives[primIdx];

        if(gltfPrim->type != cgltf_primitive_type_triangles)
        {
          continue;
        }

        if(gltfPrim->attributes_count == 0)
        {
          continue;
        }


        for(size_t attribIdx = 0; attribIdx < gltfPrim->attributes_count; attribIdx++)
        {
          const cgltf_attribute& gltfAttrib = gltfPrim->attributes[attribIdx];
          const cgltf_accessor*  accessor   = gltfAttrib.data;

          if(strcmp(gltfAttrib.name, "POSITION") == 0)
          {
            meshMemoryEstimate += sizeof(glm::vec4) * accessor->count;
          }
        }

        meshMemoryEstimate += sizeof(uint32_t) * gltfPrim->indices->count;

        meshTriangleCount += gltfPrim->indices->count / 3;
        totalTriangleCount += gltfPrim->indices->count / 3;
      }

      GeometryFingerprint fingerprint = computeGeometryFingerprint(gltfMesh, m_config);
      auto                pair        = mapMeshToGeometry.try_emplace(fingerprint, geometryToMesh.size());
      if(pair.second)
      {

        size_t geometryIndex      = geometryToMesh.size();
        meshToGeometry[meshIndex] = geometryIndex;

        geometryToMesh.push_back(meshIndex);

        taskToGeometry.push_back(geometryIndex);

        geometryTriangleCount.push_back(meshTriangleCount);
        geometryMemoryEstimate += meshMemoryEstimate;
      }
      else
      {
        meshToGeometry[meshIndex] = pair.first->second;
      }
    }
    if(!m_cacheFileView.isValid() && !m_loaderConfig.processingOnly
       && (geometryMemoryEstimate > size_t(m_loaderConfig.forcePreprocessMiB) * 1024 * 1024))
    {
      return SCENE_RESULT_NEEDS_PREPROCESS;
    }
  }

  m_geometryStorages.resize(geometryToMesh.size());
  m_geometryViews.resize(geometryToMesh.size());

  beginProcessingOnly(geometryToMesh.size());

  if(!m_cacheFileView.isValid())
  {

    processingInfo.setupCompressedGltf(gltf->buffer_views_count);
  }
  processingInfo.setupParallelism(geometryToMesh.size(), m_processingOnlyPartialCompleted, m_loaderConfig.processingMode);

  if(processingInfo.numOuterThreads > processingInfo.numInnerThreads)
  {
    std::sort(taskToGeometry.begin(), taskToGeometry.end(),
              [&](size_t l, size_t r) { return geometryTriangleCount[l] > geometryTriangleCount[r]; });
  }

  auto fnLoadAndProcessGeometry = [&](uint64_t taskIndex, uint32_t threadOuterIdx) {
    uint64_t geometryIndex = taskToGeometry[taskIndex];

    size_t meshIndex = geometryToMesh[geometryIndex];

    loadGeometryGLTF(processingInfo, geometryIndex, meshIndex, gltf.get());
  };

  processingInfo.logBegin(m_processingOnlyPartialFile ? 0 : totalTriangleCount);
  if(m_loaderConfig.progressPct)
  {

    m_loaderConfig.progressPct->store(0);
  }

  nvutils::parallel_batches_pooled<1>(geometryToMesh.size(), fnLoadAndProcessGeometry, processingInfo.numOuterThreads);


  processingInfo.logEnd();
  if(m_loaderConfig.progressPct)
  {

    m_loaderConfig.progressPct->store(100);
  }


  bool notCompleted = processingInfo.progressGeometriesCompleted != geometryToMesh.size();
  if(notCompleted)
  {

    LOGW("Error in processing geometries, completed / required mismatch\nTry using `--processingonly 1`\n");
  }
  else
  {

    computeHistogramMaxs();
  }

  if(endProcessingOnly(notCompleted))
  {
    return notCompleted ? SCENE_RESULT_ERROR : SCENE_RESULT_PREPROCESS_COMPLETED;
  }

  if(notCompleted)
  {
    return m_cacheFileView.isValid() ? SCENE_RESULT_CACHE_INVALID : SCENE_RESULT_ERROR;
  }

  if(gltf->scenes_count > 0)
  {
    const cgltf_scene scene = (gltf->scene != nullptr) ? (*(gltf->scene)) : (gltf->scenes[0]);
    for(size_t nodeIdx = 0; nodeIdx < scene.nodes_count; nodeIdx++)
    {
      addInstancesFromNodeGLTF(meshToGeometry, gltf.get(), scene.nodes[nodeIdx]);
    }
  }
  else
  {
    for(size_t nodeIdx = 0; nodeIdx < gltf->nodes_count; nodeIdx++)
    {
      if(gltf->nodes[nodeIdx].parent == nullptr)
      {
        addInstancesFromNodeGLTF(meshToGeometry, gltf.get(), &(gltf->nodes[nodeIdx]));
      }
    }
  }
  if(gltf->cameras_count > 0)
  {
    for(size_t nodeIdx = 0; nodeIdx < gltf->nodes_count; nodeIdx++)
    {
      if(gltf->nodes[nodeIdx].camera != nullptr && gltf->nodes[nodeIdx].camera->type == cgltf_camera_type_perspective)
      {
        Camera cam{};
        cam.fovy = gltf->nodes[nodeIdx].camera->data.perspective.yfov;


        // 函数：worldNodeTransform。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
        // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
        // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
        glm::mat4 worldNodeTransform(1);
        cgltf_node_transform_world(&gltf->nodes[nodeIdx], glm::value_ptr(cam.worldMatrix));

        cam.eye    = glm::vec3(cam.worldMatrix[3]);
        cam.center = (m_bbox.hi + m_bbox.lo) * 0.5f;
        cam.up     = {0, 1, 0};

        m_cameras.push_back(cam);
      }
    }
  }

  return SCENE_RESULT_SUCCESS;
}
void Scene::addInstancesFromNodeGLTF(const std::vector<size_t>& meshToGeometry,
                                     const struct cgltf_data*   data,
                                     const struct cgltf_node*   node,
                                     const glm::mat4            parentObjToWorldTransform)
{
  if(node == nullptr)
    return;


  // 函数：localNodeTransform。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
  // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
  // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
  glm::mat4 localNodeTransform(1);
  cgltf_node_transform_local(node, glm::value_ptr(localNodeTransform));
  const glm::mat4 nodeObjToWorldTransform = parentObjToWorldTransform * localNodeTransform;


  if(node->mesh != nullptr)
  {
    lodclusters::Scene::Instance instance{};
    const ptrdiff_t              meshIndex   = (node->mesh) - data->meshes;
    const cgltf_material*        material    = node->mesh->primitives[0].material;
    bool                         addInstance = true;

    if(material)
    {

      instance.materialID = uint32_t(material - data->materials);
      if(material->unlit || material->has_pbr_metallic_roughness)
      {
        instance.color.x = material->pbr_metallic_roughness.base_color_factor[0];
        instance.color.y = material->pbr_metallic_roughness.base_color_factor[1];
        instance.color.z = material->pbr_metallic_roughness.base_color_factor[2];
        instance.color.w = material->pbr_metallic_roughness.base_color_factor[3];
      }
      else if(material->has_pbr_specular_glossiness)
      {
        instance.color.x = material->pbr_specular_glossiness.diffuse_factor[0];
        instance.color.y = material->pbr_specular_glossiness.diffuse_factor[1];
        instance.color.z = material->pbr_specular_glossiness.diffuse_factor[2];
        instance.color.w = material->pbr_specular_glossiness.diffuse_factor[3];
      }

      if(material->alpha_mode == cgltf_alpha_mode_blend)
      {
        addInstance = false;
      }

      if(material->double_sided)
      {
        instance.twoSided = true;
      }
    }

    if(addInstance)
    {

      instance.geometryID = uint32_t(meshToGeometry[meshIndex]);
      instance.matrix     = nodeObjToWorldTransform;

      m_geometryViews[instance.geometryID].instanceReferenceCount++;

      if(instance.twoSided)
      {
        m_hasTwoSided = true;
      }


      m_instances.push_back(instance);
    }
  }


  const size_t numChildren = node->children_count;
  for(size_t childIdx = 0; childIdx < numChildren; childIdx++)
  {

    addInstancesFromNodeGLTF(meshToGeometry, data, node->children[childIdx], nodeObjToWorldTransform);
  }
}

bool Scene::loadCompressedViewsGLTF(ProcessingInfo&                                processingInfo,
                                    std::unordered_set<struct cgltf_buffer_view*>& compressedViews,
                                    const struct cgltf_data*                       gltf)
{
  static bool warned   = false;
  bool        hadError = false;

  for(cgltf_buffer_view* bufferView : compressedViews)
  {
    size_t bufferViewIndex = bufferView - gltf->buffer_views;


    // 函数：lock。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    SpinLock lock((std::atomic_uint32_t&)processingInfo.bufferViewLocks[bufferViewIndex]);

    uint32_t users = processingInfo.bufferViewUsers[bufferViewIndex];
    if(users == 0)
    {


      cgltf_meshopt_compression* mc = &bufferView->meshopt_compression;

      const unsigned char* source = (const unsigned char*)mc->buffer->data;
      if(!source)
        return false;
      source += mc->offset;


      void* result = malloc(mc->count * mc->stride);
      if(!result)
        return false;

      int  rc   = -1;
      bool warn = false;

      switch(mc->mode)
      {
        case cgltf_meshopt_compression_mode_attributes:
          warn = meshopt_decodeVertexVersion(source, mc->size) != 0;

          rc   = meshopt_decodeVertexBuffer(result, mc->count, mc->stride, source, mc->size);
          break;

        case cgltf_meshopt_compression_mode_triangles:
          warn = meshopt_decodeIndexVersion(source, mc->size) != 1;

          rc   = meshopt_decodeIndexBuffer(result, mc->count, mc->stride, source, mc->size);
          break;

        case cgltf_meshopt_compression_mode_indices:
          warn = meshopt_decodeIndexVersion(source, mc->size) != 1;

          rc   = meshopt_decodeIndexSequence(result, mc->count, mc->stride, source, mc->size);
          break;
      }

      if(rc != 0)
      {

        free(result);
        return false;
      }

      bufferView->data = result;

      if(warn && !warned)
      {
        LOGW("Warning: EXT_meshopt_compression data uses versions outside of the glTF specification (vertex 0 / index 1 expected)\n");
        warned = true;
      }

      switch(mc->filter)
      {
        case cgltf_meshopt_compression_filter_octahedral:

          meshopt_decodeFilterOct(result, mc->count, mc->stride);
          break;

        case cgltf_meshopt_compression_filter_quaternion:

          meshopt_decodeFilterQuat(result, mc->count, mc->stride);
          break;

        case cgltf_meshopt_compression_filter_exponential:

          meshopt_decodeFilterExp(result, mc->count, mc->stride);
          break;

        default:
          break;
      }
    }
    processingInfo.bufferViewUsers[bufferViewIndex] = users + 1;
  }

  return true;
}

void Scene::unloadCompressedViewsGLTF(ProcessingInfo&                                processingInfo,
                                      std::unordered_set<struct cgltf_buffer_view*>& compressedViews,
                                      const struct cgltf_data*                       gltf)
{
  for(cgltf_buffer_view* bufferView : compressedViews)
  {
    size_t bufferViewIndex = bufferView - gltf->buffer_views;


    // 函数：lock。封装本文件中的一段核心逻辑，保持调用方只依赖清晰的接口语义。
    // 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
    // 设计要点：该函数的主要价值在于隔离局部实现细节，使模块边界和调用顺序更容易审查。
    SpinLock lock((std::atomic_uint32_t&)processingInfo.bufferViewLocks[bufferViewIndex]);

    uint32_t users = processingInfo.bufferViewUsers[bufferViewIndex]--;

    if(users == 0)
    {

      free(bufferView->data);
    }
  }
}

template <class T, bool doQuantize, bool doBBox>
inline void readAttributesGLTF(const cgltf_accessor* accessor,
                               float*                writeAttributes,
                               size_t                attributeStride,
                               cgltf_type            expectedType,
                               uint32_t              dropBits = 0,
                               T*                    bboxMin  = nullptr,
                               T*                    bboxMax  = nullptr)
{
  if(accessor->component_type == cgltf_component_type_r_32f && accessor->type == expectedType && accessor->stride == sizeof(T))
  {
    const T* readAttributes = (const T*)(cgltf_buffer_view_data(accessor->buffer_view) + accessor->offset);
    for(size_t i = 0; i < accessor->count; i++)
    {
      T tmp = readAttributes[i];

      if(doQuantize && dropBits)
      {

        tmp = quantizeFloat(tmp, dropBits);
      }

      if(doBBox)
      {

        *bboxMin = glm::min(*bboxMin, tmp);

        *bboxMax = glm::max(*bboxMax, tmp);
      }

      *(T*)&writeAttributes[i * attributeStride] = tmp;
    }
  }
  else
  {
    for(size_t i = 0; i < accessor->count; i++)
    {
      T tmp;
      cgltf_accessor_read_float(accessor, i, &tmp.x, sizeof(T) / sizeof(float));

      if(doQuantize && dropBits)
      {

        tmp = quantizeFloat(tmp, dropBits);
      }

      if(doBBox)
      {

        *bboxMin = glm::min(*bboxMin, tmp);

        *bboxMax = glm::max(*bboxMax, tmp);
      }

      *(T*)&writeAttributes[i * attributeStride] = tmp;
    }
  }
}


// 函数：Scene::loadGeometryGLTF。从文件、缓存、GPU 缓冲或共享布局中读取数据并转换为本模块格式。
// 输入/输出：输入由参数、成员状态或绑定资源提供；输出通常表现为返回值、成员状态更新、GPU 缓冲写入或命令缓冲记录。
// 设计要点：读取路径需要校验输入合法性，并把外部格式的不确定性转化为内部确定布局。
void Scene::loadGeometryGLTF(ProcessingInfo& processingInfo, uint64_t geometryIndex, size_t meshIndex, const struct cgltf_data* gltf)
{


  if(m_processingOnlyPartialFile && m_processingOnlyGeometryOffsets[geometryIndex * 2 + 1])
  {

    uint32_t percentage = processingInfo.logCompletedGeometry();
    if(m_loaderConfig.progressPct)
    {

      m_loaderConfig.progressPct->store(percentage);
    }

    return;
  }

  std::unordered_set<cgltf_buffer_view*> compressedViews;
  bool                                   loadedCompressedViews = false;

  const cgltf_mesh& gltfMesh = gltf->meshes[meshIndex];
  GeometryStorage&  geometry = m_geometryStorages[geometryIndex];
  geometry.bbox              = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}, 0, 0};


  uint32_t triangleCount = 0;
  uint32_t verticesCount = 0;
  for(size_t primIdx = 0; primIdx < gltfMesh.primitives_count; primIdx++)
  {
    cgltf_primitive* gltfPrim = &gltfMesh.primitives[primIdx];

    if(gltfPrim->type != cgltf_primitive_type_triangles)
    {
      continue;
    }


    if(gltfPrim->attributes_count == 0)
    {
      continue;
    }

    for(size_t attribIdx = 0; attribIdx < gltfPrim->attributes_count; attribIdx++)
    {
      const cgltf_attribute& gltfAttrib = gltfPrim->attributes[attribIdx];
      const cgltf_accessor*  accessor   = gltfAttrib.data;

      if(accessor->buffer_view->has_meshopt_compression)

        compressedViews.insert(accessor->buffer_view);

      if(strcmp(gltfAttrib.name, "POSITION") == 0)
      {
        verticesCount += (uint32_t)gltfAttrib.data->count;
      }
      else if((m_config.enabledAttributes & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL) && strcmp(gltfAttrib.name, "NORMAL") == 0)
      {
        m_hasVertexNormals = true;
        geometry.attributeBits |= shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL;
      }
      else if((m_config.enabledAttributes & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT) && strcmp(gltfAttrib.name, "TANGENT") == 0)
      {
        m_hasVertexTangents = true;
        geometry.attributeBits |= shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT;
      }
      else if((m_config.enabledAttributes & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0) && strcmp(gltfAttrib.name, "TEXCOORD_0") == 0)
      {
        m_hasVertexTexCoord0 = true;
        geometry.attributeBits |= shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0;
      }
      else if((m_config.enabledAttributes & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1) && strcmp(gltfAttrib.name, "TEXCOORD_1") == 0)
      {
        m_hasVertexTexCoord1 = true;
        geometry.attributeBits |= shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1;
      }
    }

    if(gltfPrim->indices->buffer_view->has_meshopt_compression)

      compressedViews.insert(gltfPrim->indices->buffer_view);

    triangleCount += (uint32_t)gltfPrim->indices->count / 3;
  }


  memset(&geometry.lodInfo, 0, sizeof(geometry.lodInfo));
  geometry.lodInfo.inputTriangleCount = triangleCount;
  geometry.lodInfo.inputVertexCount   = verticesCount;


  bool isCached = checkCache(geometry.lodInfo, geometryIndex);


  if(m_cacheFileView.isValid() && !isCached)
  {

    LOGW("geometry mismatches scene cache file\n");
    return;
  }
  if(!isCached)
  {
    if(!(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL))
    {
      geometry.attributeBits &= ~shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT;
    }
    if(!(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0))
    {
      geometry.attributeBits &= ~shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT;
    }


    if(!(geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0))
    {
      geometry.attributeBits &= ~shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1;
    }

    size_t attributeStride = (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL ? 3 : 0)
                             + (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT ? 4 : 0)
                             + (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0 ? 2 : 0)
                             + (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1 ? 2 : 0);
    uint32_t attributeStart = 0;

    uint32_t attributeEnd   = uint32_t(attributeStride);


    if((geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL))
    {
      if(m_config.simplifyNormalWeight > 0)
      {
        geometry.attributeNormalOffset = attributeStart;
        attributeStart += 3;
      }
      else
      {
        geometry.attributeNormalOffset = attributeEnd - 3;
        attributeEnd -= 3;
      }
    }

    if((geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0))
    {
      if(m_config.simplifyTexCoordWeight > 0)
      {
        geometry.attributeTex0offset = attributeStart;
        attributeStart += 2;
      }
      else
      {
        geometry.attributeTex0offset = attributeEnd - 2;
        attributeEnd -= 2;
      }
    }

    if((geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1))
    {
      if(m_config.simplifyTexCoordWeight > 0)
      {
        geometry.attributeTex1offset = attributeStart;
        attributeStart += 2;
      }
      else
      {
        geometry.attributeTex1offset = attributeEnd - 2;
        attributeEnd -= 2;
      }
    }

    if((geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT))
    {
      if(m_config.simplifyTangentSignWeight > 0 && m_config.simplifyTangentSignWeight > 0)
      {
        geometry.attributeTangentOffset = attributeStart;
        attributeStart += 4;
      }
      else
      {
        geometry.attributeTangentOffset = attributeEnd - 4;
        attributeEnd -= 4;
      }
    }


    assert(attributeStart == attributeEnd);

    geometry.attributesWithWeights = attributeStart;

    geometry.vertexPositions.resize(verticesCount);

    geometry.vertexAttributes.resize(verticesCount * attributeStride, 0);

    geometry.triangles.resize(triangleCount);


    if(!compressedViews.empty())
    {
      if(!loadCompressedViewsGLTF(processingInfo, compressedViews, gltf))
      {

        LOGW("Error decompressing GLTF\n");
        return;
      }
      loadedCompressedViews = true;
    }


    uint32_t offsetVertices  = 0;
    uint32_t offsetTriangles = 0;

    for(size_t primIdx = 0; primIdx < gltfMesh.primitives_count; primIdx++)
    {
      cgltf_primitive* gltfPrim = &gltfMesh.primitives[primIdx];

      if(gltfPrim->type != cgltf_primitive_type_triangles)
      {
        continue;
      }


      if(gltfPrim->attributes_count == 0)
      {
        continue;
      }

      uint32_t numVertices = 0;

      for(size_t attribIdx = 0; attribIdx < gltfPrim->attributes_count; attribIdx++)
      {
        const cgltf_attribute& gltfAttrib = gltfPrim->attributes[attribIdx];
        const cgltf_accessor*  accessor   = gltfAttrib.data;

        if(strcmp(gltfAttrib.name, "POSITION") == 0)
        {
          glm::vec3* writeVertices = geometry.vertexPositions.data() + offsetVertices;

          readAttributesGLTF<glm::vec3, true, true>(accessor, (float*)writeVertices, 3, cgltf_type_vec3,
                                                    m_config.useCompressedData ? m_config.compressionPosDropBits : 0,
                                                    &geometry.bbox.lo, &geometry.bbox.hi);

          numVertices = (uint32_t)accessor->count;
        }
        else if(strcmp(gltfAttrib.name, "NORMAL") == 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_NORMAL))
        {
          float* writeAttributes = geometry.vertexAttributes.data() + (offsetVertices * attributeStride);
          writeAttributes += geometry.attributeNormalOffset;

          readAttributesGLTF<glm::vec3, false, false>(accessor, writeAttributes, attributeStride, cgltf_type_vec3);

          numVertices = (uint32_t)accessor->count;
        }
        else if(strcmp(gltfAttrib.name, "TANGENT") == 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TANGENT))
        {
          float* writeAttributes = geometry.vertexAttributes.data() + (offsetVertices * attributeStride);
          writeAttributes += geometry.attributeTangentOffset;

          readAttributesGLTF<glm::vec4, false, false>(accessor, writeAttributes, attributeStride, cgltf_type_vec4);
          numVertices = (uint32_t)accessor->count;
        }
        else if(strcmp(gltfAttrib.name, "TEXCOORD_0") == 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_0))
        {
          float* writeAttributes = geometry.vertexAttributes.data() + (offsetVertices * attributeStride);
          writeAttributes += geometry.attributeTex0offset;

          readAttributesGLTF<glm::vec2, true, false>(accessor, writeAttributes, attributeStride, cgltf_type_vec2,
                                                     m_config.useCompressedData ? m_config.compressionTexDropBits : 0);
          numVertices = (uint32_t)accessor->count;
        }
        else if(strcmp(gltfAttrib.name, "TEXCOORD_1") == 0 && (geometry.attributeBits & shaderio::CLUSTER_ATTRIBUTE_VERTEX_TEX_1))
        {
          float* writeAttributes = geometry.vertexAttributes.data() + (offsetVertices * attributeStride);
          writeAttributes += geometry.attributeTex1offset;

          readAttributesGLTF<glm::vec2, true, false>(accessor, writeAttributes, attributeStride, cgltf_type_vec2,
                                                     m_config.useCompressedData ? m_config.compressionTexDropBits : 0);
          numVertices = (uint32_t)accessor->count;
        }
      }


      {
        const cgltf_accessor* accessor = gltfPrim->indices;

        uint32_t* writeIndices = (uint32_t*)(geometry.triangles.data() + offsetTriangles);

        if(offsetVertices == 0 && accessor->component_type == cgltf_component_type_r_32u
           && accessor->type == cgltf_type_scalar && accessor->stride == sizeof(uint32_t))
        {
          memcpy(writeIndices, cgltf_buffer_view_data(accessor->buffer_view) + accessor->offset,
                 sizeof(uint32_t) * accessor->count);
        }
        else
        {
          for(size_t i = 0; i < accessor->count; i++)
          {
            writeIndices[i] = (uint32_t)cgltf_accessor_read_index(gltfPrim->indices, i) + offsetVertices;
          }
        }

        offsetTriangles += (uint32_t)accessor->count / 3;
      }

      offsetVertices += numVertices;
    }
  }


  processGeometry(processingInfo, geometryIndex, isCached);

  if(loadedCompressedViews)
  {

    unloadCompressedViewsGLTF(processingInfo, compressedViews, gltf);
  }


  uint32_t percentage = processingInfo.logCompletedGeometry(triangleCount);
  if(m_loaderConfig.progressPct)
  {

    m_loaderConfig.progressPct->store(percentage);
  }
}
}
