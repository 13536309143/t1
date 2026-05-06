//基类和通用接口
#include <random>
#include <algorithm>
#include <vector>
#include <volk.h>
#include <fmt/format.h>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext/scalar_constants.hpp>
#include "renderer.hpp"
#include "../shaders/shaderio.h"


namespace lodclusters {
bool RenderScene::init(Resources* res, const Scene* scene_, const StreamingConfig& streamingConfig_, bool useStreaming_)
{
  scene        = scene_;
  useStreaming = useStreaming_;

  if(useStreaming)
  {
    return sceneStreaming.init(res, scene_, streamingConfig_);
  }
  else
  {
    ScenePreloaded::Config preloadConfig;
    return scenePreloaded.init(res, scene_, preloadConfig);
  }
}
void RenderScene::deinit()
{
  scenePreloaded.deinit();
  sceneStreaming.deinit();
}
void RenderScene::streamingReset()
{
  if(useStreaming)
  {
    sceneStreaming.reset();
  }
}

const nvvk::BufferTyped<shaderio::Geometry>& RenderScene::getShaderGeometriesBuffer() const
{

  if(useStreaming)
    return sceneStreaming.getShaderGeometriesBuffer();
  else
    return scenePreloaded.getShaderGeometriesBuffer();
}



size_t RenderScene::getGeometrySize(bool reserved) const
{
  if(useStreaming)
    return sceneStreaming.getGeometrySize(reserved);
  else
    return scenePreloaded.getGeometrySize();
}

size_t RenderScene::getOperationsSize() const
{
  if(useStreaming)
    return sceneStreaming.getOperationsSize();
  else
    return scenePreloaded.getOperationsSize();
}
bool Renderer::initBasicShaders(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  uint32_t maxPrimitiveOutputs = config.useEXTmeshShader ? res.m_meshShaderPropsEXT.maxMeshOutputPrimitives :
                                                           res.m_meshShaderPropsNV.maxMeshOutputPrimitives;
  uint32_t maxVertexOutputs    = config.useEXTmeshShader ? res.m_meshShaderPropsEXT.maxMeshOutputVertices :
                                                           res.m_meshShaderPropsNV.maxMeshOutputVertices;


  if(config.useEXTmeshShader)
  {
    m_meshShaderWorkgroupSize = std::min(128u, std::min(res.m_meshShaderPropsEXT.maxPreferredMeshWorkGroupInvocations,
                                                        std::min(res.m_meshShaderPropsEXT.maxMeshWorkGroupSize[0],
                                                                 res.m_meshShaderPropsEXT.maxMeshWorkGroupInvocations)));
  }
  else
  {
    m_meshShaderWorkgroupSize = 32u;
  }

  m_meshShaderBoxes =
      std::min(m_meshShaderWorkgroupSize / MESHSHADER_BBOX_THREADS,
               std::min(maxPrimitiveOutputs / MESHSHADER_BBOX_LINES, maxVertexOutputs / MESHSHADER_BBOX_VERTICES));

  shaderc::CompileOptions options = res.makeCompilerOptions();
  options.AddMacroDefinition("USE_EXT_MESH_SHADER", fmt::format("{}", config.useEXTmeshShader ? 1 : 0));
  options.AddMacroDefinition("USE_STREAMING", fmt::format("{}", rscene.useStreaming ? 1 : 0));
  options.AddMacroDefinition("MESHSHADER_WORKGROUP_SIZE", fmt::format("{}", m_meshShaderWorkgroupSize));
  options.AddMacroDefinition("MESHSHADER_BBOX_COUNT", fmt::format("{}", m_meshShaderBoxes));

  res.compileShader(m_basicShaders.fullScreenVertexShader, VK_SHADER_STAGE_VERTEX_BIT, "fullscreen.vert.glsl");
  res.compileShader(m_basicShaders.fullScreenBackgroundFragShader, VK_SHADER_STAGE_FRAGMENT_BIT, "fullscreen_background.frag.glsl");
  res.compileShader(m_basicShaders.fullscreenAtomicRasterFragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT, "fullscreen_atomic.frag.glsl");
  res.compileShader(m_basicShaders.renderInstanceBboxesMeshShader, VK_SHADER_STAGE_MESH_BIT_NV,"render_instance_bbox.mesh.glsl", &options);
  res.compileShader(m_basicShaders.renderInstanceBboxesFragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT, "render_instance_bbox.frag.glsl");
  res.compileShader(m_basicShaders.renderClusterBboxesMeshShader, VK_SHADER_STAGE_MESH_BIT_NV,"render_cluster_bbox.mesh.glsl", &options);
  res.compileShader(m_basicShaders.renderClusterBboxesFragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT, "render_cluster_bbox.frag.glsl");
  if(!res.verifyShaders(m_basicShaders))
  {
    return false;
  }

  return true;
}
void Renderer::initBasics(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  initBasicPipelines(res, rscene, config);

  const Scene& scene = *rscene.scene;

  m_renderInstances.resize(scene.m_instances.size());

  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    shaderio::RenderInstance&  renderInstance = m_renderInstances[i];
    const Scene::Instance&     sceneInstance  = scene.m_instances[i];
    const Scene::GeometryView& geometry       = scene.getActiveGeometry(sceneInstance.geometryID);
    renderInstance                = {};
    renderInstance.worldMatrix    = glm::mat4x3(sceneInstance.matrix);
    renderInstance.worldMatrixI   = glm::mat4x3(glm::inverse(sceneInstance.matrix));
    renderInstance.geometryID     = sceneInstance.geometryID;
    renderInstance.materialID     = uint16_t(sceneInstance.materialID);
    renderInstance.maxLodLevelRcp = geometry.lodLevelsCount > 1 ? 1.0f / float(geometry.lodLevelsCount - 1) : 0.0f;
    renderInstance.packedColor    = glm::packUnorm4x8(sceneInstance.color);
    renderInstance.twoSided       = sceneInstance.twoSided ? 1 : 0;
    renderInstance.flipWinding =
        (!sceneInstance.twoSided && ((glm::determinant(sceneInstance.matrix) <= 0) != config.flipWinding)) ? 1 : 0;
  }

  res.createBuffer(m_renderInstanceBuffer, sizeof(shaderio::RenderInstance) * m_renderInstances.size(),
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  NVVK_DBG_NAME(m_renderInstanceBuffer.buffer);
  res.simpleUploadBuffer(m_renderInstanceBuffer, m_renderInstances.data());
  if(config.useSorting)
  {
    VrdxSorterStorageRequirements sorterRequirements = {};
    vrdxGetSorterKeyValueStorageRequirements(res.m_vrdxSorter, uint32_t(m_renderInstances.size()), &sorterRequirements);

    res.createBuffer(m_sortingAuxBuffer, sorterRequirements.size, sorterRequirements.usage);
    NVVK_DBG_NAME(m_sortingAuxBuffer.buffer);
    m_resourceReservedUsage.operationsMemBytes += logMemoryUsage(m_sortingAuxBuffer.bufferSize, "operations", "traversal sorting");
  }
}

void Renderer::deinitBasics(Resources& res)
{
  res.destroyPipelines(m_basicPipelines);
  vkDestroyPipelineLayout(res.m_device, m_basicPipelineLayout, nullptr);
  m_basicDset.deinit();
  res.m_allocator.destroyBuffer(m_renderInstanceBuffer);
  res.m_allocator.destroyBuffer(m_sortingAuxBuffer);
  m_dirtyRenderInstances.clear();
}

bool Renderer::setInstanceTransform(uint32_t instanceId, const glm::mat4& matrix, bool twoSided)
{
  if(instanceId >= m_renderInstances.size())
  {
    return false;
  }

  shaderio::RenderInstance& renderInstance = m_renderInstances[instanceId];
  renderInstance.worldMatrix               = glm::mat4x3(matrix);
  renderInstance.worldMatrixI              = glm::mat4x3(glm::inverse(matrix));
  renderInstance.twoSided                  = twoSided ? 1 : 0;
  renderInstance.flipWinding =
      (!twoSided && ((glm::determinant(matrix) <= 0) != m_config.flipWinding)) ? 1 : 0;

  if(std::find(m_dirtyRenderInstances.begin(), m_dirtyRenderInstances.end(), instanceId) == m_dirtyRenderInstances.end())
  {
    m_dirtyRenderInstances.push_back(instanceId);
  }
  return true;
}

void Renderer::setInstanceTransforms(const Scene& scene)
{
  const uint32_t count = std::min(uint32_t(scene.m_instances.size()), uint32_t(m_renderInstances.size()));
  for(uint32_t instanceId = 0; instanceId < count; instanceId++)
  {
    const Scene::Instance& instance = scene.m_instances[instanceId];
    setInstanceTransform(instanceId, instance.matrix, instance.twoSided);
  }
}

void Renderer::syncDirtyRenderInstances(VkCommandBuffer cmd)
{
  if(m_dirtyRenderInstances.empty() || !m_renderInstanceBuffer.buffer)
  {
    return;
  }

  std::sort(m_dirtyRenderInstances.begin(), m_dirtyRenderInstances.end());
  m_dirtyRenderInstances.erase(std::unique(m_dirtyRenderInstances.begin(), m_dirtyRenderInstances.end()),
                               m_dirtyRenderInstances.end());

  for(uint32_t instanceId : m_dirtyRenderInstances)
  {
    if(instanceId >= m_renderInstances.size())
    {
      continue;
    }

    vkCmdUpdateBuffer(cmd, m_renderInstanceBuffer.buffer,
                      sizeof(shaderio::RenderInstance) * VkDeviceSize(instanceId),
                      sizeof(shaderio::RenderInstance), &m_renderInstances[instanceId]);
  }

  VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr,
                       0, nullptr);

  m_dirtyRenderInstances.clear();
}

void Renderer::updateBasicDescriptors(Resources& res, RenderScene& rscene, const nvvk::Buffer* sceneBuildBuffer)
{
  nvvk::WriteSetContainer writeSets;
  writeSets.append(m_basicDset.makeWrite(BINDINGS_FRAME_UBO), res.m_commonBuffers.frameConstants);
  writeSets.append(m_basicDset.makeWrite(BINDINGS_READBACK_SSBO), res.m_commonBuffers.readBack);
  writeSets.append(m_basicDset.makeWrite(BINDINGS_RASTER_ATOMIC), res.m_frameBuffer.imgRasterAtomic.descriptor);
  writeSets.append(m_basicDset.makeWrite(BINDINGS_GEOMETRIES_SSBO), rscene.getShaderGeometriesBuffer());
  writeSets.append(m_basicDset.makeWrite(BINDINGS_RENDERINSTANCES_SSBO), m_renderInstanceBuffer);
  if(sceneBuildBuffer)
  {
    writeSets.append(m_basicDset.makeWrite(BINDINGS_SCENEBUILDING_UBO), *sceneBuildBuffer);
  }
  if(rscene.useStreaming)
  {
    writeSets.append(m_basicDset.makeWrite(BINDINGS_STREAMING_UBO), rscene.sceneStreaming.getShaderStreamingBuffer());
  }
  vkUpdateDescriptorSets(res.m_device, writeSets.size(), writeSets.data(), 0, nullptr);
}
void Renderer::initBasicPipelines(Resources& res, RenderScene& rscene, const RendererConfig& config)
{
  m_basicShaderFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT;

  nvvk::DescriptorBindings bindings;
  bindings.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_basicShaderFlags);
  bindings.addBinding(BINDINGS_READBACK_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_basicShaderFlags);
  bindings.addBinding(BINDINGS_RASTER_ATOMIC, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, m_basicShaderFlags);
  bindings.addBinding(BINDINGS_GEOMETRIES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_basicShaderFlags);
  bindings.addBinding(BINDINGS_RENDERINSTANCES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_basicShaderFlags);
  bindings.addBinding(BINDINGS_SCENEBUILDING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_basicShaderFlags);
  if(rscene.useStreaming)
  {
    bindings.addBinding(BINDINGS_STREAMING_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_basicShaderFlags);
  }
  m_basicDset.init(bindings, res.m_device);

  nvvk::createPipelineLayout(res.m_device, &m_basicPipelineLayout, {m_basicDset.getLayout()},
                             {{m_basicShaderFlags, 0, sizeof(uint32_t) * 4}});

  nvvk::GraphicsPipelineCreator graphicsGen;
  nvvk::GraphicsPipelineState   state                = res.m_basicGraphicsState;
  graphicsGen.pipelineInfo.layout                    = m_basicPipelineLayout;
  graphicsGen.renderingState.depthAttachmentFormat   = res.m_frameBuffer.pipelineRenderingInfo.depthAttachmentFormat;
  graphicsGen.renderingState.stencilAttachmentFormat = res.m_frameBuffer.pipelineRenderingInfo.stencilAttachmentFormat;
  graphicsGen.colorFormats                           = {res.m_frameBuffer.colorFormat};
  state.rasterizationState.lineWidth = float(res.m_frameBuffer.pixelScale * 2.0f);
  graphicsGen.clearShaders();
  graphicsGen.addShader(VK_SHADER_STAGE_MESH_BIT_NV, "main",nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.renderInstanceBboxesMeshShader));
  graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main",nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.renderInstanceBboxesFragmentShader));
  graphicsGen.createGraphicsPipeline(res.m_device, nullptr, state, &m_basicPipelines.renderInstanceBboxes);
  graphicsGen.clearShaders();
  state.rasterizationState.lineWidth = float(res.m_frameBuffer.pixelScale);
  graphicsGen.addShader(VK_SHADER_STAGE_MESH_BIT_NV, "main",nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.renderClusterBboxesMeshShader));
  graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main",nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.renderClusterBboxesFragmentShader));
  graphicsGen.createGraphicsPipeline(res.m_device, nullptr, state, &m_basicPipelines.renderClusterBboxes);
  state.depthStencilState.depthWriteEnable = VK_TRUE;
  state.depthStencilState.depthCompareOp   = VK_COMPARE_OP_ALWAYS;
  state.rasterizationState.cullMode        = VK_CULL_MODE_NONE;
  graphicsGen.clearShaders();
  graphicsGen.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main",nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.fullScreenVertexShader));
  graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main",nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.fullScreenBackgroundFragShader));
  graphicsGen.createGraphicsPipeline(res.m_device, nullptr, state, &m_basicPipelines.background);
  graphicsGen.clearShaders();
  graphicsGen.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main",nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.fullScreenVertexShader));
  graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main",nvvkglsl::GlslCompiler::getSpirvData(m_basicShaders.fullscreenAtomicRasterFragmentShader));
  graphicsGen.createGraphicsPipeline(res.m_device, nullptr, state, &m_basicPipelines.atomicRaster);
}
void Renderer::renderInstanceBboxes(VkCommandBuffer cmd, uint32_t selectedInstanceID, bool selectedOnly)
{
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelineLayout, 0, 1, m_basicDset.getSetPtr(), 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelines.renderInstanceBboxes);
  struct PushData
  {
    uint32_t numRenderInstances;
    uint32_t selectedInstanceID;
    uint32_t selectedOnly;
    uint32_t _pad;
  } push = {uint32_t(m_renderInstances.size()), selectedInstanceID, selectedOnly ? 1u : 0u, 0u};
  vkCmdPushConstants(cmd, m_basicPipelineLayout, m_basicShaderFlags, 0, sizeof(push), &push);
  uint32_t numRenderInstances = push.numRenderInstances;
  uint32_t workGroupCount = (numRenderInstances + m_meshShaderBoxes - 1) / m_meshShaderBoxes;
  if(m_config.useEXTmeshShader)
  {
    glm::uvec3 grid = shaderio::fit16bitLaunchGrid(workGroupCount);
    vkCmdDrawMeshTasksEXT(cmd, grid.x, grid.y, grid.z);
  }
  else
  {
    vkCmdDrawMeshTasksNV(cmd, workGroupCount, 0);
  }
}
void Renderer::renderClusterBboxes(VkCommandBuffer cmd, nvvk::Buffer sceneBuildBuffer)
{
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelineLayout, 0, 1, m_basicDset.getSetPtr(), 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelines.renderClusterBboxes);

  if(m_config.useEXTmeshShader)
  {
    vkCmdDrawMeshTasksIndirectEXT(cmd, sceneBuildBuffer.buffer,
                                  offsetof(shaderio::SceneBuilding, indirectDrawClusterBoxesEXT), 1, 0);
  }
  else
  {
    vkCmdDrawMeshTasksIndirectNV(cmd, sceneBuildBuffer.buffer,
                                 offsetof(shaderio::SceneBuilding, indirectDrawClusterBoxesNV), 1, 0);
  }
}
void Renderer::writeAtomicRaster(VkCommandBuffer cmd)
{
  uint32_t dummy = 0;
  vkCmdPushConstants(cmd, m_basicPipelineLayout, m_basicShaderFlags, 0, sizeof(uint32_t), &dummy);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelineLayout, 0, 1, m_basicDset.getSetPtr(), 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelines.atomicRaster);
  vkCmdDraw(cmd, 3, 1, 0, 0);
}
void Renderer::writeBackgroundSky(VkCommandBuffer cmd)
{
  uint32_t dummy = 0;
  vkCmdPushConstants(cmd, m_basicPipelineLayout, m_basicShaderFlags, 0, sizeof(uint32_t), &dummy);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelineLayout, 0, 1, m_basicDset.getSetPtr(), 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicPipelines.background);
  vkCmdDraw(cmd, 3, 1, 0, 0);
}
}
