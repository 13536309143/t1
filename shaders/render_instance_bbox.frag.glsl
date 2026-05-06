#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#include "shaderio.h"


layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(push_constant) uniform pushData
{
  uint numRenderInstances;
  uint selectedInstanceID;
  uint selectedOnly;
  uint _pad;
}
push;

layout(location=0) in Interpolants
{
  flat uint instanceID;
} IN;

layout(location=0,index=0) out vec4 out_Color;
void main()
{
  if(push.selectedOnly != 0 && IN.instanceID != push.selectedInstanceID)
  {
    discard;
  }

  out_Color = IN.instanceID == push.selectedInstanceID ?
              vec4(1.0, 0.86, 0.05, 1.0) :
              unpackUnorm4x8(murmurHash(IN.instanceID)) * 0.9 + 0.1;
  out_Color.w = 1.0;
}
