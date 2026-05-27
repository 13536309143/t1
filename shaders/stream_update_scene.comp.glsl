#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#include "shaderio.h"

////////////////////////////////////////////

layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_GEOMETRIES_SSBO, set = 0) buffer geometryBuffer
{
  Geometry geometries[];
};

layout(scalar, binding = BINDINGS_STREAMING_UBO, set = 0) uniform streamingBuffer
{
  SceneStreaming streaming;
};
layout(scalar, binding = BINDINGS_STREAMING_SSBO, set = 0) buffer streamingBufferRW
{
  SceneStreaming streamingRW;
};

////////////////////////////////////////////

layout(local_size_x=STREAM_UPDATE_SCENE_WORKGROUP) in;

////////////////////////////////////////////

void main()
{
  // can load pre-emptively given the array is guaranteed to be sized as multiple of STREAM_UPDATE_SCENE_WORKGROUP
  
  uint threadID = getGlobalInvocationIndex(gl_GlobalInvocationID);  

  // works for both load and unload
  StreamingPatch spatch = streaming.update.patches.d[threadID];
  
  if (threadID < streaming.update.patchGroupsCount)
  {
    uint oldResidentID = 0;
    if (threadID < streaming.update.patchUnloadGroupsCount)
    {
      Group group = Group_in(geometries[spatch.geometryID].streamingGroupAddresses.d[spatch.groupID]).d;
      oldResidentID = group.residentID;
    }
    
    geometries[spatch.geometryID].streamingGroupAddresses.d[spatch.groupID] = spatch.groupAddress;
    
    if (threadID < streaming.update.patchUnloadGroupsCount)
    {
    #if STREAMING_DEBUG_ADDRESSES
      streaming.resident.groups.d[oldResidentID].group = Group_in(STREAMING_INVALID_ADDRESS_START);
    #endif      
    }
    else
    {
      uint loadGroupIndex = threadID - streaming.update.patchUnloadGroupsCount;
      Group_in groupRef = Group_in(spatch.groupAddress);
      Group group = Group_in(groupRef).d;

      uint groupResidentID  = spatch.groupResidentID;
      groupRef.d.residentID = spatch.groupResidentID;
      groupRef.d.clusterResidentID = spatch.clusterResidentID;
    
      StreamingGroup residentGroup;
      residentGroup.geometryID   = spatch.geometryID;
      residentGroup.lodLevel     = spatch.lodLevel;
      residentGroup.age          = uint16_t(0);
      residentGroup.group        = groupRef;
    #if STREAMING_DEBUG_ADDRESSES
      if (uint64_t(streaming.resident.groups.d[groupResidentID].group) < STREAMING_INVALID_ADDRESS_START)
        streamingRW.request.errorUpdate = groupResidentID;
    #endif
      
      // update description in residency table
      streaming.resident.groups.d[groupResidentID] = residentGroup;

      // retain original groupID, used for unloading
      streaming.resident.groupIDs.d[groupResidentID] = spatch.groupID;

      // insert ourselves into the list of all active groups
      streaming.resident.activeGroups.d[streaming.update.loadActiveGroupsOffset + loadGroupIndex] = groupResidentID;
      
      // We might have a bit of divergence here, but shouldn't be a mission critical issue
      
      // All new groups need to build new clusters.
      // These are built into scratch space first, and then moved to final locations.
      
      uint newBuildOffset = spatch.clasBuildOffset;
      for (uint c = 0; c < spatch.clusterCount; c++)
      {
        uint clusterResidentID = spatch.clusterResidentID + c;
        
        Cluster_in clusterRef = Cluster_in(spatch.groupAddress + Group_size + Cluster_size * c);
        streaming.resident.clusters.d[clusterResidentID] = uint64_t(clusterRef);
      }
    }
  }

}

