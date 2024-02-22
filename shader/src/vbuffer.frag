#version 450

layout(location = 0) out uvec2 oPrimitiveID_InstanceID;

layout(location = 0) in flat uint vInstanceID;

void main() { oPrimitiveID_InstanceID = uvec2(gl_PrimitiveID, vInstanceID); }
