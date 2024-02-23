#version 450

layout(location = 0) out uvec2 oPrimitiveID_InstanceID;

layout(location = 0) in flat uint vInstanceID;

layout(push_constant) uniform uuPushConstant {
	mat4 uViewProj;
	uint uPrimitiveBase;
};

void main() { oPrimitiveID_InstanceID = uvec2(uPrimitiveBase + gl_PrimitiveID, vInstanceID); }
