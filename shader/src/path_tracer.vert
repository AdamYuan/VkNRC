#version 450

layout(push_constant) uniform uuPushConstant { vec3 uOrigin, uLook, uSide, uUp; };

layout(location = 0) out vec3 vDir;

void main() {
	gl_Position = vec4(vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0, 0.0, 1.0);
	vDir = uLook + gl_Position.x * uSide + gl_Position.y * uUp;
}
