#include "Camera.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>

void Camera::move_forward(float dist, float dir) {
	position.x -= glm::sin(yaw + dir) * dist;
	position.z -= glm::cos(yaw + dir) * dist;
}

bool Camera::DragControl(GLFWwindow *window, Control *p_control, double delta) {
	Camera prev_cam = *this;

	glm::dvec2 cur_pos;
	glfwGetCursorPos(window, &cur_pos.x, &cur_pos.y);

	if (!ImGui::GetCurrentContext()->NavWindow ||
	    (ImGui::GetCurrentContext()->NavWindow->Flags & ImGuiWindowFlags_NoBringToFrontOnFocus)) {
		auto delta_speed = float(delta * p_control->speed);
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			move_forward(delta_speed, 0.0f);
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			move_forward(delta_speed, glm::pi<float>() * 0.5f);
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			move_forward(delta_speed, -glm::pi<float>() * 0.5f);
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			move_forward(delta_speed, glm::pi<float>());
		if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
			position.y += delta_speed;
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
			position.y -= delta_speed;

		if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)) {
			glfwGetCursorPos(window, &cur_pos.x, &cur_pos.y);
			float offset_x = float(cur_pos.x - p_control->prev_cursor_pos.x) * p_control->sensitivity;
			float offset_y = float(cur_pos.y - p_control->prev_cursor_pos.y) * p_control->sensitivity;

			yaw -= offset_x;
			pitch -= offset_y;

			pitch = glm::clamp(pitch, -glm::pi<float>() * 0.5f, glm::pi<float>() * 0.5f);
			yaw = glm::mod(yaw, glm::pi<float>() * 2);
		}
	}
	p_control->prev_cursor_pos = cur_pos;

	return *this != prev_cam;
}

bool Camera::MoveControl(GLFWwindow *window, Control *p_control, double delta) {
	Camera prev_cam = *this;

	auto delta_speed = float(delta * p_control->speed);
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		move_forward(delta_speed, 0.0f);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		move_forward(delta_speed, glm::pi<float>() * 0.5f);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		move_forward(delta_speed, -glm::pi<float>() * 0.5f);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		move_forward(delta_speed, glm::pi<float>());
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
		position.y += delta_speed;
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
		position.y -= delta_speed;

	glm::dvec2 cur_pos;
	glfwGetCursorPos(window, &cur_pos.x, &cur_pos.y);
	float offset_x = float(cur_pos.x - p_control->prev_cursor_pos.x) * p_control->sensitivity;
	float offset_y = float(cur_pos.y - p_control->prev_cursor_pos.y) * p_control->sensitivity;

	yaw -= offset_x;
	pitch -= offset_y;

	pitch = glm::clamp(pitch, -glm::pi<float>() * .5f, glm::pi<float>() * .5f);
	yaw = glm::mod(yaw, glm::pi<float>() * 2.f);

	int w, h;
	glfwGetWindowSize(window, &w, &h);
	glfwSetCursorPos(window, w * 0.5, h * 0.5);

	p_control->prev_cursor_pos = {w * 0.5, h * 0.5};

	return *this != prev_cam;
}
