#ifndef HYPERCRAFT_CLIENT_CAMERA_HPP
#define HYPERCRAFT_CLIENT_CAMERA_HPP

#include <myvk/Buffer.hpp>
#include <myvk/DescriptorSet.hpp>

#include <cinttypes>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct GLFWwindow;

class Camera {
public:
	glm::vec3 position{0.0f, 0.0f, 0.0f};
	float yaw{0.0f}, pitch{0.0f};
	float fov{glm::pi<float>() / 3.f};

	inline bool operator==(const Camera &) const = default;

	struct VkLookSideUp {
		glm::vec3 look, side, up;
	};
	struct Control {
		float sensitivity, speed;
		glm::dvec2 prev_cursor_pos;
	};

private:
	void move_forward(float dist, float dir);

public:
	bool DragControl(GLFWwindow *window, Control *p_control, double delta);
	bool MoveControl(GLFWwindow *window, Control *p_control, double delta);

	inline glm::mat4 GetVkViewProjection(float aspect_ratio, float near, float far) const {
		glm::mat4 ret = glm::perspectiveZO(fov, aspect_ratio, near, far);
		ret[1][1] *= -1;
		ret = glm::rotate(ret, -pitch, glm::vec3(1.0f, 0.0f, 0.0f));
		ret = glm::rotate(ret, -yaw, glm::vec3(0.0f, 1.0f, 0.0f));
		ret = glm::translate(ret, -position);
		return ret;
	}
	inline glm::vec3 GetLook() const {
		float xz_len = glm::cos(pitch);
		return glm::vec3{xz_len * glm::sin(yaw), glm::sin(pitch), xz_len * glm::cos(yaw)};
	}
	inline VkLookSideUp GetVkLookSideUp(float aspect_ratio) const {
		auto trans = glm::identity<glm::mat4>();
		trans = glm::rotate(trans, yaw, glm::vec3(0.0f, 1.0f, 0.0f));
		trans = glm::rotate(trans, pitch, glm::vec3(1.0f, 0.0f, 0.0f));
		float tg = glm::tan(fov * 0.5f);
		glm::vec3 look = -trans[2];
		glm::vec3 side = trans[0];
		look = glm::normalize(look);
		side = glm::normalize(side) * tg * aspect_ratio;
		glm::vec3 up = glm::normalize(glm::cross(look, side)) * tg;

		return {.look = look, .side = side, .up = up};
	}
};

#endif
