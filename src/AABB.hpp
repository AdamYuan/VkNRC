//
// Created by adamyuan on 2/20/24.
//

#pragma once
#ifndef VKNRC_AABB_HPP
#define VKNRC_AABB_HPP

#include <glm/glm.hpp>
#include <numeric>

struct AABB {
	glm::vec3 min, max;

	inline AABB() : min{std::numeric_limits<float>::max()}, max{-std::numeric_limits<float>::max()} {}
	inline AABB(const glm::vec3 &t_min, const glm::vec3 &t_max) : min{t_min}, max{t_max} {}
	inline AABB(const AABB &t_a, const AABB &t_b) : min{glm::min(t_a.min, t_b.min)}, max{glm::max(t_a.max, t_b.max)} {}

	inline void Expand(const glm::vec3 &vec) {
		min = glm::min(vec, min);
		max = glm::max(vec, max);
	}
	inline void Expand(const AABB &aabb) {
		min = glm::min(aabb.min, min);
		max = glm::max(aabb.max, max);
	}
	inline void IntersectAABB(const AABB &aabb) {
		min = glm::max(min, aabb.min);
		max = glm::min(max, aabb.max);
	}
	inline bool Valid() const { return min.x <= max.x && min.y <= max.y && min.z <= max.z; }
	inline glm::vec3 GetCenter() const { return (min + max) * 0.5f; }
	template <int DIM> inline float GetDimCenter() const { return (min[DIM] + max[DIM]) * 0.5f; }
	inline float GetDimCenter(int dim) const { return (min[dim] + max[dim]) * 0.5f; }
	inline glm::vec3 GetExtent() const { return max - min; }
	inline float GetHalfArea() const {
		glm::vec3 extent = GetExtent();
		return (extent.x * (extent.y + extent.z) + extent.y * extent.z);
	}
};

#endif // VKNRC_AABB_HPP
