//
// Created by adamyuan on 2/23/24.
//

#pragma once
#ifndef VKNRC_SOBOL_HPP
#define VKNRC_SOBOL_HPP

#include <algorithm>
#include <cinttypes>
#include <ranges>
#include <vector>

class Sobol {
public:
	inline static constexpr uint32_t kDimension = 64u;

private:
	std::array<uint32_t, kDimension> m_sequence{};
	uint32_t m_index = 0;

public:
	inline void Reset() {
		m_index = 0;
		std::ranges::fill(m_sequence, 0);
	}
	inline const auto &GetU32() const { return m_sequence; }
	inline auto GetFloat() const {
		return GetU32() | std::views::transform([](uint32_t x) -> float { return float(x) / float(0xFFFFFFFFu); });
	}
	void Next();
};

#endif // VKNRC_SOBOL_HPP
