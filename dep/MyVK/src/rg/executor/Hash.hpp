//
// Created by adamyuan on 2/15/24.
//

#pragma once
#ifndef MYVK_HASH_HPP
#define MYVK_HASH_HPP

#include <cinttypes>
#include <functional>

namespace myvk_rg::executor {

template <typename T, uint32_t T::*First, uint32_t T::*Second> struct U32PairHash {
	inline std::size_t operator()(T x) const {
		uint64_t u = (uint64_t(x.*First) << uint64_t(32)) | uint64_t(x.*Second);
		return std::hash<uint64_t>{}(u);
	}
};
} // namespace myvk_rg::executor

#endif // MYVK_HASH_HPP
