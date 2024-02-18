//
// Created by adamyuan on 2/6/24.
//

#pragma once
#ifndef MYVK_RG_EXE_RELATION_HPP
#define MYVK_RG_EXE_RELATION_HPP

#include <cinttypes>
#include <vector>

namespace myvk_rg::executor {

class Relation {
private:
	std::size_t m_count_l{}, m_count_r{}, m_size_r{};
	std::vector<uint64_t> m_bit_matrix;

	inline static constexpr std::size_t BitsetSize(std::size_t bit_count) {
		return (bit_count >> 6u) + ((bit_count & 0x3f) ? 1u : 0u);
	}
	inline static constexpr bool BitsetGet(const uint64_t *data, std::size_t bit_pos) {
		return data[bit_pos >> 6u] & (1ull << (bit_pos & 0x3fu));
	}
	inline static constexpr void BitsetAdd(uint64_t *data, std::size_t bit_pos) {
		data[bit_pos >> 6u] |= (1ull << (bit_pos & 0x3fu));
	}

public:
	inline Relation() = default;
	inline Relation(std::size_t count_l, std::size_t count_r) { Reset(count_l, count_r); }
	inline void Reset(std::size_t count_l, std::size_t count_r) {
		m_count_l = count_l, m_count_r = count_r;
		m_size_r = BitsetSize(count_r);
		m_bit_matrix.clear();
		m_bit_matrix.resize(count_l * m_size_r);
	}
	inline void Add(std::size_t l, std::size_t r) { BitsetAdd(GetRowData(l), r); }
	inline void Apply(std::size_t l_src, std::size_t l_dst) {
		for (std::size_t i = 0; i < m_size_r; ++i)
			GetRowData(l_dst)[i] |= GetRowData(l_src)[i];
	}
	inline bool All(std::size_t l, const uint64_t *r_row_data) const {
		// assert(m_size_r == r_set.m_size)
		for (std::size_t i = 0; i < m_size_r; ++i) {
			if ((GetRowData(l)[i] & r_row_data[i]) != r_row_data[i])
				return false;
		}
		return true;
	}
	inline bool Get(std::size_t l, std::size_t r) const { return BitsetGet(GetRowData(l), r); }

	inline uint64_t *GetRowData(std::size_t l) { return m_bit_matrix.data() + l * m_size_r; }
	inline const uint64_t *GetRowData(std::size_t l) const { return m_bit_matrix.data() + l * m_size_r; }
	inline std::size_t GetRowSize() const { return m_size_r; }

	inline Relation GetInversed() const {
		Relation trans{m_count_r, m_count_l};
		for (std::size_t r = 0; r < m_count_r; ++r)
			for (std::size_t l = 0; l < m_count_l; ++l)
				if (Get(l, r))
					trans.Add(r, l);
		return trans;
	}
};

} // namespace myvk_rg::executor

#endif // MYVK_RG_EXE_RELATION_HPP
