#pragma once
#ifndef MYVK_RG_KEY_HPP
#define MYVK_RG_KEY_HPP

#include <cinttypes>
#include <concepts>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace myvk_rg::interface {

class PoolKey {
public:
	using LengthType = uint8_t;
	using IDType = uint32_t;
	inline constexpr static const std::size_t kMaxStrLen = 32 - sizeof(LengthType) - sizeof(IDType);

private:
	union {
		struct {
			IDType m_id;
			LengthType m_len;
			char m_str[kMaxStrLen];
		};
		std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> _32_;
	};
	static_assert(sizeof(_32_) == 32);

public:
	inline PoolKey() : _32_{} {}
	template <std::integral IntType = IDType>
	inline PoolKey(std::string_view str, IntType id) : m_str{}, m_len(std::min(str.length(), kMaxStrLen)), m_id(id) {
		std::copy(str.begin(), str.begin() + m_len, m_str);
	}
	inline PoolKey(std::string_view str)
	    : m_str{}, m_len(std::min(str.length(), kMaxStrLen)), m_id{std::numeric_limits<IDType>::max()} {
		std::copy(str.begin(), str.begin() + m_len, m_str);
	}
	inline PoolKey(const PoolKey &r) : _32_{r._32_} {}
	inline PoolKey &operator=(const PoolKey &r) {
		_32_ = r._32_;
		return *this;
	}
	inline std::string_view GetName() const { return std::string_view{m_str, m_len}; }
	inline IDType GetID() const { return m_id; }
	inline void SetName(std::string_view str) {
		m_len = std::min(str.length(), kMaxStrLen);
		std::copy(str.begin(), str.begin() + m_len, m_str);
		std::fill(m_str + m_len, m_str + kMaxStrLen, '\0');
	}
	inline void SetID(IDType id) { m_id = id; }

	inline std::string Format() const {
		return std::string{GetName()} + (m_id == std::numeric_limits<IDType>::max() ? "" : ":" + std::to_string(m_id));
	}

	inline bool operator<(const PoolKey &r) const { return _32_ < r._32_; }
	inline bool operator>(const PoolKey &r) const { return _32_ > r._32_; }
	inline bool operator==(const PoolKey &r) const { return _32_ == r._32_; }
	inline bool operator!=(const PoolKey &r) const { return _32_ != r._32_; }
	struct Hash {
		inline std::size_t operator()(PoolKey const &r) const noexcept {
			return std::get<0>(r._32_) ^ std::get<1>(r._32_) ^ std::get<2>(r._32_) ^ std::get<3>(r._32_);
			// return ((std::get<0>(r._32_) * 37 + std::get<1>(r._32_)) * 37 + std::get<2>(r._32_)) * 37 +
			//        std::get<3>(r._32_);
		}
	};
};
static_assert(sizeof(PoolKey) == 32 && std::is_move_constructible_v<PoolKey>);

class GlobalKey {
private:
	std::vector<PoolKey> m_keys;

public:
	inline GlobalKey() = default;
	inline GlobalKey(std::vector<PoolKey> &&keys) : m_keys{std::move(keys)} {}
	inline GlobalKey(const PoolKey &pool_key) { m_keys.push_back(pool_key); }
	inline GlobalKey(const GlobalKey &global_key, const PoolKey &pool_key) : m_keys(global_key.m_keys) {
		m_keys.push_back(pool_key);
	}
	inline GlobalKey(GlobalKey &&global_key, const PoolKey &pool_key) : m_keys(std::move(global_key.m_keys)) {
		m_keys.push_back(pool_key);
	}
	inline GlobalKey GetPrefix() const {
		return m_keys.empty() ? GlobalKey{} : GlobalKey{{m_keys.begin(), m_keys.end() - 1}};
	}
	inline bool Empty() const { return m_keys.empty(); }
	inline bool operator<(const GlobalKey &r) const { return m_keys < r.m_keys; }
	inline bool operator>(const GlobalKey &r) const { return m_keys > r.m_keys; }
	inline bool operator==(const GlobalKey &r) const { return m_keys == r.m_keys; }
	inline bool operator!=(const GlobalKey &r) const { return m_keys != r.m_keys; }
	inline std::string Format() const {
		auto str =
		    std::accumulate(m_keys.begin(), m_keys.end(), std::string{}, [](std::string &&f, const PoolKey &key) {
			    f += key.Format() + '.';
			    return std::move(f);
		    });
		if (!str.empty())
			str.pop_back();
		return str;
	}
};
static_assert(std::is_move_constructible_v<GlobalKey>);

} // namespace myvk_rg::interface

#endif // MYVK_RG_KEY_HPP
