#ifndef MYVK_RG_POOL_HPP
#define MYVK_RG_POOL_HPP

#include "Object.hpp"

#include <cinttypes>
#include <cstdio>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>

namespace myvk_rg::interface {

// Value Wrapper
template <typename Type> class Value {
private:
	inline constexpr static bool kUPtr = !std::is_final_v<Type>;

	mutable std::conditional_t<kUPtr, std::unique_ptr<Type>, std::optional<Type>> m_value;

public:
	template <typename TypeToCons>
	inline constexpr static bool kCanConstruct =
	    kUPtr ? std::is_constructible_v<Type *, TypeToCons *> : std::is_same_v<Type, TypeToCons>;
	template <typename TypeToGet>
	inline constexpr static bool kCanGet =
	    kUPtr ? (std::is_base_of_v<Type, TypeToGet> || std::is_base_of_v<TypeToGet, Type> ||
	             std::is_same_v<Type, TypeToGet>)
	          : (std::is_base_of_v<TypeToGet, Type> || std::is_same_v<Type, TypeToGet>);

	template <typename TypeToCons, typename... Args, typename = std::enable_if_t<kCanConstruct<TypeToCons>>>
	inline TypeToCons *Construct(Args &&...args) {
		if constexpr (kUPtr) {
			auto ptr = std::make_unique<TypeToCons>(std::forward<Args>(args)...);
			TypeToCons *ret = ptr.get();
			m_value = std::move(ptr);
			return ret;
		} else {
			m_value = TypeToCons(std::forward<Args>(args)...);
			return std::addressof(*m_value);
		}
	}
	template <typename TypeToGet = Type, typename = std::enable_if_t<kCanGet<TypeToGet>>>
	inline TypeToGet *Get() const {
		Type *ptr = std::addressof(*m_value);
		if constexpr (std::is_same_v<Type, TypeToGet>)
			return ptr;
		else
			return dynamic_cast<TypeToGet *>(ptr);
	}
};

// Variant Wrapper
template <typename... Types> class Variant {
private:
	std::variant<Value<Types>...> m_variant;

	template <std::size_t I> using TypeAt = std::tuple_element_t<I, std::tuple<Types...>>;

	template <typename TypeToCons, size_t I = 0> inline static constexpr size_t GetConstructIndex() {
		if constexpr (I >= sizeof...(Types)) {
			return -1;
		} else {
			if constexpr (Value<TypeAt<I>>::template kCanConstruct<TypeToCons>)
				return I;
			else
				return (GetConstructIndex<TypeToCons, I + 1>());
		}
	}
	template <typename TypeToGet, size_t I = 0> inline static constexpr bool CanGet() {
		if constexpr (I >= sizeof...(Types))
			return false;
		else {
			if constexpr (Value<TypeAt<I>>::template kCanGet<TypeToGet>)
				return true;
			return CanGet<TypeToGet, I + 1>();
		}
	}

public:
	template <typename TypeToCons> inline constexpr static bool kCanConstruct = GetConstructIndex<TypeToCons>() != -1;
	template <typename TypeToGet> inline constexpr static bool kCanGet = CanGet<TypeToGet>();

	template <typename TypeToCons, typename... Args, typename = std::enable_if_t<kCanConstruct<TypeToCons>>>
	inline TypeToCons *Construct(Args &&...args) {
		constexpr auto kIndex = GetConstructIndex<TypeToCons>();
		m_variant.template emplace<Value<TypeAt<kIndex>>>();
		return std::visit(
		    [&](auto &v) -> TypeToCons * {
			    if constexpr (std::decay_t<decltype(v)>::template kCanConstruct<TypeToCons>)
				    return v.template Construct<TypeToCons>(std::forward<Args>(args)...);
			    else
				    return nullptr;
		    },
		    m_variant);
	}
	template <typename TypeToGet, typename = std::enable_if_t<kCanGet<TypeToGet>>> inline TypeToGet *Get() const {
		return std::visit(
		    [](const auto &v) -> TypeToGet * {
			    if constexpr (std::decay_t<decltype(v)>::template kCanGet<TypeToGet>)
				    return v.template Get<TypeToGet>();
			    else
				    return nullptr;
		    },
		    m_variant);
	}
	template <typename Visitor> inline std::invoke_result_t<Visitor, const TypeAt<0> *> Visit(Visitor &&visitor) const {
		return std::visit([&visitor](const auto &v) { return visitor(v.template Get<>()); }, m_variant);
	}
};

// Wrapper
template <typename Type> struct WrapperAux {
	using T = Value<Type>;
};
template <typename... Types> struct WrapperAux<Variant<Types...>> {
	using T = Variant<Types...>;
};
template <typename Type> using Wrapper = typename WrapperAux<Type>::T;

// Pool Data
template <typename Type> using PoolData = std::unordered_map<PoolKey, Wrapper<Type>, PoolKey::Hash>;

template <typename Derived, typename Type> class Pool {
private:
	PoolData<Type> m_data;

public:
	inline Pool() = default;
	inline virtual ~Pool() = default;

protected:
	inline const PoolData<Type> &GetPoolData() const { return m_data; }

	template <typename TypeToCons, typename... Args> inline TypeToCons *Construct(const PoolKey &key, Args &&...args) {
		auto it = m_data.emplace(key, Wrapper<Type>{});
		if constexpr (std::is_base_of_v<ObjectBase, TypeToCons>) {
			static_assert(std::is_base_of_v<ObjectBase, Derived>);
			return it.first->second.template Construct<TypeToCons>(
			    Parent{.p_pool_key = &it.first->first,
			           .p_var_parent = (ObjectBase *)static_cast<const Derived *>(this)},
			    std::forward<Args>(args)...);
		} else
			return it.first->second.template Construct<TypeToCons>(std::forward<Args>(args)...);
	}
	inline bool Exist(const PoolKey &key) const { return m_data.count(key); }
	inline void Delete(const PoolKey &key) { m_data.erase(key); }
	template <typename TypeToGet> inline TypeToGet *Get(const PoolKey &key) const {
		auto it = m_data.find(key);
		return it == m_data.end() ? nullptr : it->second.template Get<TypeToGet>();
	}
	inline void Clear() { m_data.clear(); }
};

} // namespace myvk_rg::interface

#endif
