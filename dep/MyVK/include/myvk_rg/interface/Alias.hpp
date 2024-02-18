#pragma once
#ifndef MYVK_RG_ALIAS_HPP
#define MYVK_RG_ALIAS_HPP

#include <cassert>

#include "Key.hpp"
#include "ResourceType.hpp"

namespace myvk_rg::interface {

enum class AliasState : uint8_t { kRaw, kOutput };
#define MAKE_ALIAS_CLASS_VAL(Type, State) uint8_t(static_cast<uint8_t>(State) << 1u | static_cast<uint8_t>(Type))
enum class AliasClass : uint8_t {
	kRawImage = MAKE_ALIAS_CLASS_VAL(ResourceType::kImage, AliasState::kRaw),
	kRawBuffer = MAKE_ALIAS_CLASS_VAL(ResourceType::kBuffer, AliasState::kRaw),
	kOutputImage = MAKE_ALIAS_CLASS_VAL(ResourceType::kImage, AliasState::kOutput),
	kOutputBuffer = MAKE_ALIAS_CLASS_VAL(ResourceType::kBuffer, AliasState::kOutput),
};
inline constexpr AliasClass MakeAliasClass(ResourceType type, AliasState state) {
	return static_cast<AliasClass>(MAKE_ALIAS_CLASS_VAL(type, state));
}
inline constexpr ResourceType GetResourceType(AliasClass res_class) {
	return static_cast<ResourceType>(uint8_t(static_cast<uint8_t>(res_class) & 1u));
}
inline constexpr AliasState GetAliasState(AliasClass res_class) {
	return static_cast<AliasState>(uint8_t(static_cast<uint8_t>(res_class) >> 1u));
}
#undef MAKE_ALIAS_CLASS_VAL

class RawImageAlias;
class RawBufferAlias;

class AliasBase {
private:
	AliasClass m_class{};
	GlobalKey m_source;

public:
	inline virtual ~AliasBase() = default;
	inline AliasBase() = default;
	inline AliasBase(AliasClass alias_class, GlobalKey key) : m_class{alias_class}, m_source{std::move(key)} {}

	inline ResourceType GetType() const { return GetResourceType(m_class); }
	inline AliasState GetState() const { return GetAliasState(m_class); }
	inline AliasClass GetClass() const { return m_class; }
	inline bool Empty() const { return m_source.Empty(); }
	inline explicit operator bool() const { return !Empty(); }

	inline const GlobalKey &GetSourceKey() const { return m_source; }

	template <typename Visitor>
	inline std::invoke_result_t<Visitor, const RawImageAlias *> Visit(Visitor &&visitor) const;
};

class ImageBase;
class ImageInput;
class ImageAliasBase : public AliasBase {
public:
	inline ~ImageAliasBase() override = default;
	inline ImageAliasBase() = default;
	inline ImageAliasBase(AliasState state, GlobalKey source)
	    : AliasBase(MakeAliasClass(ResourceType::kImage, state), std::move(source)) {}

	inline ResourceType GetType() const { return ResourceType::kImage; }

	template <typename Visitor>
	inline std::invoke_result_t<Visitor, const RawImageAlias *> Visit(Visitor &&visitor) const;
};

class RawImageAlias final : public ImageAliasBase {
public:
	inline ~RawImageAlias() final = default;
	inline RawImageAlias() = default;
	RawImageAlias(const ImageBase *image);

	inline ResourceType GetType() const { return ResourceType::kImage; }
	inline AliasState GetState() const { return AliasState::kRaw; }
	inline AliasClass GetClass() const { return AliasClass::kRawImage; }
};

class OutputImageAlias final : public ImageAliasBase {
public:
	inline ~OutputImageAlias() final = default;
	inline OutputImageAlias() = default;
	OutputImageAlias(const ImageInput *image_input);

	inline GlobalKey GetSourcePassKey() const { return GetSourceKey().GetPrefix(); }

	inline ResourceType GetType() const { return ResourceType::kImage; }
	inline AliasState GetState() const { return AliasState::kOutput; }
	inline AliasClass GetClass() const { return AliasClass::kOutputImage; }
};

class BufferBase;
class BufferInput;
class BufferAliasBase : public AliasBase {
public:
	inline ~BufferAliasBase() override = default;
	inline BufferAliasBase() = default;
	inline BufferAliasBase(AliasState state, GlobalKey source)
	    : AliasBase(MakeAliasClass(ResourceType::kBuffer, state), std::move(source)) {}

	inline ResourceType GetType() const { return ResourceType::kBuffer; }

	template <typename Visitor>
	inline std::invoke_result_t<Visitor, const RawBufferAlias *> Visit(Visitor &&visitor) const;
};

class RawBufferAlias final : public BufferAliasBase {
public:
	inline ~RawBufferAlias() final = default;
	inline RawBufferAlias() = default;
	RawBufferAlias(const BufferBase *buffer);

	inline ResourceType GetType() const { return ResourceType::kBuffer; }
	inline AliasState GetState() const { return AliasState::kRaw; }
	inline AliasClass GetClass() const { return AliasClass::kRawBuffer; }
};

class OutputBufferAlias final : public BufferAliasBase {
public:
	inline ~OutputBufferAlias() final = default;
	inline OutputBufferAlias() = default;
	OutputBufferAlias(const BufferInput *buffer_input);

	inline GlobalKey GetSourcePassKey() const { return GetSourceKey().GetPrefix(); }

	inline ResourceType GetType() const { return ResourceType::kBuffer; }
	inline AliasState GetState() const { return AliasState::kOutput; }
	inline AliasClass GetClass() const { return AliasClass::kOutputBuffer; }
};

template <typename Visitor>
std::invoke_result_t<Visitor, const RawImageAlias *> AliasBase::Visit(Visitor &&visitor) const {
	switch (GetClass()) {
	case AliasClass::kRawImage:
		return visitor(static_cast<const RawImageAlias *>(this));
	case AliasClass::kRawBuffer:
		return visitor(static_cast<const RawBufferAlias *>(this));
	case AliasClass::kOutputImage:
		return visitor(static_cast<const OutputImageAlias *>(this));
	case AliasClass::kOutputBuffer:
		return visitor(static_cast<const OutputBufferAlias *>(this));
	}
	assert(false);
	return visitor(static_cast<const RawImageAlias *>(nullptr));
}

template <typename Visitor>
std::invoke_result_t<Visitor, const RawImageAlias *> ImageAliasBase::Visit(Visitor &&visitor) const {
	switch (GetState()) {
	case AliasState::kRaw:
		return visitor(static_cast<const RawImageAlias *>(this));
	case AliasState::kOutput:
		return visitor(static_cast<const OutputImageAlias *>(this));
	}
	assert(false);
	return visitor(static_cast<const RawImageAlias *>(nullptr));
}

template <typename Visitor>
std::invoke_result_t<Visitor, const RawBufferAlias *> BufferAliasBase::Visit(Visitor &&visitor) const {
	switch (GetState()) {
	case AliasState::kRaw:
		return visitor(static_cast<const RawBufferAlias *>(this));
	case AliasState::kOutput:
		return visitor(static_cast<const OutputBufferAlias *>(this));
	}
	assert(false);
	return visitor(static_cast<const RawBufferAlias *>(nullptr));
}

template <typename T>
concept RawAlias = std::same_as<T, RawBufferAlias> || std::same_as<T, RawImageAlias>;

template <typename T>
concept OutputAlias = std::same_as<T, OutputBufferAlias> || std::same_as<T, OutputImageAlias>;

} // namespace myvk_rg::interface

#endif // MYVK_ALIAS_HPP
