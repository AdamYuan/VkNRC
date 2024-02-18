#ifndef MYVK_RG_RESOURCE_IO_HPP
#define MYVK_RG_RESOURCE_IO_HPP

#include <cassert>
#include <optional>

#include "Alias.hpp"
#include "Object.hpp"
#include "Usage.hpp"

#include "myvk/DescriptorSet.hpp"
#include "myvk/Sampler.hpp"

namespace myvk_rg::interface {

struct DescriptorIndex {
	uint32_t binding, array_element;
	inline auto operator<=>(const DescriptorIndex &r) const = default;
};

class ImageInput;
class BufferInput;

class InputBase : public ObjectBase {
private:
	AliasBase m_input_alias{};
	Usage m_usage{};
	VkPipelineStageFlags2 m_pipeline_stages{};
	std::optional<DescriptorIndex> m_opt_descriptor_index;

public:
	inline InputBase(Parent parent, const AliasBase &input_alias, Usage usage, VkPipelineStageFlags2 pipeline_stages,
	                 std::optional<DescriptorIndex> opt_descriptor_index)
	    : ObjectBase(parent), m_input_alias{input_alias}, m_usage{usage}, m_pipeline_stages{pipeline_stages},
	      m_opt_descriptor_index{opt_descriptor_index} {}
	inline ~InputBase() override = default;
	inline Usage GetUsage() const { return m_usage; }
	inline VkPipelineStageFlags2 GetPipelineStages() const { return m_pipeline_stages; }
	inline const auto &GetOptDescriptorIndex() const { return m_opt_descriptor_index; }

	inline const AliasBase &GetInputAlias() const { return m_input_alias; }
	inline ResourceType GetType() const { return m_input_alias.GetType(); }

	template <typename Visitor> inline std::invoke_result_t<Visitor, ImageInput *> Visit(Visitor &&visitor) const;
};

class ImageInput final : public InputBase {
private:
	std::optional<uint32_t> m_opt_attachment_index{};
	myvk::Ptr<myvk::Sampler> m_sampler{};

public:
	inline ImageInput(Parent parent, const ImageAliasBase &resource, Usage usage, VkPipelineStageFlags2 pipeline_stages)
	    : InputBase(parent, resource, usage, pipeline_stages, std::nullopt) {}
	// Descriptor Index
	inline ImageInput(Parent parent, const ImageAliasBase &resource, Usage usage, VkPipelineStageFlags2 pipeline_stages,
	                  DescriptorIndex descriptor_index)
	    : InputBase(parent, resource, usage, pipeline_stages, descriptor_index) {}
	inline ImageInput(Parent parent, const ImageAliasBase &resource, Usage usage, VkPipelineStageFlags2 pipeline_stages,
	                  DescriptorIndex descriptor_index, myvk::Ptr<myvk::Sampler> sampler)
	    : InputBase(parent, resource, usage, pipeline_stages, descriptor_index), m_sampler{std::move(sampler)} {}
	// Attachment Index
	inline ImageInput(Parent parent, const ImageAliasBase &resource, Usage usage, VkPipelineStageFlags2 pipeline_stages,
	                  uint32_t attachment_index)
	    : InputBase(parent, resource, usage, pipeline_stages, std::nullopt), m_opt_attachment_index{attachment_index} {}
	inline ImageInput(Parent parent, const ImageAliasBase &resource, Usage usage, VkPipelineStageFlags2 pipeline_stages,
	                  uint32_t attachment_index, DescriptorIndex descriptor_index)
	    : InputBase(parent, resource, usage, pipeline_stages, descriptor_index),
	      m_opt_attachment_index{attachment_index} {}
	inline ~ImageInput() final = default;

	inline const ImageAliasBase &GetInputAlias() const {
		return static_cast<const ImageAliasBase &>(InputBase::GetInputAlias());
	}
	inline static ResourceType GetType() { return ResourceType::kImage; }
	inline const auto &GetOptAttachmentIndex() const { return m_opt_attachment_index; }
	inline const auto &GetVkSampler() const { return m_sampler; }
	inline auto GetOutput() const { return OutputImageAlias(this); }
};

class BufferInput final : public InputBase {
public:
	inline BufferInput(Parent parent, const BufferAliasBase &resource, Usage usage,
	                   VkPipelineStageFlags2 pipeline_stages)
	    : InputBase(parent, resource, usage, pipeline_stages, std::nullopt) {}
	inline BufferInput(Parent parent, const BufferAliasBase &resource, Usage usage,
	                   VkPipelineStageFlags2 pipeline_stages, DescriptorIndex descriptor_index)
	    : InputBase(parent, resource, usage, pipeline_stages, descriptor_index) {}
	inline ~BufferInput() final = default;

	inline const BufferAliasBase &GetInputAlias() const {
		return static_cast<const BufferAliasBase &>(InputBase::GetInputAlias());
	}
	inline static ResourceType GetType() { return ResourceType::kBuffer; }
	inline auto GetOutput() const { return OutputBufferAlias(this); }
};

template <typename Visitor> std::invoke_result_t<Visitor, ImageInput *> InputBase::Visit(Visitor &&visitor) const {
	switch (GetType()) {
	case ResourceType::kImage:
		return visitor(static_cast<const ImageInput *>(this));
	case ResourceType::kBuffer:
		return visitor(static_cast<const BufferInput *>(this));
	default:
		assert(false);
	}
	return visitor(static_cast<const BufferInput *>(nullptr));
}

} // namespace myvk_rg::interface

#endif
