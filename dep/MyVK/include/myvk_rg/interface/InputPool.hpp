#pragma once
#ifndef MYVK_RG_INPUTPOOL_HPP
#define MYVK_RG_INPUTPOOL_HPP

#include "Input.hpp"
#include "Pool.hpp"

namespace myvk_rg::interface {

template <typename Derived> class InputPool : public Pool<Derived, Variant<BufferInput, ImageInput>> {
private:
	using PoolBase = Pool<Derived, Variant<BufferInput, ImageInput>>;

	template <typename InputType, typename... Args> inline auto add_input(const PoolKey &input_key, Args &&...args) {
		static_cast<ObjectBase *>(static_cast<Derived *>(this))->EmitEvent(Event::kInputChanged);
		return PoolBase::template Construct<InputType>(input_key, std::forward<Args>(args)...);
	}

	template <typename> friend class DescriptorInputSlot;
	template <typename> friend class AttachmentInputSlot;

public:
	inline InputPool() = default;
	inline ~InputPool() override = default;

	inline const auto &GetInputPoolData() const { return PoolBase::GetPoolData(); }

protected:
	template <Usage Usage,
	          typename = std::enable_if_t<!kUsageIsAttachment<Usage> && !kUsageIsDescriptor<Usage> &&
	                                      kUsageHasSpecifiedPipelineStages<Usage> && kUsageForBuffer<Usage>>>
	inline void AddInput(const PoolKey &input_key, const BufferAliasBase &buffer) {
		add_input<BufferInput>(input_key, buffer, Usage, kUsageGetSpecifiedPipelineStages<Usage>);
	}
	template <
	    Usage Usage, VkPipelineStageFlags2 PipelineStageFlags,
	    typename = std::enable_if_t<
	        !kUsageIsAttachment<Usage> && !kUsageIsDescriptor<Usage> && !kUsageHasSpecifiedPipelineStages<Usage> &&
	        (PipelineStageFlags & kUsageGetOptionalPipelineStages<Usage>) == PipelineStageFlags &&
	        kUsageForBuffer<Usage>>>
	inline void AddInput(const PoolKey &input_key, const BufferAliasBase &buffer) {
		add_input<BufferInput>(input_key, buffer, Usage, PipelineStageFlags);
	}
	template <Usage Usage,
	          typename = std::enable_if_t<!kUsageIsAttachment<Usage> && !kUsageIsDescriptor<Usage> &&
	                                      kUsageHasSpecifiedPipelineStages<Usage> && kUsageForImage<Usage>>>
	inline void AddInput(const PoolKey &input_key, const ImageAliasBase &image) {
		add_input<ImageInput>(input_key, image, Usage, kUsageGetSpecifiedPipelineStages<Usage>);
	}
	template <
	    Usage Usage, VkPipelineStageFlags2 PipelineStageFlags,
	    typename = std::enable_if_t<
	        !kUsageIsAttachment<Usage> && !kUsageIsDescriptor<Usage> && !kUsageHasSpecifiedPipelineStages<Usage> &&
	        (PipelineStageFlags & kUsageGetOptionalPipelineStages<Usage>) == PipelineStageFlags &&
	        kUsageForImage<Usage>>>
	inline void AddInput(const PoolKey &input_key, const ImageAliasBase &image) {
		add_input<ImageInput>(input_key, image, Usage, PipelineStageFlags);
	}

	template <typename InputType = InputBase> inline const InputType *GetInput(const PoolKey &input_key) const {
		return PoolBase::template Get<InputType>(input_key);
	}

	inline OutputBufferAlias MakeBufferOutput(const PoolKey &input_key) const {
		return PoolBase::template Get<BufferInput>(input_key)->GetOutput();
	}
	inline OutputImageAlias MakeImageOutput(const PoolKey &input_key) const {
		return PoolBase::template Get<ImageInput>(input_key)->GetOutput();
	}
	inline void ClearInputs() {
		InputPool::Clear();
		static_cast<ObjectBase *>(static_cast<Derived *>(this))->EmitEvent(Event::kInputChanged);
	}
};

template <typename Derived> class DescriptorInputSlot {
private:
	template <typename InputType, typename... Args> inline auto add_input(const PoolKey &input_key, Args &&...args) {
		static_cast<ObjectBase *>(static_cast<Derived *>(this))->EmitEvent(Event::kDescriptorChanged);
		static_assert(std::is_base_of_v<InputPool<Derived>, Derived>);
		return static_cast<InputPool<Derived> *>(static_cast<Derived *>(this))
		    ->template add_input<InputType>(input_key, std::forward<Args>(args)...);
	}
	template <typename> friend class AttachmentInputSlot;

public:
	inline DescriptorInputSlot() = default;
	inline ~DescriptorInputSlot() = default;

protected:
	// Buffer don't specify pipeline stage
	template <Usage Usage,
	          typename = std::enable_if_t<!kUsageIsAttachment<Usage> && kUsageIsDescriptor<Usage> &&
	                                      kUsageHasSpecifiedPipelineStages<Usage> && kUsageForBuffer<Usage>>>
	inline void AddDescriptorInput(DescriptorIndex index, const PoolKey &input_key, const BufferAliasBase &buffer) {
		add_input<BufferInput>(input_key, buffer, Usage, kUsageGetSpecifiedPipelineStages<Usage>, index);
	}

	// Buffer specify pipeline stage
	template <Usage Usage, VkPipelineStageFlags2 PipelineStageFlags,
	          typename = std::enable_if_t<
	              !kUsageIsAttachment<Usage> && kUsageIsDescriptor<Usage> && !kUsageHasSpecifiedPipelineStages<Usage> &&
	              (PipelineStageFlags & kUsageGetOptionalPipelineStages<Usage>) == PipelineStageFlags &&
	              kUsageForBuffer<Usage>>>
	inline void AddDescriptorInput(DescriptorIndex index, const PoolKey &input_key, const BufferAliasBase &buffer) {
		add_input<BufferInput>(input_key, buffer, Usage, PipelineStageFlags, index);
	}

	// Image don't specify pipeline stage
	template <Usage Usage,
	          typename = std::enable_if_t<!kUsageIsAttachment<Usage> && Usage != Usage::kSampledImage &&
	                                      kUsageIsDescriptor<Usage> && kUsageHasSpecifiedPipelineStages<Usage> &&
	                                      kUsageForImage<Usage>>>
	inline void AddDescriptorInput(DescriptorIndex index, const PoolKey &input_key, const ImageAliasBase &image) {
		add_input<ImageInput>(input_key, image, Usage, kUsageGetSpecifiedPipelineStages<Usage>, index);
	}

	// Image specify pipeline stage
	template <Usage Usage, VkPipelineStageFlags2 PipelineStageFlags,
	          typename = std::enable_if_t<!kUsageIsAttachment<Usage> && Usage != Usage::kSampledImage &&
	                                      kUsageIsDescriptor<Usage> && !kUsageHasSpecifiedPipelineStages<Usage> &&
	                                      (PipelineStageFlags & kUsageGetOptionalPipelineStages<Usage>) ==
	                                          PipelineStageFlags &&
	                                      kUsageForImage<Usage>>>
	inline void AddDescriptorInput(DescriptorIndex index, const PoolKey &input_key, const ImageAliasBase &image) {
		add_input<ImageInput>(input_key, image, Usage, PipelineStageFlags, index);
	}

	// Image + sampler don't specify pipeline stage
	template <Usage Usage,
	          typename = std::enable_if_t<Usage == Usage::kSampledImage && kUsageHasSpecifiedPipelineStages<Usage> &&
	                                      kUsageForImage<Usage>>>
	inline void AddDescriptorInput(DescriptorIndex index, const PoolKey &input_key, const ImageAliasBase &image,
	                               const myvk::Ptr<myvk::Sampler> &sampler) {
		add_input<ImageInput>(input_key, image, Usage, kUsageGetSpecifiedPipelineStages<Usage>, index, sampler);
	}

	// Image + sampler specify pipeline stage
	template <Usage Usage, VkPipelineStageFlags2 PipelineStageFlags,
	          typename = std::enable_if_t<Usage == Usage::kSampledImage && !kUsageHasSpecifiedPipelineStages<Usage> &&
	                                      (PipelineStageFlags & kUsageGetOptionalPipelineStages<Usage>) ==
	                                          PipelineStageFlags &&
	                                      kUsageForImage<Usage>>>
	inline void AddDescriptorInput(DescriptorIndex index, const PoolKey &input_key, const ImageAliasBase &image,
	                               const myvk::Ptr<myvk::Sampler> &sampler) {
		add_input<ImageInput>(input_key, image, Usage, PipelineStageFlags, index, sampler);
	}
};

template <typename Derived> class AttachmentInputSlot {
private:
	template <typename InputType, template <typename> typename InputPoolBase, typename... Args>
	inline auto add_input(const PoolKey &input_key, Args &&...args) {
		static_cast<ObjectBase *>(static_cast<Derived *>(this))->EmitEvent(Event::kAttachmentChanged);
		static_assert(std::derived_from<Derived, InputPoolBase<Derived>>);
		return static_cast<InputPoolBase<Derived> *>(static_cast<Derived *>(this))
		    ->template add_input<InputType>(input_key, std::forward<Args>(args)...);
	}

public:
	inline AttachmentInputSlot() = default;
	inline ~AttachmentInputSlot() = default;

protected:
	template <Usage Usage, typename = std::enable_if_t<kUsageIsColorAttachment<Usage>>>
	inline void AddColorAttachmentInput(uint32_t index, const PoolKey &input_key, const ImageAliasBase &image) {
		static_assert(kUsageHasSpecifiedPipelineStages<Usage>);
		add_input<ImageInput, InputPool>(input_key, image, Usage, kUsageGetSpecifiedPipelineStages<Usage>, index);
	}

	inline void AddInputAttachmentInput(uint32_t attachment_index, DescriptorIndex descriptor_index,
	                                    const PoolKey &input_key, const ImageAliasBase &image) {
		static_assert(kUsageHasSpecifiedPipelineStages<Usage::kInputAttachment>);
		add_input<ImageInput, DescriptorInputSlot>(input_key, image, Usage::kInputAttachment,
		                                           kUsageGetSpecifiedPipelineStages<Usage::kInputAttachment>,
		                                           attachment_index, descriptor_index);
	}

	template <Usage Usage, typename = std::enable_if_t<kUsageIsDepthAttachment<Usage>>>
	inline void AddDepthAttachmentInput(const PoolKey &input_key, const ImageAliasBase &image) {
		static_assert(kUsageHasSpecifiedPipelineStages<Usage>);
		add_input<ImageInput, InputPool>(input_key, image, Usage, kUsageGetSpecifiedPipelineStages<Usage>);
	}
};

} // namespace myvk_rg::interface

#endif // MYVK_INPUTPOOL_HPP
