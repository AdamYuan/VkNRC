#ifndef MYVK_RG_RESOURCE_HPP
#define MYVK_RG_RESOURCE_HPP

#include <myvk/BufferBase.hpp>
#include <myvk/CommandBuffer.hpp>
#include <myvk/FrameManager.hpp>
#include <myvk/ImageView.hpp>

#include "Alias.hpp"
#include "Object.hpp"
#include "ResourceType.hpp"

#include <cassert>
#include <cinttypes>
#include <type_traits>
#include <variant>

namespace myvk_rg {
enum class ExternalSyncType : uint8_t {
	kCustom,   // Sync with the stages and accesses (and layouts) specified
	kLastFrame // Sync with (Load) LastFrame's Resource (LastFrame can be in another RenderGraph)
};
} // namespace myvk_rg

namespace myvk_rg::interface {

enum class ResourceState : uint8_t { kManaged, kCombined, kExternal };
#define MAKE_RESOURCE_CLASS_VAL(Type, State) uint8_t(static_cast<uint8_t>(State) << 1u | static_cast<uint8_t>(Type))
enum class ResourceClass : uint8_t {
	kManagedImage = MAKE_RESOURCE_CLASS_VAL(ResourceType::kImage, ResourceState::kManaged),
	kExternalImageBase = MAKE_RESOURCE_CLASS_VAL(ResourceType::kImage, ResourceState::kExternal),
	kCombinedImage = MAKE_RESOURCE_CLASS_VAL(ResourceType::kImage, ResourceState::kCombined),

	kManagedBuffer = MAKE_RESOURCE_CLASS_VAL(ResourceType::kBuffer, ResourceState::kManaged),
	kExternalBufferBase = MAKE_RESOURCE_CLASS_VAL(ResourceType::kBuffer, ResourceState::kExternal),
	kCombinedBuffer = MAKE_RESOURCE_CLASS_VAL(ResourceType::kBuffer, ResourceState::kCombined),
};
inline constexpr ResourceClass MakeResourceClass(ResourceType type, ResourceState state) {
	return static_cast<ResourceClass>(MAKE_RESOURCE_CLASS_VAL(type, state));
}
inline constexpr ResourceType GetResourceType(ResourceClass res_class) {
	return static_cast<ResourceType>(uint8_t(static_cast<uint8_t>(res_class) & 1u));
}
inline constexpr ResourceState GetResourceState(ResourceClass res_class) {
	return static_cast<ResourceState>(uint8_t(static_cast<uint8_t>(res_class) >> 1u));
}
#undef MAKE_RESOURCE_CLASS_VAL

struct BufferView {
	myvk::Ptr<myvk::BufferBase> buffer;
	VkDeviceSize offset, size;
	myvk::Ptr<myvk::DeviceObjectBase> data;
	inline auto operator<=>(const BufferView &) const = default;
};

class ManagedImage;
class ManagedBuffer;
class ResourceBase : public ObjectBase {
private:
	ResourceClass m_class{};

public:
	inline ~ResourceBase() override = default;
	inline ResourceBase(Parent parent, ResourceClass resource_class) : ObjectBase(parent), m_class{resource_class} {}

	inline ResourceType GetType() const { return GetResourceType(m_class); }
	inline ResourceState GetState() const { return GetResourceState(m_class); }
	inline ResourceClass GetClass() const { return m_class; }

	template <typename Visitor> std::invoke_result_t<Visitor, ManagedImage *> Visit(Visitor &&visitor) const;
};

class BufferBase : public ResourceBase {
public:
	inline constexpr ResourceType GetType() const { return ResourceType::kBuffer; }

	inline ~BufferBase() override = default;
	inline explicit BufferBase(Parent parent, ResourceState state)
	    : ResourceBase(parent, MakeResourceClass(ResourceType::kBuffer, state)) {}

	template <typename Visitor> std::invoke_result_t<Visitor, ManagedBuffer *> inline Visit(Visitor &&visitor) const;
	inline RawBufferAlias Alias() const { return RawBufferAlias{this}; }

	inline const BufferView &GetBufferView() const {
		return Visit([](auto *buffer) -> const BufferView & { return buffer->GetBufferView(); });
	};
};

class ImageBase : public ResourceBase {
public:
	inline constexpr ResourceType GetType() const { return ResourceType::kImage; }

	inline ~ImageBase() override = default;
	inline explicit ImageBase(Parent parent, ResourceState state)
	    : ResourceBase(parent, MakeResourceClass(ResourceType::kImage, state)) {}

	template <typename Visitor> std::invoke_result_t<Visitor, ManagedImage *> inline Visit(Visitor &&visitor) const;
	inline RawImageAlias Alias() const { return RawImageAlias{this}; }

	inline const myvk::Ptr<myvk::ImageView> &GetVkImageView() const {
		return Visit([](auto *image) -> const myvk::Ptr<myvk::ImageView> & { return image->GetVkImageView(); });
	};
};

template <typename Derived> class ImageAttachmentInfo {
private:
	VkAttachmentLoadOp m_load_op{VK_ATTACHMENT_LOAD_OP_DONT_CARE};
	VkClearValue m_clear_value{};

public:
	inline ImageAttachmentInfo() = default;
	inline ImageAttachmentInfo(VkAttachmentLoadOp load_op, const VkClearValue &clear_value)
	    : m_load_op{load_op}, m_clear_value{clear_value} {
		static_assert(std::is_base_of_v<ObjectBase, Derived>);
	}

	inline void SetLoadOp(VkAttachmentLoadOp load_op) {
		if (m_load_op != load_op) {
			m_load_op = load_op;
			static_cast<ObjectBase *>(static_cast<Derived *>(this))->EmitEvent(Event::kImageLoadOpChanged);
		}
	}
	inline void SetClearColorValue(const VkClearColorValue &clear_color_value) {
		m_clear_value.color = clear_color_value;
	}
	inline void SetClearDepthStencilValue(const VkClearDepthStencilValue &clear_depth_stencil_value) {
		m_clear_value.depthStencil = clear_depth_stencil_value;
	}

	inline VkAttachmentLoadOp GetLoadOp() const { return m_load_op; }
	inline const VkClearValue &GetClearValue() const { return m_clear_value; }
};

// External Resources
template <typename Derived> class ExternalResourceInfo {
private:
	VkPipelineStageFlags2 m_src_stages{VK_PIPELINE_STAGE_2_NONE}, m_dst_stages{VK_PIPELINE_STAGE_2_NONE};
	VkAccessFlags2 m_src_accesses{VK_ACCESS_2_NONE}, m_dst_accesses{VK_ACCESS_2_NONE};
	ExternalSyncType m_sync_type{ExternalSyncType::kCustom};

public:
	ExternalResourceInfo() {}
	inline VkPipelineStageFlags2 GetSrcPipelineStages() const { return m_src_stages; }
	inline void SetSrcPipelineStages(VkPipelineStageFlags2 src_stages) {
		if (m_src_stages != src_stages) {
			m_src_stages = src_stages;
			static_cast<ObjectBase *>(static_cast<Derived *>(this))->EmitEvent(Event::kExternalStageChanged);
		}
	}
	inline VkPipelineStageFlags2 GetDstPipelineStages() const { return m_dst_stages; }
	inline void SetDstPipelineStages(VkPipelineStageFlags2 dst_stages) {
		if (m_dst_stages != dst_stages) {
			m_dst_stages = dst_stages;
			static_cast<ObjectBase *>(static_cast<Derived *>(this))->EmitEvent(Event::kExternalStageChanged);
		}
	}
	inline VkAccessFlags2 GetSrcAccessFlags() const { return m_src_accesses; }
	inline void SetSrcAccessFlags(VkAccessFlags2 src_accesses) {
		if (m_src_accesses != src_accesses) {
			m_src_accesses = src_accesses;
			static_cast<ObjectBase *>(static_cast<Derived *>(this))->EmitEvent(Event::kExternalAccessChanged);
		}
	}
	inline VkAccessFlags2 GetDstAccessFlags() const { return m_dst_accesses; }
	inline void SetDstAccessFlags(VkAccessFlags2 dst_accesses) {
		if (m_dst_accesses != dst_accesses) {
			m_dst_accesses = dst_accesses;
			static_cast<ObjectBase *>(static_cast<Derived *>(this))->EmitEvent(Event::kExternalAccessChanged);
		}
	}
	inline ExternalSyncType GetSyncType() const { return m_sync_type; }
	inline void SetSyncType(ExternalSyncType sync_type) {
		if (m_sync_type != sync_type) {
			m_sync_type = sync_type;
			static_cast<ObjectBase *>(static_cast<Derived *>(this))->EmitEvent(Event::kExternalSyncChanged);
		}
	}
};
class ExternalImageBase : public ImageBase,
                          public ImageAttachmentInfo<ExternalImageBase>,
                          public ExternalResourceInfo<ExternalImageBase> {
public:
	inline constexpr ResourceState GetState() const { return ResourceState::kExternal; }
	inline constexpr ResourceClass GetClass() const { return ResourceClass::kExternalImageBase; }

	virtual const myvk::Ptr<myvk::ImageView> &GetVkImageView() const = 0;

	inline ExternalImageBase(Parent parent) : ImageBase(parent, ResourceState::kExternal) {}
	inline ~ExternalImageBase() override = default;

private:
	VkImageLayout m_src_layout{VK_IMAGE_LAYOUT_UNDEFINED}, m_dst_layout{VK_IMAGE_LAYOUT_GENERAL};

public:
	inline VkImageLayout GetSrcLayout() const { return m_src_layout; }
	inline void SetSrcLayout(VkImageLayout src_layout) {
		if (m_src_layout != src_layout) {
			m_src_layout = src_layout;
			EmitEvent(Event::kExternalImageLayoutChanged);
		}
	}
	inline VkImageLayout GetDstLayout() const { return m_dst_layout; }
	inline void SetDstLayout(VkImageLayout dst_layout) {
		if (m_dst_layout != dst_layout) {
			m_dst_layout = dst_layout;
			EmitEvent(Event::kExternalImageLayoutChanged);
		}
	}
};

class ExternalBufferBase : public BufferBase, public ExternalResourceInfo<ExternalBufferBase> {
public:
	inline constexpr ResourceState GetState() const { return ResourceState::kExternal; }
	inline constexpr ResourceClass GetClass() const { return ResourceClass::kExternalBufferBase; }

	virtual const BufferView &GetBufferView() const = 0;

	inline static VkImageLayout GetSrcLayout() { return VK_IMAGE_LAYOUT_UNDEFINED; }
	inline static VkImageLayout GetDstLayout() { return VK_IMAGE_LAYOUT_UNDEFINED; }

	inline ExternalBufferBase(Parent parent) : BufferBase(parent, ResourceState::kExternal) {}
	inline ~ExternalBufferBase() override = default;
};

// Managed Resources
template <typename Derived, typename SizeType> class ManagedResourceInfo {
public:
	using SizeFunc = std::function<SizeType(const VkExtent2D &)>;

private:
	SizeType m_size{};
	SizeFunc m_size_func{};

public:
	inline std::variant<SizeType, SizeFunc> GetSize() const {
		if (m_size_func)
			return m_size_func;
		return m_size;
	}
	template <typename... Args> inline void SetSize(Args &&...args) {
		constexpr Event kResizeEvent =
		    std::is_base_of_v<ImageBase, Derived> ? Event::kImageResized : Event::kBufferResized;

		SizeType size(std::forward<Args>(args)...);
		m_size_func = nullptr;
		if (m_size != size) {
			m_size = size;
			static_cast<ObjectBase *>(static_cast<Derived *>(this))->EmitEvent(kResizeEvent);
		}
	}
	inline void SetSizeFunc(const SizeFunc &func) {
		constexpr Event kResizeEvent =
		    std::is_base_of_v<ImageBase, Derived> ? Event::kImageResized : Event::kBufferResized;

		m_size_func = func;
		static_cast<ObjectBase *>(static_cast<Derived *>(this))->EmitEvent(kResizeEvent);
	}
};

class ManagedBuffer final : public BufferBase, public ManagedResourceInfo<ManagedBuffer, VkDeviceSize> {
private:
	bool m_mapped{false};

public:
	inline constexpr ResourceState GetState() const { return ResourceState::kManaged; }
	inline constexpr ResourceClass GetClass() const { return ResourceClass::kManagedBuffer; }

	inline explicit ManagedBuffer(Parent parent) : BufferBase(parent, ResourceState::kManaged) {}
	inline ManagedBuffer(Parent parent, VkDeviceSize size) : BufferBase(parent, ResourceState::kManaged) {
		SetSize(size);
	}
	~ManagedBuffer() override = default;

	void SetMapped(bool mapped) {
		if (m_mapped != mapped) {
			m_mapped = mapped;
			EmitEvent(Event::kBufferMappedChanged);
		}
	}
	inline bool IsMapped() const { return m_mapped; }

	const BufferView &GetBufferView() const;
	template <typename T> inline T *GetMappedData() { return static_cast<T *>(GetMappedData()); }
	void *GetMappedData() const;
};

class SubImageSize {
private:
	VkExtent3D m_extent{};
	uint32_t m_layers{}, m_base_mip_level{}, m_mip_levels{};

public:
	inline SubImageSize() = default;
	inline explicit SubImageSize(const VkExtent3D &extent, uint32_t layers = 1, uint32_t base_mip_level = 0,
	                             uint32_t mip_levels = 1)
	    : m_extent{extent}, m_layers{layers}, m_base_mip_level{base_mip_level}, m_mip_levels{mip_levels} {}
	inline explicit SubImageSize(const VkExtent2D &extent_2d, uint32_t layers = 1, uint32_t base_mip_level = 0,
	                             uint32_t mip_levels = 1)
	    : m_extent{extent_2d.width, extent_2d.height, 1}, m_layers{layers}, m_base_mip_level{base_mip_level},
	      m_mip_levels{mip_levels} {}
	bool operator==(const SubImageSize &r) const {
		return std::tie(m_extent.width, m_extent.height, m_extent.depth, m_layers, m_base_mip_level, m_mip_levels) ==
		       std::tie(r.m_extent.width, r.m_extent.height, r.m_extent.depth, r.m_layers, r.m_base_mip_level,
		                r.m_mip_levels);
	}
	bool operator!=(const SubImageSize &r) const {
		return std::tie(m_extent.width, m_extent.height, m_extent.depth, m_layers, m_base_mip_level, m_mip_levels) !=
		       std::tie(r.m_extent.width, r.m_extent.height, r.m_extent.depth, r.m_layers, r.m_base_mip_level,
		                r.m_mip_levels);
	}

	inline const VkExtent3D &GetExtent() const { return m_extent; }
	inline uint32_t GetBaseMipLevel() const { return m_base_mip_level; }
	inline uint32_t GetMipLevels() const { return m_mip_levels; }
	inline uint32_t GetArrayLayers() const { return m_layers; }

	inline VkExtent3D GetBaseMipExtent() const {
		return VkExtent3D{
		    .width = std::max(m_extent.width >> m_base_mip_level, 1u),
		    .height = std::max(m_extent.height >> m_base_mip_level, 1u),
		    .depth = std::max(m_extent.depth >> m_base_mip_level, 1u),
		};
	}

	inline bool Merge(const SubImageSize &r) {
		if (m_layers == 0)
			*this = r;
		else {
			if (std::tie(m_extent.width, m_extent.height, m_extent.depth) !=
			    std::tie(r.m_extent.width, r.m_extent.height, r.m_extent.depth))
				return false;

			if (m_layers == r.m_layers && m_base_mip_level + m_mip_levels == r.m_base_mip_level)
				m_mip_levels += r.m_mip_levels; // Merge MipMap
			else if (m_base_mip_level == r.m_base_mip_level && m_mip_levels == r.m_mip_levels)
				m_layers += r.m_layers; // Merge Layer
			else
				return false;
		}
		return true;
	}
};

class ManagedImage final : public ImageBase,
                           public ImageAttachmentInfo<ManagedImage>,
                           public ManagedResourceInfo<ManagedImage, SubImageSize> {
private:
	VkImageViewType m_view_type{};
	VkFormat m_format{};

public:
	inline constexpr ResourceState GetState() const { return ResourceState::kManaged; }
	inline constexpr ResourceClass GetClass() const { return ResourceClass::kManagedImage; }

	inline ManagedImage(Parent parent, VkFormat format, VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D)
	    : ImageBase(parent, ResourceState::kManaged) {
		m_format = format;
		m_view_type = view_type;
		SetCanvasSize();
	}
	~ManagedImage() override = default;

	inline VkImageViewType GetViewType() const { return m_view_type; }
	inline VkFormat GetFormat() const { return m_format; }

	inline void SetSize2D(const VkExtent2D &extent_2d, uint32_t base_mip_level = 0, uint32_t mip_levels = 1) {
		SetSize(extent_2d, (uint32_t)1u, base_mip_level, mip_levels);
	}
	inline void SetSize2DArray(const VkExtent2D &extent_2d, uint32_t layer_count, uint32_t base_mip_level = 0,
	                           uint32_t mip_levels = 1) {
		SetSize(extent_2d, layer_count, base_mip_level, mip_levels);
	}
	inline void SetSize3D(const VkExtent3D &extent_3d, uint32_t base_mip_level = 0, uint32_t mip_levels = 1) {
		SetSize(extent_3d, (uint32_t)1u, base_mip_level, mip_levels);
	}
	inline void SetCanvasSize(uint32_t base_mip_level = 0, uint32_t mip_levels = 1) {
		SetSizeFunc([base_mip_level, mip_levels](const VkExtent2D &extent) {
			return SubImageSize{extent, (uint32_t)1u, base_mip_level, mip_levels};
		});
	}

	const myvk::Ptr<myvk::ImageView> &GetVkImageView() const;
};

class CombinedImage final : public ImageBase {
private:
	VkImageViewType m_view_type;
	std::vector<OutputImageAlias> m_images;

public:
	inline constexpr ResourceState GetState() const { return ResourceState::kCombined; }
	inline constexpr ResourceClass GetClass() const { return ResourceClass::kCombinedImage; }

	inline VkImageViewType GetViewType() const { return m_view_type; }

	const myvk::Ptr<myvk::ImageView> &GetVkImageView() const;

	inline CombinedImage(Parent parent, VkImageViewType view_type, std::vector<OutputImageAlias> &&images)
	    : ImageBase(parent, ResourceState::kCombined), m_images(std::move(images)), m_view_type(view_type) {}
	inline const std::vector<OutputImageAlias> &GetSubAliases() const { return m_images; }
	~CombinedImage() override = default;
};

class CombinedBuffer final : public BufferBase {
private:
	std::vector<OutputBufferAlias> m_buffers;

public:
	inline constexpr ResourceState GetState() const { return ResourceState::kCombined; }
	inline constexpr ResourceClass GetClass() const { return ResourceClass::kCombinedBuffer; }

	const BufferView &GetBufferView() const;

	template <typename T> inline T *GetMappedData() { return static_cast<T *>(GetMappedData()); }
	void *GetMappedData() const;

	inline CombinedBuffer(Parent parent, std::vector<OutputBufferAlias> &&buffers)
	    : BufferBase(parent, ResourceState::kCombined), m_buffers(std::move(buffers)) {}
	inline const std::vector<OutputBufferAlias> &GetSubAliases() const { return m_buffers; }
	~CombinedBuffer() override = default;
};

template <typename Visitor> std::invoke_result_t<Visitor, ManagedImage *> ResourceBase::Visit(Visitor &&visitor) const {
	switch (m_class) {
	case ResourceClass::kManagedImage:
		return visitor(static_cast<const ManagedImage *>(this));
	case ResourceClass::kExternalImageBase:
		return visitor(static_cast<const ExternalImageBase *>(this));
	case ResourceClass::kCombinedImage:
		return visitor(static_cast<const CombinedImage *>(this));

	case ResourceClass::kManagedBuffer:
		return visitor(static_cast<const ManagedBuffer *>(this));
	case ResourceClass::kExternalBufferBase:
		return visitor(static_cast<const ExternalBufferBase *>(this));
	case ResourceClass::kCombinedBuffer:
		return visitor(static_cast<const CombinedBuffer *>(this));
	}
	assert(false);
	return visitor(static_cast<const ManagedImage *>(nullptr));
}
template <typename Visitor> std::invoke_result_t<Visitor, ManagedImage *> ImageBase::Visit(Visitor &&visitor) const {
	switch (GetState()) {
	case ResourceState::kManaged:
		return visitor(static_cast<const ManagedImage *>(this));
	case ResourceState::kExternal:
		return visitor(static_cast<const ExternalImageBase *>(this));
	case ResourceState::kCombined:
		return visitor(static_cast<const CombinedImage *>(this));
	}
	assert(false);
	return visitor(static_cast<const ManagedImage *>(nullptr));
}
template <typename Visitor> std::invoke_result_t<Visitor, ManagedBuffer *> BufferBase::Visit(Visitor &&visitor) const {
	switch (GetState()) {
	case ResourceState::kManaged:
		return visitor(static_cast<const ManagedBuffer *>(this));
	case ResourceState::kExternal:
		return visitor(static_cast<const ExternalBufferBase *>(this));
	case ResourceState::kCombined:
		return visitor(static_cast<const CombinedBuffer *>(this));
	default:
		assert(false);
	}
	return visitor(static_cast<const ManagedBuffer *>(nullptr));
}

template <typename T>
concept ImageResource = std::derived_from<T, ImageBase>;
template <typename T>
concept BufferResource = std::derived_from<T, BufferBase>;

template <typename T>
concept ExternalResource = std::derived_from<T, ExternalResourceInfo<T>>;

template <typename T>
concept ManagedResource = std::same_as<T, ManagedImage> || std::same_as<T, ManagedBuffer>;

template <typename T>
concept CombinedResource = std::same_as<T, CombinedImage> || std::same_as<T, CombinedBuffer>;

template <typename T>
concept InternalImage = std::same_as<T, ManagedImage> || std::same_as<T, CombinedImage>;
template <typename T>
concept InternalBuffer = std::same_as<T, ManagedBuffer> || std::same_as<T, CombinedBuffer>;
template <typename T>
concept InternalResource = InternalImage<T> || InternalBuffer<T>;

template <typename T>
concept AttachmentImage = std::derived_from<T, ImageAttachmentInfo<T>>;

} // namespace myvk_rg::interface

#endif
