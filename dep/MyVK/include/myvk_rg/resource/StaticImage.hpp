#ifndef MYVK_RG_STATIC_IMAGE_HPP
#define MYVK_RG_STATIC_IMAGE_HPP

#include <myvk_rg/RenderGraph.hpp>

namespace myvk_rg {
class StaticImage final : public myvk_rg::ExternalImageBase {
private:
	myvk::Ptr<myvk::ImageView> m_image_view;

public:
	inline StaticImage(myvk_rg::Parent parent, myvk::Ptr<myvk::ImageView> image_view, VkImageLayout layout)
	    : myvk_rg::ExternalImageBase(parent) {
		m_image_view = std::move(image_view);
		SetSrcLayout(layout);
		SetDstLayout(layout);
	}
	inline ~StaticImage() final = default;

	inline const myvk::Ptr<myvk::ImageView> &GetVkImageView() const final { return m_image_view; }
	inline bool IsStatic() const final { return true; }
};
} // namespace myvk_rg

#endif
