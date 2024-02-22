#ifndef MYVK_RG_SWAPCHAIN_IMAGE_HPP
#define MYVK_RG_SWAPCHAIN_IMAGE_HPP

#include <myvk_rg/RenderGraph.hpp>

namespace myvk_rg {
#ifdef MYVK_ENABLE_GLFW
class SwapchainImage final : public ExternalImageBase {
private:
	myvk::Ptr<myvk::FrameManager> m_frame_manager;

public:
	inline SwapchainImage(myvk_rg::Parent parent, const myvk::Ptr<myvk::FrameManager> &frame_manager)
	    : ExternalImageBase(parent) {
		m_frame_manager = frame_manager;
		SetSyncType(ExternalSyncType::kCustom);
		SetDstLayout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
	}
	~SwapchainImage() final = default;

	inline const myvk::Ptr<myvk::ImageView> &GetVkImageView() const final {
		return m_frame_manager->GetCurrentSwapchainImageView();
	}
};
#endif
} // namespace myvk_rg

#endif
