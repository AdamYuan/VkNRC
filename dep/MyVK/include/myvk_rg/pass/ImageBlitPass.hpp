#ifndef MYVK_RG_IMAGE_BLIT_PASS_HPP
#define MYVK_RG_IMAGE_BLIT_PASS_HPP

#include <myvk_rg/RenderGraph.hpp>

namespace myvk_rg {
class ImageBlitPass final : public TransferPassBase {
private:
	VkFilter m_filter{};

public:
	inline ImageBlitPass(Parent parent, const myvk_rg::Image &src, const myvk_rg::Image &dst, VkFilter filter)
	    : TransferPassBase(parent) {
		m_filter = filter;
		AddInput<myvk_rg::Usage::kTransferImageSrc, VK_PIPELINE_STAGE_2_BLIT_BIT>({"src"}, src);
		AddInput<myvk_rg::Usage::kTransferImageDst, VK_PIPELINE_STAGE_2_BLIT_BIT>({"dst"}, dst);
	}
	inline ~ImageBlitPass() final = default;

	inline void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const final {
		command_buffer->CmdBlitImage(GetInputImage({"src"})->GetVkImageView(), GetInputImage({"dst"})->GetVkImageView(),
		                             m_filter);
	}
	inline auto GetDstOutput() { return MakeImageOutput({"dst"}); }
};
} // namespace myvk_rg

#endif
