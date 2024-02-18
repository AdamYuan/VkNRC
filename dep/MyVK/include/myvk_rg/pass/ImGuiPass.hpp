#ifdef MYVK_ENABLE_IMGUI
#ifndef MYVK_RG_IMGUIPASS_HPP
#define MYVK_RG_IMGUIPASS_HPP

#include <myvk/ImGuiRenderer.hpp>
#include <myvk_rg/RenderGraph.hpp>

namespace myvk_rg {

class ImGuiPass final : public GraphicsPassBase {
private:
	myvk::Ptr<myvk::ImGuiRenderer> m_imgui_renderer;

public:
	inline ImGuiPass(Parent parent, const Image &image) : GraphicsPassBase(parent) {
		AddColorAttachmentInput<myvk_rg::Usage::kColorAttachmentRW>(0, {"out"}, image);
	}
	inline ~ImGuiPass() final = default;

	inline void CreatePipeline() final {
		m_imgui_renderer = myvk::ImGuiRenderer::Create(GetVkRenderPass(), GetSubpass(), 1);
	}
	inline void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &command_buffer) const final {
		m_imgui_renderer->CmdDrawPipeline(command_buffer, 0);
	}
	inline auto GetImageOutput() { return MakeImageOutput({"out"}); }
};

} // namespace myvk_rg

#endif
#endif
