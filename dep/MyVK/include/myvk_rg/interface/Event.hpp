#pragma once
#ifndef MYVK_RG_EVENT_HPP
#define MYVK_RG_EVENT_HPP

namespace myvk_rg::interface {

enum class Event {
	kCanvasResized,

	kResultChanged,

	kPassChanged,

	kResourceChanged,

	kRenderAreaChanged,

	kUpdatePipeline,

	kInputChanged,
	kDescriptorChanged,
	kAttachmentChanged,

	kImageResized,
	kImageLoadOpChanged,

	kBufferResized,
	kBufferMapTypeChanged,

	kInitTransferChanged,
	kInitTransferFuncChanged,

	kExternalStageChanged,
	kExternalAccessChanged,
	kExternalImageLayoutChanged,
};

}

#endif // MYVK_EVENT_HPP
