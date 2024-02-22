#ifdef MYVK_ENABLE_RG

#ifndef MYVK_RG_RENDER_GRAPH_HPP
#define MYVK_RG_RENDER_GRAPH_HPP

#include "interface/RenderGraph.hpp"

namespace myvk_rg {

using Key = interface::PoolKey;
using Parent = interface::Parent;

using BufferView = interface::BufferView;

using GraphicsPassBase = interface::GraphicsPassBase;
using ComputePassBase = interface::ComputePassBase;
using TransferPassBase = interface::TransferPassBase;
using PassGroupBase = interface::PassGroupBase;

using Image = interface::ImageAliasBase;
using ImageOutput = interface::OutputImageAlias;
using ManagedImage = interface::ManagedImage;
using CombinedImage = interface::CombinedImage;
using ExternalImageBase = interface::ExternalImageBase;

using Buffer = interface::BufferAliasBase;
using BufferOutput = interface::OutputBufferAlias;
using ManagedBuffer = interface::ManagedBuffer;
using CombinedBuffer = interface::CombinedBuffer;
using ExternalBufferBase = interface::ExternalBufferBase;

using SubImageSize = interface::SubImageSize;
using RenderPassArea = interface::RenderPassArea;

using DescriptorIndex = interface::DescriptorIndex;

using RenderGraphBase = interface::RenderGraphBase;

} // namespace myvk_rg

#endif

#endif