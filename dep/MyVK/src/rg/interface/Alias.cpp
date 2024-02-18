#include <myvk_rg/interface/Alias.hpp>

#include <myvk_rg/interface/Input.hpp>
#include <myvk_rg/interface/Resource.hpp>

namespace myvk_rg::interface {

RawImageAlias::RawImageAlias(const ImageBase *image) : ImageAliasBase(AliasState::kRaw, image->GetGlobalKey()) {}
OutputImageAlias::OutputImageAlias(const ImageInput *image_input)
    : ImageAliasBase(AliasState::kOutput, image_input->GetGlobalKey()) {}

RawBufferAlias::RawBufferAlias(const BufferBase *buffer) : BufferAliasBase(AliasState::kRaw, buffer->GetGlobalKey()) {}
OutputBufferAlias::OutputBufferAlias(const BufferInput *buffer_input)
    : BufferAliasBase(AliasState::kOutput, buffer_input->GetGlobalKey()) {}

} // namespace myvk_rg::interface