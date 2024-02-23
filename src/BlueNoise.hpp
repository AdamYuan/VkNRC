//
// Created by adamyuan on 2/24/24.
//

#pragma once
#ifndef VKNRC_BLUENOISE_HPP
#define VKNRC_BLUENOISE_HPP

#include <cinttypes>
#include <span>

struct BlueNoise {
	static std::span<const uint8_t, 256 * 256 * 2> Get256RG();
};

#endif // VKNRC_BLUENOISE_HPP
