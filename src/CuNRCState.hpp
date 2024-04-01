//
// Created by adamyuan on 4/1/24.
//

#pragma once
#ifndef VKNRC_CUNRCSTATE_HPP
#define VKNRC_CUNRCSTATE_HPP

#include <memory>

class CuNRCState {
private:
	struct TCNNImpl;

	std::unique_ptr<TCNNImpl> m_p_tcnn_impl;

public:
	CuNRCState();

};

#endif // VKNRC_CUNRCSTATE_HPP
