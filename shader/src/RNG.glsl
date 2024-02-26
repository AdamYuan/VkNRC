#ifndef RNG_GLSL
#define RNG_GLSL

uint _RNG_state;

// Steps the RNG and returns a floating-point value between 0 and 1 inclusive.
float RNGNext() {
	// Condensed version of pcg_output_rxs_m_xs_32_32, with simple conversion to floating-point [0,1].
	_RNG_state = _RNG_state * 747796405 + 1;
	uint word = ((_RNG_state >> ((_RNG_state >> 28) + 4)) ^ _RNG_state) * 277803737;
	word = (word >> 22) ^ word;
	return float(word) / 4294967295.0f;
}

void RNGSetState(in const uint rng_state) { _RNG_state = rng_state; }

#endif
