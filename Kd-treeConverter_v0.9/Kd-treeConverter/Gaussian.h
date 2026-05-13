#include "Kd-treeConverter.h"

#include <cmath>


inline float calcKernelScale(float density, float kernelMinResponse = KERNEL_MIN_RESPONSE, uint32_t opts = 1, float kernelDegree = KERNEL_DEGREE) {
	const float responseModulation = (opts & 1 /* MOGRenderAdaptiveKernelClamping */) ? density : 1.0f;
	const float minResponse = std::min(kernelMinResponse / responseModulation, 0.97f);

	const float b = kernelDegree;
	const float a = -4.5f / std::pow(3.0f, b);

	// 3. e^{a * r^b} = minResponse 를 만족하는 r(반지름) 계산
	// r = (ln(minResponse) / a)^(1/b)
	return std::pow(std::log(minResponse) / a, 1.0f / b);
}