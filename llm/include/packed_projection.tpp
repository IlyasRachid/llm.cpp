// normalized input: [B,T,C]
// packed weight: [3C,C]   (out_features, in_features — PyTorch nn.Linear convention)
// packed bias: [3C]
#include <stdexcept>
#include "linear.hpp"

template <typename T, std::size_t Rank>
Tensor<T, Rank> qkv_projection(const Tensor<T, Rank>& input, const Tensor<T, 2>& weight, const Tensor<T, 1>& bias) {
    return linear(input, weight, bias);
}