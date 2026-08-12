#pragma once

#include "tensor.hpp"

template <typename T, std::size_t Rank>
Tensor<T, Rank> layerNorm(const Tensor<T, Rank>& input, const Tensor<T, 1>& gamma, const Tensor<T, 1>& beta, float eps=1e-5f);

#include "layerNorm.tpp"