#pragma once

#include "tensor.hpp"

template <typename T, std::size_t Rank>
Tensor<T, Rank> qkv_projection(const Tensor<T, Rank>& input, const Tensor<T, 2>& weight, const Tensor<T, 1>& bias);

#include "packed_projection.tpp"