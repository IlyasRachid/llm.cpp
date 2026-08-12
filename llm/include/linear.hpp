#pragma once

#include "tensor.hpp"


template <typename T, std::size_t Rank>
Tensor<T, Rank> linear(
    const Tensor<T, Rank>& input,
    const Tensor<T, 2>& weight,
    const Tensor<T, 1>& bias);

#include "linear.tpp"