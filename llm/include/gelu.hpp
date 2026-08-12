#pragma once

#include "tensor.hpp"

template <typename T, std::size_t Rank>
Tensor<T, Rank> gelu(const Tensor<T, Rank>& input);

#include "gelu.tpp"