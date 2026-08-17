#pragma once

#include "tensor.hpp"

template <typename T, std::size_t Rank>
Tensor<T, Rank> softmax(const Tensor<T, Rank>& input);

#include "softmax.tpp"