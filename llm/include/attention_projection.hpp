#pragma once

#include "tensor.hpp"

template <typename T, std::size_t Rank>
Tensor<T, Rank> attention_projection(const Tensor<T, Rank>& attention, const Tensor<T, 2>& projection, const Tensor<T, 1>& bias);

#include "attention_projection.tpp"