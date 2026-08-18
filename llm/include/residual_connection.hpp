#pragma once

#include "tensor.hpp"

template <typename T, std::size_t Rank>
Tensor<T, Rank> residual(const Tensor<T, Rank>& input, const Tensor<T, Rank>& sublayer);

#include "residual_connection.tpp"