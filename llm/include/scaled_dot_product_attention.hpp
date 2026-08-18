#pragma once

#include "tensor.hpp"

template <typename T, std::size_t Rank>
Tensor<T, Rank> scaledDotProductScores(const Tensor<T, Rank>& A, const Tensor<T, Rank>& B);

#include "scaled_dot_product_attention.tpp"