#pragma once

#include "tensor.hpp"

template <typename T, std::size_t Rank>
Tensor<T, Rank> scaledDotProductScores(const Tensor<T, Rank>& query, const Tensor<T, Rank>& key);

#include "scaled_dot_product_attention.tpp"