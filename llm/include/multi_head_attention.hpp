#pragma once

#include "tensor.hpp"

template <typename T, std::size_t Rank>
Tensor<T, Rank>  multi_head_attention(const Tensor<T, Rank>& Q, const Tensor<T, Rank>& K, const Tensor<T, Rank>& V);

#include "multi_head_attention.tpp"