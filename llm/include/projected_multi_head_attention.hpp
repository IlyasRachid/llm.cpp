#pragma once

#include "tensor.hpp"

template <typename T, std::size_t Rank>
Tensor<T, Rank> projected_multi_head_attention(
    const Tensor<T, Rank>& qkv,
    std::size_t num_heads,
    const Tensor<T, 2>& W_o,
    const Tensor<T, 1>& b_o
);

#include "projected_multi_head_attention.tpp"