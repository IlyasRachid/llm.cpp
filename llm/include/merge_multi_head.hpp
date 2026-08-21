#pragma once 

#include "tensor.hpp"

template <typename T, std::size_t Rank>
Tensor<T, Rank-1> merge_multi_head(const Tensor<T, Rank>& attention_heads);

#include "merge_multi_head.tpp"