#pragma once

#include "tensor.hpp"

template <typename T, std::size_t Rank>
Tensor<T, Rank> attention_probs(const Tensor<T, Rank>& query, const Tensor<T, Rank>& key);

#include "attention_probs.tpp"