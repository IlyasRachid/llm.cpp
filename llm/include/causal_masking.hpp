#pragma once

#include "tensor.hpp"

template <typename T, std::size_t Rank>
Tensor<T, Rank> causal_mask(const Tensor<T, Rank>& scores);

template <typename T, std::size_t Rank>
Tensor<T, Rank>& causal_mask_(Tensor<T, Rank>& scores);

#include "causal_masking.tpp"