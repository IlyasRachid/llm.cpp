#pragma once

#include "tensor.hpp"

template <typename T, std::size_t Rank>
Tensor<T, Rank> aggregate(const Tensor<T, Rank>& attention, const Tensor<T, Rank>& values);

#include "attention_aggregation.tpp"

