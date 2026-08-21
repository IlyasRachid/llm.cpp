#pragma once

#include "tensor.hpp"

template <typename T, std::size_t Rank>
Tensor<T, Rank+1> multi_head_reshape(const Tensor<T, Rank>& input, size_t H);

#include "multi_head_reshape.tpp"