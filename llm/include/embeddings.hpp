#pragma once

#include "tensor.hpp"

template <typename T, std::size_t Rank>
Tensor<T, Rank> positional_embedding(const Tensor<T, Rank>& token_embedding, const Tensor<T, Rank - 1>& positional_embedding);

#include "embeddings.tpp"