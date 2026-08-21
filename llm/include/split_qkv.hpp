#pragma once

#include <cstddef>
#include "tensor.hpp"

template <typename T>
struct QKV {
    Tensor<T, 3> Q;
    Tensor<T, 3> K;
    Tensor<T, 3> V;
};

template <typename T>
QKV<T> split_qkv(const Tensor<T, 3>& packed);

#include "split_qkv.tpp"