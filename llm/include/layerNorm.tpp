#include <stdexcept>
#include <cmath>
#include <array>

template <typename T, std::size_t Rank>
Tensor<T, Rank> layerNorm(const Tensor<T, Rank>& input, const Tensor<T, 1>& gamma, const Tensor<T, 1>& beta, float eps) {
    static_assert(Rank >= 2, "layerNorm requires input rank >= 2");

    if (eps <= 0) {
        throw std::invalid_argument("epsilon must be bigger than 0");
    }

    const auto& shape = input.shape();
    std::size_t C = shape[Rank - 1];

    if (C == 0) {
        throw std::invalid_argument("last dimension cannot be null");
    }
    if (gamma.shape()[0] != C || beta.shape()[0] != C) {
        throw std::invalid_argument("gamma/beta must have shape [C] matching input's last dimension");
    }

    Tensor<T, Rank> output(shape);

    const auto& in_strides = input.stride();
    const auto& out_strides = output.stride();
    const auto& g_strides = gamma.stride();
    const auto& b_strides = beta.stride();

    const T* in_data = input.data_ptr();
    T* out_data = output.data_ptr();
    const T* g_data = gamma.data_ptr();
    const T* b_data = beta.data_ptr();

    std::size_t rows = 1;
    for (std::size_t d = 0; d < Rank - 1; d++) rows *= shape[d];

    std::array<std::size_t, Rank> idx{};

    for (std::size_t row = 0; row < rows; row++) {
        std::size_t rem = row;
        for (std::size_t d = Rank - 1; d-- > 0; ) {
            idx[d] = rem % shape[d];
            rem /= shape[d];
        }

        std::size_t in_row_off = 0, out_row_off = 0;
        for (std::size_t d = 0; d < Rank - 1; d++) {
            in_row_off += idx[d] * in_strides[d];
            out_row_off += idx[d] * out_strides[d];
        }

        T mean = 0;
        for (std::size_t i = 0; i < C; i++) {
            mean += in_data[in_row_off + i * in_strides[Rank-1]];
        }
        mean /= static_cast<T>(C);

        T variance = 0;
        for (std::size_t i = 0; i < C; i++) {
            T diff = in_data[in_row_off + i * in_strides[Rank-1]] - mean;
            variance += diff * diff;
        }
        variance /= static_cast<T>(C);

        T inv_std = T(1) / std::sqrt(variance + static_cast<T>(eps));
        for (std::size_t i = 0; i < C; i++) {
            T x = in_data[in_row_off + i * in_strides[Rank-1]];
            T g = g_data[i * g_strides[0]];
            T b = b_data[i * b_strides[0]];
            out_data[out_row_off + i * out_strides[Rank-1]] = g * (x - mean) * inv_std + b;
        }
    }

    return output;
}