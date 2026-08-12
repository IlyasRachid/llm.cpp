#include <stdexcept>
#include <cmath>
#include <array>

template <typename T, std::size_t Rank>
Tensor<T, Rank> layerNorm(const Tensor<T, Rank>& input, const Tensor<T, 1>& gamma, const Tensor<T, 1>& beta, float eps) {
    if (Rank <= 2) {
        throw std::invalid_argument("Rank must be at least equal to 3");
    }
    if (eps <= 0) {
        throw std::invalid_argument("epsilon must be bigger than 0");
    }
    size_t C = input.shape()[Rank - 1];

    if (C == 0) {
        throw std::invalid_argument("last dimension cannot be null");
    }
    if (gamma.shape()[0] != C || beta.shape()[0] != C) {
        throw std::invalid_argument("gamma/beta must have shape [C] matching input's last dimension");
    }

    std::array<size_t, Rank> arr = input.shape();
    Tensor<T, Rank> output(arr);

    const size_t total = input.numel() / C;
    const T* data_ptr = input.data_ptr();
    T* out_ptr = output.data_ptr();

    for (size_t stride = 0; stride < total; stride += 1) {
        const T* curr = data_ptr + stride * C;

        T mean = 0;
        T variance = 0;

        for (size_t i = 0; i < C; i++) {
            mean += curr[i];
        }
        mean /= C;

        for (size_t i = 0; i < C; i++) {
            T diff = curr[i] - mean;
            variance += diff * diff;
        }
        variance /= C;

        T* out_row = out_ptr + stride * C;
        for (size_t i = 0; i < C; i++) {
            T result = gamma(i) * ((curr[i] - mean) / std::sqrt(variance + eps)) + beta(i);
            out_row[i] = result;
        }
    }

    return output;
}