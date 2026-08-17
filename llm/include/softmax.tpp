#include <limits>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

template <typename T, std::size_t Rank>
Tensor<T, Rank> softmax(const Tensor<T, Rank>& input) {
    static_assert(Rank >= 1, "softmax requires Rank >= 1");

    const auto& shape = input.shape();
    const auto& in_strides = input.stride();

    if (shape[Rank - 1] == 0) {
        throw std::invalid_argument("channel dimension must be bigger than 0");
    }

    Tensor<T, Rank> output(shape);
    const auto& out_strides = output.stride();

    const T* in_data = input.data_ptr();
    T* out_data = output.data_ptr();

    std::size_t block_size = shape[Rank - 1];

    std::size_t num_blocks = 1;
    for (std::size_t d = 0; d < Rank - 1; d++) num_blocks *= shape[d];

    std::array<std::size_t, Rank> idx{};

    for (std::size_t block = 0; block < num_blocks; block++) {
        std::size_t rem = block;
        for (std::size_t d = Rank - 1; d-- > 0; ) {
            idx[d] = rem % shape[d];
            rem /= shape[d];
        }

        std::size_t in_row_off = 0, out_row_off = 0;
        for (std::size_t d = 0; d < Rank - 1; d++) {
            in_row_off += idx[d] * in_strides[d];
            out_row_off += idx[d] * out_strides[d];
        }

        // pass 1: find max
        T curr_max = std::numeric_limits<T>::lowest();
        for (std::size_t k = 0; k < block_size; k++) {
            T val = in_data[in_row_off + k * in_strides[Rank - 1]];
            curr_max = std::max(curr_max, val);
        }

        // pass 2: exponentiate and accumulate
        T total = 0;
        for (std::size_t k = 0; k < block_size; k++) {
            T val = in_data[in_row_off + k * in_strides[Rank - 1]];
            T e = std::exp(val - curr_max);
            out_data[out_row_off + k * out_strides[Rank - 1]] = e;
            total += e;
        }

        // pass 3: normalize
        for (std::size_t k = 0; k < block_size; k++) {
            std::size_t off = out_row_off + k * out_strides[Rank - 1];
            out_data[off] /= total;
        }
    }

    return output;
}