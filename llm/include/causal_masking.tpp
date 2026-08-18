#include <array>
#include <limits>
#include <stdexcept>
#include <type_traits>

template <typename T, std::size_t Rank>
Tensor<T, Rank> causal_mask(const Tensor<T, Rank>& scores) {
    static_assert(Rank >= 2, "Rank must be at least 2");
    static_assert(std::is_floating_point_v<T>, "causal_mask requires a floating-point tensor type");

    const auto& shape = scores.shape();
    size_t N = shape[Rank-1];

    if (N != shape[Rank-2]) {
        throw std::invalid_argument("the last two dimensions must be equal");
    }
    const auto& c_strides = scores.stride();

    Tensor<T, Rank> masked(shape);

    const T* c_data = scores.data_ptr();
    T* m_data = masked.data_ptr();

    size_t batch_size = 1;
    for (size_t b = 0; b < Rank-2; b++) { batch_size *= shape[b]; }

    std::array<size_t, Rank> batch_idx{};

    for (size_t b = 0; b < batch_size; b++) {
        size_t c_batch_off = 0;
        size_t rem = b;
        for (size_t d = Rank-2; d-- > 0; ) {
            batch_idx[d] = rem % shape[d];
            rem /= shape[d];
        }

        for (size_t d = 0; d < Rank - 2; d++) {
            c_batch_off += c_strides[d] * batch_idx[d];
        }

        for (size_t q = 0; q < N; q++) {
            for (size_t k = 0; k < N; k++) {
                size_t c_off = c_batch_off + q * c_strides[Rank-2] + k * c_strides[Rank-1];
                if (k <= q) {
                    m_data[c_off] = c_data[c_off];
                } else {
                    m_data[c_off] = -std::numeric_limits<T>::infinity();
                }
            }
        }
    }

    return masked;
}

template <typename T, std::size_t Rank>
Tensor<T, Rank>& causal_mask_(Tensor<T, Rank>& scores) {
    static_assert(Rank >= 2, "Rank must be at least 2");
    static_assert(std::is_floating_point_v<T>, "causal_mask requires a floating-point tensor type");

    const auto& shape = scores.shape();
    size_t N = shape[Rank-1];
    if (N != shape[Rank-2]) {
        throw std::invalid_argument("the last two dimensions must be equal");
    }

    const auto& strides = scores.stride();
    T* data = scores.data_ptr();

    size_t batch_size = 1;
    for (size_t b = 0; b < Rank-2; b++) batch_size *= shape[b];

    std::array<size_t, Rank> batch_idx{};

    for (size_t b = 0; b < batch_size; b++) {
        size_t batch_off = 0;
        size_t rem = b;
        for (size_t d = Rank-2; d-- > 0; ) {
            batch_idx[d] = rem % shape[d];
            rem /= shape[d];
        }
        for (size_t d = 0; d < Rank - 2; d++) batch_off += strides[d] * batch_idx[d];

        for (size_t q = 0; q < N; q++) {
            for (size_t k = q + 1; k < N; k++) {   // only touch what needs masking
                size_t off = batch_off + q * strides[Rank-2] + k * strides[Rank-1];
                data[off] = -std::numeric_limits<T>::infinity();
            }
        }
    }
    return scores;
}