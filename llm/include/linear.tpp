#include <stdexcept>
#include <array>

// By convention the shapes are:
// input: [..., C_in]   (any rank >= 2, leading dims treated as batch/rows)
// weight: [C_out, C_in]
// bias: [C_out]
// output: [..., C_out]

template <typename T, std::size_t Rank>
Tensor<T, Rank> linear(const Tensor<T, Rank>& input, const Tensor<T, 2>& weight, const Tensor<T, 1>& bias) {
    static_assert(Rank >= 2, "linear requires input rank >= 2");

    const auto& input_shape = input.shape();
    const auto& weight_shape = weight.shape();
    const auto& bias_shape = bias.shape();

    if (input_shape[Rank-1] != weight_shape[1] || bias_shape[0] != weight_shape[0]) {
        throw std::invalid_argument("Tensor shape mismatch");
    }

    const std::size_t cin = input_shape[Rank-1];
    const std::size_t cout = weight_shape[0];
    if (cin == 0) {
        throw std::invalid_argument("Input channel dimension must be non-zero");
    }

    std::array<std::size_t, Rank> out_shape(input_shape);
    out_shape[Rank-1] = cout;
    Tensor<T, Rank> output(out_shape);

    const auto& in_strides = input.stride();
    const auto& out_strides = output.stride();
    const auto& w_strides = weight.stride();
    const auto& b_strides = bias.stride();

    const T* in_data = input.data_ptr();
    T* out_data = output.data_ptr();
    const T* w_data = weight.data_ptr();
    const T* b_data = bias.data_ptr();

    std::size_t rows = 1;
    for (std::size_t d = 0; d < Rank - 1; d++) rows *= input_shape[d];

    std::array<std::size_t, Rank> idx{};

    for (std::size_t row = 0; row < rows; ++row) {
        // decompose flat row index into leading-dim indices
        std::size_t rem = row;
        for (std::size_t d = Rank - 1; d-- > 0; ) {
            idx[d] = rem % input_shape[d];
            rem /= input_shape[d];
        }

        std::size_t in_row_off = 0, out_row_off = 0;
        for (std::size_t d = 0; d < Rank - 1; d++) {
            in_row_off += idx[d] * in_strides[d];
            out_row_off += idx[d] * out_strides[d];
        }

        for (std::size_t oc = 0; oc < cout; ++oc) {
            T total = b_data[oc * b_strides[0]];

            std::size_t w_row_off = oc * w_strides[0];
            for (std::size_t ic = 0; ic < cin; ++ic) {
                std::size_t in_off = in_row_off + ic * in_strides[Rank - 1];
                std::size_t w_off = w_row_off + ic * w_strides[1];
                total += in_data[in_off] * w_data[w_off];
            }

            std::size_t out_off = out_row_off + oc * out_strides[Rank - 1];
            out_data[out_off] = total;
        }
    }

    return output;
}