#include <stdexcept>
#include <array>

// By convention the shapes are:
// input: [B, T, C_in]
// weight: [C_out, C_in]
// bias: [C_out]
// output: [B, T, C_out]

template <typename T, std::size_t Rank>
Tensor<T, Rank> linear(const Tensor<T, Rank>& input, const Tensor<T, 2>& weight, const Tensor<T, 1>& bias) {
    if (Rank <= 2) {
        throw std::invalid_argument("Rank must be at least 3");
    }

    auto& input_shape = input.shape();
    auto& weight_shape = weight.shape();
    auto& bias_shape = bias.shape();

    if (input_shape[Rank-1] != weight_shape[1] || bias_shape[0] != weight_shape[0]) {
        throw std::invalid_argument("Tensor shape mismatch");
    }

    std::array<std::size_t, Rank> out_shape(input_shape);
    out_shape[Rank-1] = weight_shape[0];

    Tensor<T, Rank> output(out_shape);

    const size_t cin = input_shape[Rank-1];
    const size_t cout = weight_shape[0];
    // A zero-width feature vector has no defined linear reduction here.
    // It must be rejected explicitly instead of dividing by zero when computing rows.
    if (cin == 0) {
        throw std::invalid_argument("Input channel dimension must be non-zero");
    }
    const size_t rows = input.numel() / cin;

    for (size_t row = 0; row < rows; ++row) {
        const T* input_row = input.data_ptr() + row * cin;
        T* output_row = output.data_ptr() + row * cout;

        for (size_t out_channel = 0; out_channel < cout; ++out_channel) {
            const T* weight_row = weight.data_ptr() + out_channel * cin;
            T total = bias.data_ptr()[out_channel];

            for (size_t in_channel = 0; in_channel < cin; ++in_channel) {
                total += input_row[in_channel] * weight_row[in_channel];
            }

            output_row[out_channel] = total;
        }
    }

    return output;
}
