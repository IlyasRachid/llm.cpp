#include <array>
#include <stdexcept>

template <typename T, std::size_t Rank>
Tensor<T, Rank> positional_embedding(const Tensor<T, Rank>& token_embedding, const Tensor<T, Rank - 1>& positional_embedding) {
    if (Rank < 2) {
        throw std::invalid_argument("positional_embedding needs a batch dim plus at least one inner dim");
    }

    const auto& tok_shape = token_embedding.shape();
    const auto& pos_shape = positional_embedding.shape();

    // token_embedding.shape()[1:] must equal positional_embedding.shape().
    for (std::size_t i = 0; i < Rank - 1; ++i) {
        if (tok_shape[i + 1] != pos_shape[i]) {
            throw std::invalid_argument("positional_embedding: token_embedding.shape()[1:] must match positional_embedding.shape()");
        }
    }

    Tensor<T, Rank> out(tok_shape);

    const std::size_t batch = tok_shape[0];
    const std::size_t inner = positional_embedding.numel(); // product of all dims but the batch dim

    const T* tok_ptr = token_embedding.data_ptr();
    const T* pos_ptr = positional_embedding.data_ptr();
    T* out_ptr = out.data_ptr();

    for (std::size_t b = 0; b < batch; ++b) {
        const T* tok_row = tok_ptr + b * inner;
        T* out_row = out_ptr + b * inner;
        for (std::size_t i = 0; i < inner; ++i) {
            out_row[i] = tok_row[i] + pos_ptr[i];
        }
    }

    return out;
}