
template <typename T, std::size_t Rank>
Tensor<T, Rank> residual(const Tensor<T, Rank>& input, const Tensor<T, Rank>& sublayer) {
    Tensor<T, Rank> output = input + sublayer;
    return output;
}
