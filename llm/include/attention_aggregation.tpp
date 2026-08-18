
template <typename T, std::size_t Rank>
Tensor<T, Rank> aggregate(const Tensor<T, Rank>& attention, const Tensor<T, Rank>& values) {
    Tensor<T, Rank> aggregate = attention.matmul(values, false);
    return aggregate;
}
