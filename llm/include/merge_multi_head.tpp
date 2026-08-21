#include <cassert>

template <typename T, std::size_t Rank>
Tensor<T, Rank-1> merge_multi_head(const Tensor<T, Rank>& attention_heads) {
    static_assert(Rank >= 3, "merge_heads requires at least [..., H, T, D]");

    Tensor<T, Rank> transposed = attention_heads.transpose(Rank-3, Rank-2); // [...,H,T,D] -> [...,T,H,D]
    Tensor<T, Rank-1> merged = transposed.template merge<2>(Rank-2); // [...,T,H,D] -> [...,T,H*D]

    return merged;
}