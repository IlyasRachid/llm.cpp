#include "scaled_dot_product_attention.hpp"
#include "causal_masking.hpp"
#include "softmax.hpp"
#include "attention_aggregation.hpp"

// here we assume that Q, K, V: [B,H,T,D]
// query, ket and values tensors are already 4-Rank, so no need for reshaping

template <typename T, std::size_t Rank>
Tensor<T, Rank>  multi_head_attention(const Tensor<T, Rank>& Q, const Tensor<T, Rank>& K, const Tensor<T, Rank>& V) {
    Tensor<T, Rank> scores = scaledDotProductScores(Q, K);
    Tensor<T, Rank> masked = causal_mask(scores);
    Tensor<T, Rank> attention = softmax(masked);
    Tensor<T, Rank> attention_scores = aggregate(attention, V);

    return attention_scores;
}