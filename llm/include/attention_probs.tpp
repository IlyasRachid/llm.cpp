#include "scaled_dot_product_attention.hpp"
#include "causal_masking.hpp"
#include "softmax.hpp"

template <typename T, std::size_t Rank>
Tensor<T, Rank> attention_probs(const Tensor<T, Rank>& query, const Tensor<T, Rank>& key) {
    Tensor<T, Rank> scores = scaledDotProductScores(query, key);
    Tensor<T, Rank> masked_scores = causal_mask(scores);
    Tensor<T, Rank> attention = softmax(masked_scores);
    return attention;
}