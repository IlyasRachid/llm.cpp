#include "split_qkv.hpp"
#include "multi_head_attention.hpp"
#include "multi_head_reshape.hpp"
#include "merge_multi_head.hpp"
#include "attention_projection.hpp"

template <typename T, std::size_t Rank>
Tensor<T, Rank> projected_multi_head_attention(const Tensor<T, Rank>& qkv, std::size_t num_heads, const Tensor<T, 2>& W_o, const Tensor<T, 1>& b_o) {

    QKV<T> parts = split_qkv(qkv);

    Tensor<T, Rank+1> multi_head_q = multi_head_reshape(parts.Q, num_heads);
    Tensor<T, Rank+1> multi_head_k = multi_head_reshape(parts.K, num_heads);
    Tensor<T, Rank+1> multi_head_v = multi_head_reshape(parts.V, num_heads);

    Tensor<T, Rank+1> attentions = multi_head_attention(multi_head_q, multi_head_k, multi_head_v);

    Tensor<T, Rank> merged = merge_multi_head(attentions);

    Tensor<T, Rank> projection = attention_projection(merged, W_o, b_o);

    return projection;
}