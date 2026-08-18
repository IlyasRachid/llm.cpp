#include "attention_probs.hpp"
#include "attention_aggregation.hpp"


template <typename T, std::size_t Rank>
Tensor<T, Rank> single_head_attention(const Tensor<T, Rank>& Q,
    const Tensor<T, Rank>& K,
    const Tensor<T, Rank>& V) {

        Tensor<T, Rank> attention = attention_probs(Q, K);
        Tensor<T, Rank> output = aggregate(attention, V);

        return output;

}