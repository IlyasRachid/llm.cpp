#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "attention_probs.hpp"

namespace {

constexpr float kEps = 1e-5f;

bool nearly_equal(float actual, float expected, float tolerance = kEps) {
    return std::fabs(actual - expected) <= tolerance;
}

void assert_causal_probability_rows(const Tensor<float, 3>& probabilities) {
    const size_t batches = probabilities.shape()[0];
    const size_t time = probabilities.shape()[1];
    for (size_t batch = 0; batch < batches; ++batch) {
        for (size_t query_time = 0; query_time < time; ++query_time) {
            float sum = 0.0f;
            for (size_t key_time = 0; key_time < time; ++key_time) {
                const float probability = probabilities(batch, query_time, key_time);
                if (key_time > query_time) {
                    assert(probability == 0.0f);
                } else {
                    assert(probability >= 0.0f);
                    sum += probability;
                }
            }
            assert(nearly_equal(sum, 1.0f));
        }
    }
}

void test_hand_computable_causal_probabilities() {
    Tensor<float, 3> query({1, 2, 1});
    Tensor<float, 3> key({1, 2, 1});
    query(0, 0, 0) = 1.0f;
    query(0, 1, 0) = 2.0f;
    key(0, 0, 0) = 3.0f;
    key(0, 1, 0) = 4.0f;

    const auto probabilities = attention_probs(query, key);
    const float second_row_first = 1.0f / (1.0f + std::exp(2.0f));

    assert(probabilities.shape() == (std::array<size_t, 3>{1, 2, 2}));
    assert(nearly_equal(probabilities(0, 0, 0), 1.0f));
    assert(nearly_equal(probabilities(0, 0, 1), 0.0f));
    assert(nearly_equal(probabilities(0, 1, 0), second_row_first));
    assert(nearly_equal(probabilities(0, 1, 1), 1.0f - second_row_first));
}

void test_multiple_batches() {
    Tensor<float, 3> query({2, 2, 2});
    Tensor<float, 3> key({2, 2, 2});
    const float query_values[] = {
        1.0f, 0.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 2.0f, 0.0f,
    };
    const float key_values[] = {
        1.0f, 0.0f, 0.0f, 1.0f,
        1.0f, -1.0f, 0.0f, 2.0f,
    };
    for (size_t i = 0; i < query.numel(); ++i) {
        query.data_ptr()[i] = query_values[i];
        key.data_ptr()[i] = key_values[i];
    }

    const auto probabilities = attention_probs(query, key);

    assert_causal_probability_rows(probabilities);
    assert(probabilities(0, 1, 1) > probabilities(0, 1, 0));
    assert(probabilities(1, 1, 0) > probabilities(1, 1, 1));
}

void test_time_and_channel_dimensions_can_differ() {
    Tensor<float, 3> query({1, 3, 2});
    Tensor<float, 3> key({1, 3, 2});
    const float query_values[] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    const float key_values[] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, -1.0f};
    for (size_t i = 0; i < query.numel(); ++i) {
        query.data_ptr()[i] = query_values[i];
        key.data_ptr()[i] = key_values[i];
    }

    const auto probabilities = attention_probs(query, key);

    assert(probabilities.shape() == (std::array<size_t, 3>{1, 3, 3}));
    assert_causal_probability_rows(probabilities);
}

void test_negative_values() {
    Tensor<float, 3> query({1, 2, 2});
    Tensor<float, 3> key({1, 2, 2});
    const float query_values[] = {-1.0f, 2.0f, 3.0f, -4.0f};
    const float key_values[] = {5.0f, -6.0f, -7.0f, 8.0f};
    for (size_t i = 0; i < query.numel(); ++i) {
        query.data_ptr()[i] = query_values[i];
        key.data_ptr()[i] = key_values[i];
    }

    const auto probabilities = attention_probs(query, key);

    assert_causal_probability_rows(probabilities);
    assert(probabilities(0, 1, 0) > probabilities(0, 1, 1));
}

void test_single_channel() {
    Tensor<float, 3> query({1, 1, 1});
    Tensor<float, 3> key({1, 1, 1});
    query(0, 0, 0) = -7.0f;
    key(0, 0, 0) = 3.0f;

    const auto probabilities = attention_probs(query, key);

    assert(probabilities.shape() == (std::array<size_t, 3>{1, 1, 1}));
    assert(nearly_equal(probabilities(0, 0, 0), 1.0f));
}

void test_zero_sized_dimensions() {
    // batch dim = 0 or seq_lenght dim = 0 are fine
    Tensor<float, 3> empty_batch_query({0, 3, 2});
    Tensor<float, 3> empty_batch_key({0, 3, 2});
    bool batch_threw = true;
    try {
        const auto empty_batch = attention_probs(empty_batch_query, empty_batch_key);
        (void)empty_batch;
    } catch (const std::invalid_argument&) {
        batch_threw = false;
    }
    assert(batch_threw);

    Tensor<float, 3> empty_time_query({2, 0, 3});
    Tensor<float, 3> empty_time_key({2, 0, 3});
    bool time_threw = false;
    try {
        const auto empty_time = attention_probs(empty_time_query, empty_time_key);
        (void)empty_time;
    } catch (const std::invalid_argument&) {
        time_threw = true;
    }
    assert(time_threw);
}

void test_mismatched_batch_or_channel_dimensions_throw() {
    Tensor<float, 3> query({2, 2, 2});
    Tensor<float, 3> wrong_batch_key({3, 2, 2});
    Tensor<float, 3> wrong_channel_key({2, 2, 3});

    bool batch_threw = false;
    try {
        const auto probabilities = attention_probs(query, wrong_batch_key);
        (void)probabilities;
    } catch (const std::invalid_argument&) {
        batch_threw = true;
    }
    assert(batch_threw);

    bool channel_threw = false;
    try {
        const auto probabilities = attention_probs(query, wrong_channel_key);
        (void)probabilities;
    } catch (const std::invalid_argument&) {
        channel_threw = true;
    }
    assert(channel_threw);
}

void test_inputs_remain_unchanged() {
    Tensor<float, 3> query({1, 2, 2});
    Tensor<float, 3> key({1, 2, 2});
    const float query_values[] = {1.0f, -2.0f, 3.0f, -4.0f};
    const float key_values[] = {-5.0f, 6.0f, -7.0f, 8.0f};
    for (size_t i = 0; i < query.numel(); ++i) {
        query.data_ptr()[i] = query_values[i];
        key.data_ptr()[i] = key_values[i];
    }

    const auto probabilities = attention_probs(query, key);
    (void)probabilities;

    for (size_t i = 0; i < query.numel(); ++i) {
        assert(query.data_ptr()[i] == query_values[i]);
        assert(key.data_ptr()[i] == key_values[i]);
    }
}

} // namespace

int main() {
    test_hand_computable_causal_probabilities();
    test_multiple_batches();
    test_time_and_channel_dimensions_can_differ();
    test_negative_values();
    test_single_channel();
    test_zero_sized_dimensions();
    test_mismatched_batch_or_channel_dimensions_throw();
    test_inputs_remain_unchanged();

    std::puts("test_attention_probs passed");
    return 0;
}
