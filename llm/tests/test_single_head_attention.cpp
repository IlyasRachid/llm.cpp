#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>

#include "single_head_attention.hpp"

namespace {

constexpr float kEps = 1e-5f;

bool nearly_equal(float actual, float expected) {
    return std::fabs(actual - expected) <= kEps;
}

void test_numerical_agreement_with_tiny_manual_case() {
    Tensor<float, 3> query({1, 2, 1});
    Tensor<float, 3> key({1, 2, 1});
    Tensor<float, 3> values({1, 2, 1});
    query(0, 0, 0) = 1.0f;
    query(0, 1, 0) = 2.0f;
    key(0, 0, 0) = 3.0f;
    key(0, 1, 0) = 4.0f;
    values(0, 0, 0) = 10.0f;
    values(0, 1, 0) = 20.0f;

    const auto output = single_head_attention(query, key, values);
    const float first_key_probability = 1.0f / (1.0f + std::exp(2.0f));
    const float expected_second =
        first_key_probability * 10.0f + (1.0f - first_key_probability) * 20.0f;

    assert(nearly_equal(output(0, 0, 0), 10.0f));
    assert(nearly_equal(output(0, 1, 0), expected_second));
}

void test_causal_behavior() {
    Tensor<float, 3> query({1, 3, 1});
    Tensor<float, 3> key({1, 3, 1});
    Tensor<float, 3> values({1, 3, 1});
    for (size_t time = 0; time < 3; ++time) {
        query(0, time, 0) = static_cast<float>(time + 1);
        key(0, time, 0) = static_cast<float>(time + 2);
        values(0, time, 0) = static_cast<float>((time + 1) * 10);
    }

    const auto baseline = single_head_attention(query, key, values);

    key(0, 2, 0) = -100.0f;
    values(0, 2, 0) = 999.0f;
    const auto changed_future = single_head_attention(query, key, values);

    assert(nearly_equal(baseline(0, 0, 0), changed_future(0, 0, 0)));
    assert(nearly_equal(baseline(0, 1, 0), changed_future(0, 1, 0)));
    assert(nearly_equal(baseline(0, 0, 0), 10.0f));
}

void test_output_shape() {
    Tensor<float, 3> query({2, 3, 2});
    Tensor<float, 3> key({2, 3, 2});
    Tensor<float, 3> values({2, 3, 4});

    const auto output = single_head_attention(query, key, values);

    assert(output.shape() == (std::array<size_t, 3>{2, 3, 4}));
    assert(output.numel() == 24);
}

void test_multiple_batches() {
    Tensor<float, 3> query({2, 2, 1});
    Tensor<float, 3> key({2, 2, 1});
    Tensor<float, 3> values({2, 2, 1});
    const float query_values[] = {1.0f, 2.0f, -1.0f, 3.0f};
    const float key_values[] = {1.0f, 2.0f, 4.0f, -2.0f};
    const float value_values[] = {10.0f, 20.0f, -5.0f, 15.0f};
    for (size_t i = 0; i < query.numel(); ++i) {
        query.data_ptr()[i] = query_values[i];
        key.data_ptr()[i] = key_values[i];
        values.data_ptr()[i] = value_values[i];
    }

    const auto output = single_head_attention(query, key, values);

    assert(nearly_equal(output(0, 0, 0), 10.0f));
    assert(nearly_equal(output(1, 0, 0), -5.0f));
    assert(output(0, 1, 0) > 10.0f);
    assert(output(1, 1, 0) < 0.0f);
}

void test_multiple_sequence_lengths_and_channel_sizes() {
    Tensor<float, 3> query_one({1, 1, 1});
    Tensor<float, 3> key_one({1, 1, 1});
    Tensor<float, 3> values_one({1, 1, 1});
    values_one(0, 0, 0) = 7.0f;
    const auto output_one = single_head_attention(query_one, key_one, values_one);
    assert(output_one.shape() == (std::array<size_t, 3>{1, 1, 1}));
    assert(nearly_equal(output_one(0, 0, 0), 7.0f));

    Tensor<float, 3> query_three({1, 3, 2});
    Tensor<float, 3> key_three({1, 3, 2});
    Tensor<float, 3> values_three({1, 3, 4});
    for (size_t i = 0; i < values_three.numel(); ++i) {
        values_three.data_ptr()[i] = static_cast<float>(i) - 4.0f;
    }
    const auto output_three =
        single_head_attention(query_three, key_three, values_three);
    assert(output_three.shape() == (std::array<size_t, 3>{1, 3, 4}));
    assert(output_three.numel() == 12);
}

void test_inputs_remain_unchanged() {
    Tensor<float, 3> query({1, 2, 2});
    Tensor<float, 3> key({1, 2, 2});
    Tensor<float, 3> values({1, 2, 3});
    const float query_values[] = {1.0f, -2.0f, 3.0f, -4.0f};
    const float key_values[] = {-5.0f, 6.0f, -7.0f, 8.0f};
    const float value_values[] = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
    for (size_t i = 0; i < query.numel(); ++i) {
        query.data_ptr()[i] = query_values[i];
        key.data_ptr()[i] = key_values[i];
    }
    for (size_t i = 0; i < values.numel(); ++i) {
        values.data_ptr()[i] = value_values[i];
    }

    const auto output = single_head_attention(query, key, values);
    (void)output;

    for (size_t i = 0; i < query.numel(); ++i) {
        assert(query.data_ptr()[i] == query_values[i]);
        assert(key.data_ptr()[i] == key_values[i]);
    }
    for (size_t i = 0; i < values.numel(); ++i) {
        assert(values.data_ptr()[i] == value_values[i]);
    }
}

} // namespace

int main() {
    test_numerical_agreement_with_tiny_manual_case();
    test_causal_behavior();
    test_output_shape();
    test_multiple_batches();
    test_multiple_sequence_lengths_and_channel_sizes();
    test_inputs_remain_unchanged();

    std::puts("test_single_head_attention passed");
    return 0;
}
