#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>

#include "multi_head_attention.hpp"
#include "single_head_attention.hpp"

namespace {

constexpr float kEps = 1e-5f;

bool nearly_equal(float actual, float expected) {
    return std::fabs(actual - expected) <= kEps;
}

void test_output_shape() {
    Tensor<float, 4> query({2, 3, 4, 2});
    Tensor<float, 4> key({2, 3, 4, 2});
    Tensor<float, 4> values({2, 3, 4, 2});

    const auto output = multi_head_attention(query, key, values);

    assert(output.shape() == (std::array<size_t, 4>{2, 3, 4, 2}));
    assert(output.numel() == query.numel());
}

void test_causal_behavior_is_independent_per_head() {
    Tensor<float, 4> query({1, 2, 3, 1});
    Tensor<float, 4> key({1, 2, 3, 1});
    Tensor<float, 4> values({1, 2, 3, 1});
    const float head_zero_values[] = {10.0f, 20.0f, 30.0f};
    const float head_one_values[] = {-1.0f, -2.0f, -3.0f};
    for (size_t time = 0; time < 3; ++time) {
        values(0, 0, time, 0) = head_zero_values[time];
        values(0, 1, time, 0) = head_one_values[time];
    }

    const auto baseline = multi_head_attention(query, key, values);

    values(0, 0, 2, 0) = 999.0f;
    const auto changed_future = multi_head_attention(query, key, values);

    assert(nearly_equal(baseline(0, 0, 0, 0), changed_future(0, 0, 0, 0)));
    assert(nearly_equal(baseline(0, 0, 1, 0), changed_future(0, 0, 1, 0)));
    assert(nearly_equal(baseline(0, 1, 0, 0), changed_future(0, 1, 0, 0)));
    assert(nearly_equal(baseline(0, 1, 1, 0), changed_future(0, 1, 1, 0)));
    assert(nearly_equal(baseline(0, 0, 0, 0), 10.0f));
    assert(nearly_equal(baseline(0, 1, 0, 0), -1.0f));
}

void test_multiple_batches() {
    Tensor<float, 4> query({2, 2, 2, 1});
    Tensor<float, 4> key({2, 2, 2, 1});
    Tensor<float, 4> values({2, 2, 2, 1});
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t head = 0; head < 2; ++head) {
            for (size_t time = 0; time < 2; ++time) {
                values(batch, head, time, 0) =
                    static_cast<float>(100 * batch + 10 * head + time);
            }
        }
    }

    const auto output = multi_head_attention(query, key, values);

    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t head = 0; head < 2; ++head) {
            assert(nearly_equal(output(batch, head, 0, 0),
                                values(batch, head, 0, 0)));
        }
    }
}

void test_multiple_heads() {
    Tensor<float, 4> query({1, 3, 2, 2});
    Tensor<float, 4> key({1, 3, 2, 2});
    Tensor<float, 4> values({1, 3, 2, 2});
    for (size_t head = 0; head < 3; ++head) {
        for (size_t time = 0; time < 2; ++time) {
            for (size_t channel = 0; channel < 2; ++channel) {
                values(0, head, time, channel) =
                    static_cast<float>(100 * head + 10 * time + channel);
            }
        }
    }

    const auto output = multi_head_attention(query, key, values);

    for (size_t head = 0; head < 3; ++head) {
        for (size_t channel = 0; channel < 2; ++channel) {
            assert(nearly_equal(output(0, head, 0, channel),
                                values(0, head, 0, channel)));
        }
    }
}

void test_single_head_agrees_with_single_head_attention() {
    Tensor<float, 4> multi_query({1, 1, 2, 2});
    Tensor<float, 4> multi_key({1, 1, 2, 2});
    Tensor<float, 4> multi_values({1, 1, 2, 2});
    Tensor<float, 3> single_query({1, 2, 2});
    Tensor<float, 3> single_key({1, 2, 2});
    Tensor<float, 3> single_values({1, 2, 2});
    const float query_values[] = {1.0f, -2.0f, 3.0f, 4.0f};
    const float key_values[] = {-1.0f, 5.0f, 2.0f, -3.0f};
    const float value_values[] = {10.0f, 20.0f, 30.0f, 40.0f};
    for (size_t time = 0; time < 2; ++time) {
        for (size_t channel = 0; channel < 2; ++channel) {
            const size_t index = time * 2 + channel;
            multi_query(0, 0, time, channel) = query_values[index];
            multi_key(0, 0, time, channel) = key_values[index];
            multi_values(0, 0, time, channel) = value_values[index];
            single_query(0, time, channel) = query_values[index];
            single_key(0, time, channel) = key_values[index];
            single_values(0, time, channel) = value_values[index];
        }
    }

    const auto multi_output =
        multi_head_attention(multi_query, multi_key, multi_values);
    const auto single_output =
        single_head_attention(single_query, single_key, single_values);

    for (size_t time = 0; time < 2; ++time) {
        for (size_t channel = 0; channel < 2; ++channel) {
            assert(nearly_equal(multi_output(0, 0, time, channel),
                                single_output(0, time, channel)));
        }
    }
}

void test_different_sequence_lengths_and_head_dimensions() {
    Tensor<float, 4> query_one({1, 2, 1, 1});
    Tensor<float, 4> key_one({1, 2, 1, 1});
    Tensor<float, 4> values_one({1, 2, 1, 1});
    values_one(0, 0, 0, 0) = 3.0f;
    values_one(0, 1, 0, 0) = -4.0f;
    const auto output_one = multi_head_attention(query_one, key_one, values_one);
    assert(output_one.shape() == (std::array<size_t, 4>{1, 2, 1, 1}));
    assert(nearly_equal(output_one(0, 0, 0, 0), 3.0f));
    assert(nearly_equal(output_one(0, 1, 0, 0), -4.0f));

    Tensor<float, 4> query_three({1, 2, 3, 3});
    Tensor<float, 4> key_three({1, 2, 3, 3});
    Tensor<float, 4> values_three({1, 2, 3, 3});
    for (size_t i = 0; i < values_three.numel(); ++i) {
        values_three.data_ptr()[i] = static_cast<float>(i) - 9.0f;
    }
    const auto output_three =
        multi_head_attention(query_three, key_three, values_three);
    assert(output_three.shape() == (std::array<size_t, 4>{1, 2, 3, 3}));
    assert(output_three.numel() == values_three.numel());
}

void test_inputs_remain_unchanged() {
    Tensor<float, 4> query({1, 2, 2, 2});
    Tensor<float, 4> key({1, 2, 2, 2});
    Tensor<float, 4> values({1, 2, 2, 2});
    for (size_t i = 0; i < query.numel(); ++i) {
        query.data_ptr()[i] = static_cast<float>(i) - 4.0f;
        key.data_ptr()[i] = static_cast<float>(2 * i) - 7.0f;
        values.data_ptr()[i] = static_cast<float>(3 * i) - 5.0f;
    }
    const auto query_before = query.clone();
    const auto key_before = key.clone();
    const auto values_before = values.clone();

    const auto output = multi_head_attention(query, key, values);
    (void)output;

    for (size_t i = 0; i < query.numel(); ++i) {
        assert(query.data_ptr()[i] == query_before.data_ptr()[i]);
        assert(key.data_ptr()[i] == key_before.data_ptr()[i]);
        assert(values.data_ptr()[i] == values_before.data_ptr()[i]);
    }
}

} // namespace

int main() {
    test_output_shape();
    test_causal_behavior_is_independent_per_head();
    test_multiple_batches();
    test_multiple_heads();
    test_single_head_agrees_with_single_head_attention();
    test_different_sequence_lengths_and_head_dimensions();
    test_inputs_remain_unchanged();

    std::puts("test_multi_head_attention passed");
    return 0;
}
