#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "scaled_dot_product_attention.hpp"

namespace {

constexpr float kEps = 1e-5f;

bool nearly_equal(float actual, float expected) {
    return std::fabs(actual - expected) <= kEps;
}

void test_hand_computable_values_with_different_t_and_c() {
    Tensor<float, 3> query({1, 2, 3});
    Tensor<float, 3> key({1, 2, 3});
    const float query_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float key_values[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    for (size_t i = 0; i < query.numel(); ++i) {
        query.data_ptr()[i] = query_values[i];
        key.data_ptr()[i] = key_values[i];
    }

    const auto scores = scaledDotProductScores(query, key);
    const float scale = std::sqrt(3.0f);

    assert(scores.shape() == (std::array<size_t, 3>{1, 2, 2}));
    assert(nearly_equal(scores(0, 0, 0), 50.0f / scale));
    assert(nearly_equal(scores(0, 0, 1), 68.0f / scale));
    assert(nearly_equal(scores(0, 1, 0), 122.0f / scale));
    assert(nearly_equal(scores(0, 1, 1), 167.0f / scale));
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

    const auto scores = scaledDotProductScores(query, key);
    const float scale = std::sqrt(2.0f);

    assert(nearly_equal(scores(0, 0, 0), 1.0f / scale));
    assert(nearly_equal(scores(0, 0, 1), 0.0f));
    assert(nearly_equal(scores(0, 1, 0), 0.0f));
    assert(nearly_equal(scores(0, 1, 1), 1.0f / scale));
    assert(nearly_equal(scores(1, 0, 0), 0.0f));
    assert(nearly_equal(scores(1, 0, 1), 2.0f / scale));
    assert(nearly_equal(scores(1, 1, 0), 2.0f / scale));
    assert(nearly_equal(scores(1, 1, 1), 0.0f));
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

    const auto scores = scaledDotProductScores(query, key);
    const float scale = std::sqrt(2.0f);

    assert(nearly_equal(scores(0, 0, 0), -17.0f / scale));
    assert(nearly_equal(scores(0, 0, 1), 23.0f / scale));
    assert(nearly_equal(scores(0, 1, 0), 39.0f / scale));
    assert(nearly_equal(scores(0, 1, 1), -53.0f / scale));
}

void test_single_channel() {
    Tensor<float, 3> query({1, 2, 1});
    Tensor<float, 3> key({1, 2, 1});
    query(0, 0, 0) = 2.0f;
    query(0, 1, 0) = -3.0f;
    key(0, 0, 0) = 4.0f;
    key(0, 1, 0) = 5.0f;

    const auto scores = scaledDotProductScores(query, key);

    assert(scores.shape() == (std::array<size_t, 3>{1, 2, 2}));
    assert(nearly_equal(scores(0, 0, 0), 8.0f));
    assert(nearly_equal(scores(0, 0, 1), 10.0f));
    assert(nearly_equal(scores(0, 1, 0), -12.0f));
    assert(nearly_equal(scores(0, 1, 1), -15.0f));
}

void test_zero_sized_dimensions() {
    Tensor<float, 3> empty_batch_query({0, 3, 2});
    Tensor<float, 3> empty_batch_key({0, 3, 2});
    const auto empty_batch_scores =
        scaledDotProductScores(empty_batch_query, empty_batch_key);
    assert(empty_batch_scores.shape() == (std::array<size_t, 3>{0, 3, 3}));
    assert(empty_batch_scores.numel() == 0);

    Tensor<float, 3> empty_time_query({2, 0, 3});
    Tensor<float, 3> empty_time_key({2, 0, 3});
    const auto empty_time_scores =
        scaledDotProductScores(empty_time_query, empty_time_key);
    assert(empty_time_scores.shape() == (std::array<size_t, 3>{2, 0, 0}));
    assert(empty_time_scores.numel() == 0);

    Tensor<float, 3> empty_channel_query({1, 2, 0});
    Tensor<float, 3> empty_channel_key({1, 2, 0});
    bool threw = false;
    try {
        const auto scores =
            scaledDotProductScores(empty_channel_query, empty_channel_key);
        (void)scores;
    } catch (const std::domain_error&) {
        threw = true;
    }
    assert(threw);
}

void test_mismatched_batch_or_channel_dimensions_throw() {
    Tensor<float, 3> query({2, 2, 2});
    Tensor<float, 3> wrong_batch_key({3, 2, 2});
    Tensor<float, 3> wrong_channel_key({2, 2, 3});

    bool batch_threw = false;
    try {
        const auto scores = scaledDotProductScores(query, wrong_batch_key);
        (void)scores;
    } catch (const std::invalid_argument&) {
        batch_threw = true;
    }
    assert(batch_threw);

    bool channel_threw = false;
    try {
        const auto scores = scaledDotProductScores(query, wrong_channel_key);
        (void)scores;
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

    const auto scores = scaledDotProductScores(query, key);
    (void)scores;

    for (size_t i = 0; i < query.numel(); ++i) {
        assert(query.data_ptr()[i] == query_values[i]);
        assert(key.data_ptr()[i] == key_values[i]);
    }
}

} // namespace

int main() {
    test_hand_computable_values_with_different_t_and_c();
    test_multiple_batches();
    test_negative_values();
    test_single_channel();
    test_zero_sized_dimensions();
    test_mismatched_batch_or_channel_dimensions_throw();
    test_inputs_remain_unchanged();

    std::puts("test_scaled_dot_product_attention passed");
    return 0;
}
