#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "causal_masking.hpp"

namespace {

void test_t_equals_one() {
    Tensor<float, 3> scores({1, 1, 1});
    scores(0, 0, 0) = 2.0f;

    const auto masked = causal_mask(scores);

    assert(masked.shape() == (std::array<size_t, 3>{1, 1, 1}));
    assert(masked(0, 0, 0) == 2.0f);
}

void test_future_positions_are_negative_infinity() {
    Tensor<float, 3> scores({1, 3, 3});
    for (size_t i = 0; i < scores.numel(); ++i) {
        scores.data_ptr()[i] = static_cast<float>(i) - 4.0f;
    }

    const auto masked = causal_mask(scores);

    for (size_t query_time = 0; query_time < 3; ++query_time) {
        for (size_t key_time = query_time + 1; key_time < 3; ++key_time) {
            assert(std::isinf(masked(0, query_time, key_time)));
            assert(masked(0, query_time, key_time) < 0.0f);
        }
    }
}

void test_current_and_previous_positions_remain_unchanged() {
    Tensor<float, 3> scores({1, 3, 3});
    const float values[] = {
        1.0f, -2.0f, 3.0f,
        4.0f, 5.0f, -6.0f,
        7.0f, -8.0f, 9.0f,
    };
    for (size_t i = 0; i < scores.numel(); ++i) {
        scores.data_ptr()[i] = values[i];
    }

    const auto masked = causal_mask(scores);

    for (size_t query_time = 0; query_time < 3; ++query_time) {
        for (size_t key_time = 0; key_time <= query_time; ++key_time) {
            assert(masked(0, query_time, key_time) ==
                   scores(0, query_time, key_time));
        }
    }
}

void test_multiple_batches() {
    Tensor<float, 3> scores({2, 2, 2});
    const float values[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, -2.0f, -3.0f, -4.0f,
    };
    for (size_t i = 0; i < scores.numel(); ++i) {
        scores.data_ptr()[i] = values[i];
    }

    const auto masked = causal_mask(scores);

    for (size_t batch = 0; batch < 2; ++batch) {
        assert(std::isinf(masked(batch, 0, 1)));
        assert(masked(batch, 0, 1) < 0.0f);
        assert(masked(batch, 0, 0) == scores(batch, 0, 0));
        assert(masked(batch, 1, 0) == scores(batch, 1, 0));
        assert(masked(batch, 1, 1) == scores(batch, 1, 1));
    }
}

void test_zero_sized_dimensions() {
    Tensor<float, 3> empty_batch({0, 3, 3});
    const auto empty_batch_output = causal_mask(empty_batch);
    assert(empty_batch_output.shape() == (std::array<size_t, 3>{0, 3, 3}));
    assert(empty_batch_output.numel() == 0);

    Tensor<float, 3> empty_time({2, 0, 0});
    const auto empty_time_output = causal_mask(empty_time);
    assert(empty_time_output.shape() == (std::array<size_t, 3>{2, 0, 0}));
    assert(empty_time_output.numel() == 0);
}

void test_input_remains_unchanged() {
    Tensor<float, 3> scores({1, 2, 2});
    const float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (size_t i = 0; i < scores.numel(); ++i) {
        scores.data_ptr()[i] = values[i];
    }

    const auto masked = causal_mask(scores);
    (void)masked;

    for (size_t i = 0; i < scores.numel(); ++i) {
        assert(scores.data_ptr()[i] == values[i]);
    }
}

void test_non_square_scores_throw() {
    Tensor<float, 3> scores({1, 2, 3});

    bool threw = false;
    try {
        const auto masked = causal_mask(scores);
        (void)masked;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

} // namespace

int main() {
    test_t_equals_one();
    test_future_positions_are_negative_infinity();
    test_current_and_previous_positions_remain_unchanged();
    test_multiple_batches();
    test_zero_sized_dimensions();
    test_input_remains_unchanged();
    test_non_square_scores_throw();

    std::puts("test_causal_masking passed");
    return 0;
}
