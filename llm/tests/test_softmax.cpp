#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "softmax.hpp"

namespace {

constexpr float kEps = 1e-5f;

bool nearly_equal(float actual, float expected, float tolerance = kEps) {
    return std::fabs(actual - expected) <= tolerance;
}

template <std::size_t Rank>
void assert_rows_sum_to_one(const Tensor<float, Rank>& output) {
    const size_t channels = output.shape()[Rank - 1];
    assert(channels > 0);

    const size_t rows = output.numel() / channels;
    const float* values = output.data_ptr();
    for (size_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (size_t channel = 0; channel < channels; ++channel) {
            sum += values[row * channels + channel];
        }
        assert(nearly_equal(sum, 1.0f));
    }
}

void test_probabilities_sum_to_one() {
    Tensor<float, 3> input({2, 2, 3});
    const float values[] = {
        -2.0f, 0.0f, 1.0f,
        3.0f, 2.0f, -1.0f,
        0.5f, -0.5f, 4.0f,
        -3.0f, -2.0f, -1.0f,
    };
    for (size_t i = 0; i < input.numel(); ++i) {
        input.data_ptr()[i] = values[i];
    }

    const auto output = softmax(input);

    assert_rows_sum_to_one(output);
}

void test_uniform_inputs_produce_uniform_probabilities() {
    Tensor<float, 3> input({2, 1, 4});
    for (size_t i = 0; i < input.numel(); ++i) {
        input.data_ptr()[i] = 7.5f;
    }

    const auto output = softmax(input);

    for (size_t i = 0; i < output.numel(); ++i) {
        assert(nearly_equal(output.data_ptr()[i], 0.25f));
    }
}

void test_larger_values_receive_larger_probabilities() {
    Tensor<float, 1> input({4});
    input(0) = -1.0f;
    input(1) = 0.0f;
    input(2) = 1.0f;
    input(3) = 2.0f;

    const auto output = softmax(input);

    assert(output(0) < output(1));
    assert(output(1) < output(2));
    assert(output(2) < output(3));
    assert_rows_sum_to_one(output);
}

void test_negative_values() {
    Tensor<float, 1> input({3});
    input(0) = -3.0f;
    input(1) = -2.0f;
    input(2) = -1.0f;

    const auto output = softmax(input);

    assert(nearly_equal(output(0), 0.09003057f));
    assert(nearly_equal(output(1), 0.24472847f));
    assert(nearly_equal(output(2), 0.66524096f));
    assert_rows_sum_to_one(output);
}

void test_extreme_values_remain_finite() {
    Tensor<float, 2> input({2, 3});
    input(0, 0) = 1000.0f;
    input(0, 1) = 1001.0f;
    input(0, 2) = 999.0f;
    input(1, 0) = -1000.0f;
    input(1, 1) = -1001.0f;
    input(1, 2) = -999.0f;

    const auto output = softmax(input);

    for (size_t i = 0; i < output.numel(); ++i) {
        assert(std::isfinite(output.data_ptr()[i]));
    }
    assert_rows_sum_to_one(output);
    assert(output(0, 1) > output(0, 0));
    assert(output(1, 2) > output(1, 0));
}

void test_single_channel_is_one() {
    Tensor<float, 3> input({2, 3, 1});
    for (size_t i = 0; i < input.numel(); ++i) {
        input.data_ptr()[i] = static_cast<float>(i) - 3.0f;
    }

    const auto output = softmax(input);

    for (size_t i = 0; i < output.numel(); ++i) {
        assert(nearly_equal(output.data_ptr()[i], 1.0f));
    }
}

void test_zero_sized_dimensions() {
    Tensor<float, 3> zero_batch({0, 2, 3});
    const auto zero_batch_output = softmax(zero_batch);
    assert(zero_batch_output.shape() == (std::array<size_t, 3>{0, 2, 3}));
    assert(zero_batch_output.numel() == 0);

    Tensor<float, 3> zero_sequence({2, 0, 3});
    const auto zero_sequence_output = softmax(zero_sequence);
    assert(zero_sequence_output.shape() == (std::array<size_t, 3>{2, 0, 3}));
    assert(zero_sequence_output.numel() == 0);

    Tensor<float, 3> zero_channels({2, 3, 0});
    bool threw = false;
    try {
        const auto output = softmax(zero_channels);
        (void)output;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_shape_is_preserved() {
    Tensor<float, 4> input({2, 3, 2, 4});
    for (size_t i = 0; i < input.numel(); ++i) {
        input.data_ptr()[i] = static_cast<float>(i % 7) - 3.0f;
    }

    const auto output = softmax(input);

    assert(output.shape() == input.shape());
    assert(output.numel() == input.numel());
    assert_rows_sum_to_one(output);
}

} // namespace

int main() {
    test_probabilities_sum_to_one();
    test_uniform_inputs_produce_uniform_probabilities();
    test_larger_values_receive_larger_probabilities();
    test_negative_values();
    test_extreme_values_remain_finite();
    test_single_channel_is_one();
    test_zero_sized_dimensions();
    test_shape_is_preserved();

    std::puts("test_softmax passed");
    return 0;
}
