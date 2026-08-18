#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "residual_connection.hpp"

namespace {

constexpr float kEps = 1e-5f;

bool nearly_equal(float actual, float expected) {
    return std::fabs(actual - expected) <= kEps;
}

void test_shape_is_preserved() {
    Tensor<float, 4> input({2, 3, 4, 5});
    Tensor<float, 4> sublayer({2, 3, 4, 5});

    const auto output = residual(input, sublayer);

    assert(output.shape() == input.shape());
    assert(output.numel() == input.numel());
}

void test_hand_computable_values() {
    Tensor<float, 3> input({1, 2, 2});
    Tensor<float, 3> sublayer({1, 2, 2});
    const float input_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float sublayer_values[] = {10.0f, 20.0f, 30.0f, 40.0f};
    for (size_t i = 0; i < input.numel(); ++i) {
        input.data_ptr()[i] = input_values[i];
        sublayer.data_ptr()[i] = sublayer_values[i];
    }

    const auto output = residual(input, sublayer);

    assert(nearly_equal(output(0, 0, 0), 11.0f));
    assert(nearly_equal(output(0, 0, 1), 22.0f));
    assert(nearly_equal(output(0, 1, 0), 33.0f));
    assert(nearly_equal(output(0, 1, 1), 44.0f));
}

void test_negative_values() {
    Tensor<float, 2> input({2, 2});
    Tensor<float, 2> sublayer({2, 2});
    input(0, 0) = -1.0f; input(0, 1) = 2.0f;
    input(1, 0) = -3.0f; input(1, 1) = 4.0f;
    sublayer(0, 0) = 5.0f; sublayer(0, 1) = -6.0f;
    sublayer(1, 0) = -7.0f; sublayer(1, 1) = 8.0f;

    const auto output = residual(input, sublayer);

    assert(nearly_equal(output(0, 0), 4.0f));
    assert(nearly_equal(output(0, 1), -4.0f));
    assert(nearly_equal(output(1, 0), -10.0f));
    assert(nearly_equal(output(1, 1), 12.0f));
}

void test_zero_tensors_produce_zero_output() {
    Tensor<float, 3> input({2, 2, 3});
    Tensor<float, 3> sublayer({2, 2, 3});

    const auto output = residual(input, sublayer);

    for (size_t i = 0; i < output.numel(); ++i) {
        assert(nearly_equal(output.data_ptr()[i], 0.0f));
    }
}

void test_shape_mismatch_throws() {
    Tensor<float, 3> input({1, 2, 3});
    Tensor<float, 3> sublayer({1, 3, 2});

    bool threw = false;
    try {
        const auto output = residual(input, sublayer);
        (void)output;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_inputs_remain_unchanged() {
    Tensor<float, 3> input({1, 2, 2});
    Tensor<float, 3> sublayer({1, 2, 2});
    const float input_values[] = {1.0f, -2.0f, 3.0f, -4.0f};
    const float sublayer_values[] = {-5.0f, 6.0f, -7.0f, 8.0f};
    for (size_t i = 0; i < input.numel(); ++i) {
        input.data_ptr()[i] = input_values[i];
        sublayer.data_ptr()[i] = sublayer_values[i];
    }

    const auto output = residual(input, sublayer);
    (void)output;

    for (size_t i = 0; i < input.numel(); ++i) {
        assert(input.data_ptr()[i] == input_values[i]);
        assert(sublayer.data_ptr()[i] == sublayer_values[i]);
    }
}

} // namespace

int main() {
    test_shape_is_preserved();
    test_hand_computable_values();
    test_negative_values();
    test_zero_tensors_produce_zero_output();
    test_shape_mismatch_throws();
    test_inputs_remain_unchanged();

    std::puts("test_residual_connection passed");
    return 0;
}
