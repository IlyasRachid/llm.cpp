#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "packed_projection.hpp"

namespace {

constexpr float kEps = 1e-5f;

bool nearly_equal(float actual, float expected) {
    return std::fabs(actual - expected) <= kEps;
}

Tensor<float, 3> reference_projection(const Tensor<float, 3>& input,
                                      const Tensor<float, 2>& weight,
                                      const Tensor<float, 1>& bias) {
    Tensor<float, 3> expected(
        {input.shape()[0], input.shape()[1], weight.shape()[0]});
    for (size_t batch = 0; batch < input.shape()[0]; ++batch) {
        for (size_t time = 0; time < input.shape()[1]; ++time) {
            for (size_t output_channel = 0; output_channel < weight.shape()[0];
                 ++output_channel) {
                float total = bias(output_channel);
                for (size_t channel = 0; channel < input.shape()[2]; ++channel) {
                    total += input(batch, time, channel) *
                             weight(output_channel, channel);
                }
                expected(batch, time, output_channel) = total;
            }
        }
    }
    return expected;
}

void test_hand_computable_values_and_qkv_sections() {
    Tensor<float, 3> input({1, 1, 2});
    Tensor<float, 2> weight({6, 2});
    Tensor<float, 1> bias({6});
    input(0, 0, 0) = 1.0f;
    input(0, 0, 1) = 2.0f;

    const float weight_values[] = {
        1.0f, 0.0f,  // Q[0]
        0.0f, 1.0f,  // Q[1]
        1.0f, 1.0f,  // K[0]
        -1.0f, 1.0f, // K[1]
        2.0f, 0.0f,  // V[0]
        0.0f, 2.0f,  // V[1]
    };
    const float bias_values[] = {0.0f, 0.0f, 1.0f, 0.0f, -2.0f, 3.0f};
    for (size_t i = 0; i < weight.numel(); ++i) {
        weight.data_ptr()[i] = weight_values[i];
    }
    for (size_t i = 0; i < bias.numel(); ++i) {
        bias.data_ptr()[i] = bias_values[i];
    }

    const auto output = qkv_projection(input, weight, bias);
    const auto query = output.slice({{0, 1}, {0, 1}, {0, 2}});
    const auto key = output.slice({{0, 1}, {0, 1}, {2, 4}});
    const auto value = output.slice({{0, 1}, {0, 1}, {4, 6}});

    assert(output.shape() == (std::array<size_t, 3>{1, 1, 6}));
    assert(nearly_equal(output(0, 0, 0), 1.0f));
    assert(nearly_equal(output(0, 0, 1), 2.0f));
    assert(nearly_equal(output(0, 0, 2), 4.0f));
    assert(nearly_equal(output(0, 0, 3), 1.0f));
    assert(nearly_equal(output(0, 0, 4), 0.0f));
    assert(nearly_equal(output(0, 0, 5), 7.0f));
    assert(query.shape() == (std::array<size_t, 3>{1, 1, 2}));
    assert(key.shape() == (std::array<size_t, 3>{1, 1, 2}));
    assert(value.shape() == (std::array<size_t, 3>{1, 1, 2}));
    assert(nearly_equal(query(0, 0, 0), 1.0f));
    assert(nearly_equal(query(0, 0, 1), 2.0f));
    assert(nearly_equal(key(0, 0, 0), 4.0f));
    assert(nearly_equal(key(0, 0, 1), 1.0f));
    assert(nearly_equal(value(0, 0, 0), 0.0f));
    assert(nearly_equal(value(0, 0, 1), 7.0f));
}

void test_output_shape_and_multiple_batches() {
    Tensor<float, 3> input({2, 2, 2});
    Tensor<float, 2> weight({6, 2});
    Tensor<float, 1> bias({6});
    const float input_values[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 5.0f, 2.0f, -3.0f,
    };
    const float weight_values[] = {
        1.0f, 2.0f, -1.0f, 1.0f, 0.0f, -2.0f,
        3.0f, -1.0f, 2.0f, 2.0f, -3.0f, 1.0f,
    };
    const float bias_values[] = {0.5f, -1.0f, 2.0f, 0.0f, 1.0f, -2.0f};
    for (size_t i = 0; i < input.numel(); ++i) {
        input.data_ptr()[i] = input_values[i];
    }
    for (size_t i = 0; i < weight.numel(); ++i) {
        weight.data_ptr()[i] = weight_values[i];
    }
    for (size_t i = 0; i < bias.numel(); ++i) {
        bias.data_ptr()[i] = bias_values[i];
    }

    const auto output = qkv_projection(input, weight, bias);
    const auto expected = reference_projection(input, weight, bias);

    assert(output.shape() == (std::array<size_t, 3>{2, 2, 6}));
    for (size_t i = 0; i < output.numel(); ++i) {
        assert(nearly_equal(output.data_ptr()[i], expected.data_ptr()[i]));
    }
}

void test_negative_values() {
    Tensor<float, 3> input({1, 1, 1});
    Tensor<float, 2> weight({3, 1});
    Tensor<float, 1> bias({3});
    input(0, 0, 0) = -2.0f;
    weight(0, 0) = 3.0f;
    weight(1, 0) = -4.0f;
    weight(2, 0) = 0.0f;
    bias(0) = 1.0f;
    bias(1) = -1.0f;
    bias(2) = -5.0f;

    const auto output = qkv_projection(input, weight, bias);

    assert(nearly_equal(output(0, 0, 0), -5.0f));
    assert(nearly_equal(output(0, 0, 1), 7.0f));
    assert(nearly_equal(output(0, 0, 2), -5.0f));
}

void test_shape_mismatches_throw() {
    Tensor<float, 3> input({1, 2, 2});
    Tensor<float, 2> wrong_weight({6, 3});
    Tensor<float, 2> valid_weight({6, 2});
    Tensor<float, 1> valid_bias({6});
    Tensor<float, 1> wrong_bias({5});

    bool channel_threw = false;
    try {
        const auto output = qkv_projection(input, wrong_weight, valid_bias);
        (void)output;
    } catch (const std::invalid_argument&) {
        channel_threw = true;
    }
    assert(channel_threw);

    bool bias_threw = false;
    try {
        const auto output = qkv_projection(input, valid_weight, wrong_bias);
        (void)output;
    } catch (const std::invalid_argument&) {
        bias_threw = true;
    }
    assert(bias_threw);
}

void test_inputs_remain_unchanged() {
    Tensor<float, 3> input({1, 2, 2});
    Tensor<float, 2> weight({6, 2});
    Tensor<float, 1> bias({6});
    const float input_values[] = {1.0f, -2.0f, 3.0f, -4.0f};
    const float weight_values[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f,
    };
    const float bias_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    for (size_t i = 0; i < input.numel(); ++i) {
        input.data_ptr()[i] = input_values[i];
    }
    for (size_t i = 0; i < weight.numel(); ++i) {
        weight.data_ptr()[i] = weight_values[i];
    }
    for (size_t i = 0; i < bias.numel(); ++i) {
        bias.data_ptr()[i] = bias_values[i];
    }

    const auto output = qkv_projection(input, weight, bias);
    (void)output;

    for (size_t i = 0; i < input.numel(); ++i) {
        assert(input.data_ptr()[i] == input_values[i]);
    }
    for (size_t i = 0; i < weight.numel(); ++i) {
        assert(weight.data_ptr()[i] == weight_values[i]);
    }
    for (size_t i = 0; i < bias.numel(); ++i) {
        assert(bias.data_ptr()[i] == bias_values[i]);
    }
}

} // namespace

int main() {
    test_hand_computable_values_and_qkv_sections();
    test_output_shape_and_multiple_batches();
    test_negative_values();
    test_shape_mismatches_throw();
    test_inputs_remain_unchanged();

    std::puts("test_packed_projection passed");
    return 0;
}
