#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "attention_projection.hpp"

namespace {

constexpr float kEps = 1e-5f;

bool nearly_equal(float actual, float expected) {
    return std::fabs(actual - expected) <= kEps;
}

Tensor<float, 3> reference_projection(const Tensor<float, 3>& attention,
                                      const Tensor<float, 2>& projection,
                                      const Tensor<float, 1>& bias) {
    const auto& shape = attention.shape();
    Tensor<float, 3> expected({shape[0], shape[1], projection.shape()[0]});
    for (size_t batch = 0; batch < shape[0]; ++batch) {
        for (size_t time = 0; time < shape[1]; ++time) {
            for (size_t out_channel = 0; out_channel < projection.shape()[0]; ++out_channel) {
                float total = bias(out_channel);
                for (size_t channel = 0; channel < shape[2]; ++channel) {
                    total += attention(batch, time, channel) *
                             projection(out_channel, channel);
                }
                expected(batch, time, out_channel) = total;
            }
        }
    }
    return expected;
}

void assert_matches_reference(const Tensor<float, 3>& attention,
                              const Tensor<float, 2>& projection,
                              const Tensor<float, 1>& bias) {
    const auto expected = reference_projection(attention, projection, bias);
    const auto actual = attention_projection(attention, projection, bias);
    assert(actual.shape() == expected.shape());
    for (size_t i = 0; i < actual.numel(); ++i) {
        assert(nearly_equal(actual.data_ptr()[i], expected.data_ptr()[i]));
    }
}

void test_hand_computable_projection() {
    Tensor<float, 3> attention({1, 1, 2});
    Tensor<float, 2> projection({3, 2});
    Tensor<float, 1> bias({3});
    attention(0, 0, 0) = 1.0f;
    attention(0, 0, 1) = 2.0f;
    projection(0, 0) = 3.0f; projection(0, 1) = 4.0f;
    projection(1, 0) = 5.0f; projection(1, 1) = 6.0f;
    projection(2, 0) = -1.0f; projection(2, 1) = 2.0f;
    bias(0) = 10.0f; bias(1) = 20.0f; bias(2) = -3.0f;

    const auto output = attention_projection(attention, projection, bias);

    assert(output.shape() == (std::array<size_t, 3>{1, 1, 3}));
    assert(nearly_equal(output(0, 0, 0), 21.0f));
    assert(nearly_equal(output(0, 0, 1), 37.0f));
    assert(nearly_equal(output(0, 0, 2), 0.0f));
}

void test_multiple_batches() {
    Tensor<float, 3> attention({2, 2, 2});
    Tensor<float, 2> projection({2, 2});
    Tensor<float, 1> bias({2});
    const float attention_values[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 5.0f, 2.0f, -3.0f,
    };
    const float projection_values[] = {1.0f, -2.0f, 3.0f, 4.0f};
    const float bias_values[] = {0.5f, -1.0f};
    for (size_t i = 0; i < attention.numel(); ++i) {
        attention.data_ptr()[i] = attention_values[i];
    }
    for (size_t i = 0; i < projection.numel(); ++i) {
        projection.data_ptr()[i] = projection_values[i];
    }
    for (size_t i = 0; i < bias.numel(); ++i) {
        bias.data_ptr()[i] = bias_values[i];
    }

    assert_matches_reference(attention, projection, bias);
}

void test_non_square_input_and_output_channels() {
    Tensor<float, 3> attention({1, 3, 2});
    Tensor<float, 2> projection({4, 2});
    Tensor<float, 1> bias({4});
    for (size_t i = 0; i < attention.numel(); ++i) {
        attention.data_ptr()[i] = static_cast<float>(i) - 2.0f;
    }
    const float projection_values[] = {1.0f, 0.0f, 0.0f, 1.0f,
                                       1.0f, 1.0f, -1.0f, 2.0f};
    for (size_t i = 0; i < projection.numel(); ++i) {
        projection.data_ptr()[i] = projection_values[i];
    }

    const auto output = attention_projection(attention, projection, bias);

    assert(output.shape() == (std::array<size_t, 3>{1, 3, 4}));
    assert_matches_reference(attention, projection, bias);
}

void test_zero_and_negative_values() {
    Tensor<float, 3> attention({1, 2, 3});
    Tensor<float, 2> projection({2, 3});
    Tensor<float, 1> bias({2});
    const float attention_values[] = {0.0f, -2.0f, 4.0f, -1.0f, 0.0f, -3.0f};
    const float projection_values[] = {0.0f, -1.0f, 2.0f, -2.0f, 0.0f, -1.0f};
    for (size_t i = 0; i < attention.numel(); ++i) {
        attention.data_ptr()[i] = attention_values[i];
    }
    for (size_t i = 0; i < projection.numel(); ++i) {
        projection.data_ptr()[i] = projection_values[i];
    }

    assert_matches_reference(attention, projection, bias);
    const auto output = attention_projection(attention, projection, bias);
    assert(nearly_equal(output(0, 0, 0), 10.0f));
    assert(nearly_equal(output(0, 0, 1), -4.0f));
    assert(nearly_equal(output(0, 1, 0), -6.0f));
    assert(nearly_equal(output(0, 1, 1), 5.0f));
}

void test_shape_mismatches_throw() {
    Tensor<float, 3> attention({1, 2, 2});
    Tensor<float, 2> wrong_projection({3, 3});
    Tensor<float, 2> valid_projection({3, 2});
    Tensor<float, 1> valid_bias({3});
    Tensor<float, 1> wrong_bias({2});

    bool projection_threw = false;
    try {
        const auto output = attention_projection(attention, wrong_projection, valid_bias);
        (void)output;
    } catch (const std::invalid_argument&) {
        projection_threw = true;
    }
    assert(projection_threw);

    bool bias_threw = false;
    try {
        const auto output = attention_projection(attention, valid_projection, wrong_bias);
        (void)output;
    } catch (const std::invalid_argument&) {
        bias_threw = true;
    }
    assert(bias_threw);
}

void test_inputs_remain_unchanged() {
    Tensor<float, 3> attention({1, 2, 2});
    Tensor<float, 2> projection({3, 2});
    Tensor<float, 1> bias({3});
    const float attention_values[] = {1.0f, -2.0f, 3.0f, -4.0f};
    const float projection_values[] = {5.0f, 6.0f, -1.0f, 2.0f, 3.0f, -5.0f};
    const float bias_values[] = {7.0f, 8.0f, -9.0f};
    for (size_t i = 0; i < attention.numel(); ++i) {
        attention.data_ptr()[i] = attention_values[i];
    }
    for (size_t i = 0; i < projection.numel(); ++i) {
        projection.data_ptr()[i] = projection_values[i];
    }
    for (size_t i = 0; i < bias.numel(); ++i) {
        bias.data_ptr()[i] = bias_values[i];
    }

    const auto output = attention_projection(attention, projection, bias);
    (void)output;

    for (size_t i = 0; i < attention.numel(); ++i) {
        assert(attention.data_ptr()[i] == attention_values[i]);
    }
    for (size_t i = 0; i < projection.numel(); ++i) {
        assert(projection.data_ptr()[i] == projection_values[i]);
    }
    for (size_t i = 0; i < bias.numel(); ++i) {
        assert(bias.data_ptr()[i] == bias_values[i]);
    }
}

} // namespace

int main() {
    test_hand_computable_projection();
    test_multiple_batches();
    test_non_square_input_and_output_channels();
    test_zero_and_negative_values();
    test_shape_mismatches_throw();
    test_inputs_remain_unchanged();

    std::puts("test_attention_projection passed");
    return 0;
}
