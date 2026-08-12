#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "linear.hpp"

namespace {

constexpr float kEps = 1e-5f;

bool nearly_equal(float actual, float expected) {
    return std::fabs(actual - expected) <= kEps;
}

std::vector<float> values(const Tensor<float, 3>& tensor) {
    return std::vector<float>(tensor.data_ptr(),
                              tensor.data_ptr() + tensor.numel());
}

std::vector<float> values(const Tensor<float, 2>& tensor) {
    return std::vector<float>(tensor.data_ptr(),
                              tensor.data_ptr() + tensor.numel());
}

std::vector<float> values(const Tensor<float, 1>& tensor) {
    return std::vector<float>(tensor.data_ptr(),
                              tensor.data_ptr() + tensor.numel());
}

// Deliberately use tensor coordinates and the mathematical definition rather
// than the implementation's flattened-row addressing.
Tensor<float, 3> reference_linear(const Tensor<float, 3>& input,
                                  const Tensor<float, 2>& weight,
                                  const Tensor<float, 1>& bias) {
    const size_t batches = input.shape()[0];
    const size_t tokens = input.shape()[1];
    const size_t cin = input.shape()[2];
    const size_t cout = weight.shape()[0];
    Tensor<float, 3> expected({batches, tokens, cout});

    for (size_t batch = 0; batch < batches; ++batch) {
        for (size_t token = 0; token < tokens; ++token) {
            for (size_t out_channel = 0; out_channel < cout; ++out_channel) {
                float total = bias(out_channel);
                for (size_t in_channel = 0; in_channel < cin; ++in_channel) {
                    total += input(batch, token, in_channel) *
                             weight(out_channel, in_channel);
                }
                expected(batch, token, out_channel) = total;
            }
        }
    }
    return expected;
}

void assert_matches_reference(const Tensor<float, 3>& input,
                              const Tensor<float, 2>& weight,
                              const Tensor<float, 1>& bias) {
    const auto expected = reference_linear(input, weight, bias);
    const auto actual = linear(input, weight, bias);
    assert(actual.shape() == expected.shape());
    for (size_t i = 0; i < actual.numel(); ++i) {
        assert(nearly_equal(actual.data_ptr()[i], expected.data_ptr()[i]));
    }
}

void test_hand_computable_case() {
    Tensor<float, 3> input({1, 1, 2});
    input(0, 0, 0) = 1.0f;
    input(0, 0, 1) = 2.0f;
    Tensor<float, 2> weight({2, 2});
    weight(0, 0) = 3.0f; weight(0, 1) = 4.0f;
    weight(1, 0) = 5.0f; weight(1, 1) = 6.0f;
    Tensor<float, 1> bias({2});
    bias(0) = 10.0f; bias(1) = 20.0f;

    const auto output = linear(input, weight, bias);
    assert(output.shape() == (std::array<size_t, 3>{1, 1, 2}));
    assert(output(0, 0, 0) == 21.0f);
    assert(output(0, 0, 1) == 37.0f);
}

void test_batches_positions_and_non_square_dimensions() {
    Tensor<float, 3> input({2, 3, 2});
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t token = 0; token < 3; ++token) {
            input(batch, token, 0) = static_cast<float>(batch * 4 + token - 2);
            input(batch, token, 1) = static_cast<float>(batch - token + 3);
        }
    }
    Tensor<float, 2> weight({4, 2});
    const float weight_values[4][2] = {{1, 2}, {-3, 1}, {0, -2}, {4, -1}};
    for (size_t out = 0; out < 4; ++out)
        for (size_t in = 0; in < 2; ++in)
            weight(out, in) = weight_values[out][in];
    Tensor<float, 1> bias({4});
    bias(0) = 1.0f; bias(1) = -2.0f; bias(2) = 3.0f; bias(3) = 0.5f;

    assert_matches_reference(input, weight, bias);
}

void test_zero_and_negative_values_with_zero_bias() {
    Tensor<float, 3> input({1, 2, 3});
    input(0, 0, 0) = 0.0f;  input(0, 0, 1) = -2.0f; input(0, 0, 2) = 4.0f;
    input(0, 1, 0) = -1.0f; input(0, 1, 1) = 0.0f;  input(0, 1, 2) = -3.0f;
    Tensor<float, 2> weight({2, 3});
    weight(0, 0) = 0.0f; weight(0, 1) = -1.0f; weight(0, 2) = 2.0f;
    weight(1, 0) = -2.0f; weight(1, 1) = 0.0f; weight(1, 2) = -1.0f;
    Tensor<float, 1> bias({2});
    bias(0) = 0.0f; bias(1) = 0.0f;

    assert_matches_reference(input, weight, bias);
}

void test_shape_mismatches_throw() {
    Tensor<float, 3> input({1, 1, 2});
    Tensor<float, 2> wrong_weight({3, 3});
    Tensor<float, 2> valid_weight({3, 2});
    Tensor<float, 1> valid_bias({3});
    Tensor<float, 1> wrong_bias({2});

    bool threw = false;
    try { static_cast<void>(linear(input, wrong_weight, valid_bias)); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);

    threw = false;
    try { static_cast<void>(linear(input, valid_weight, wrong_bias)); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
}

void test_zero_sized_dimensions() {
    // Empty leading dimensions are valid: no rows means no reductions.
    Tensor<float, 3> empty_input({0, 3, 2});
    Tensor<float, 2> weight({4, 2});
    Tensor<float, 1> bias({4});
    const auto output = linear(empty_input, weight, bias);
    assert(output.shape() == (std::array<size_t, 3>{0, 3, 4}));
    assert(output.numel() == 0);

    // A zero final dimension has no linear reduction and is rejected.
    Tensor<float, 3> zero_channels({2, 3, 0});
    Tensor<float, 2> zero_width_weight({4, 0});
    bool threw = false;
    try { static_cast<void>(linear(zero_channels, zero_width_weight, bias)); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
}

void test_inputs_are_not_modified() {
    Tensor<float, 3> input({1, 2, 2});
    input(0, 0, 0) = 1.0f; input(0, 0, 1) = -2.0f;
    input(0, 1, 0) = 3.0f; input(0, 1, 1) = 4.0f;
    Tensor<float, 2> weight({2, 2});
    weight(0, 0) = 5.0f; weight(0, 1) = 6.0f;
    weight(1, 0) = -1.0f; weight(1, 1) = 2.0f;
    Tensor<float, 1> bias({2});
    bias(0) = 7.0f; bias(1) = 8.0f;
    const auto input_before = values(input);
    const auto weight_before = values(weight);
    const auto bias_before = values(bias);

    static_cast<void>(linear(input, weight, bias));

    assert(values(input) == input_before);
    assert(values(weight) == weight_before);
    assert(values(bias) == bias_before);
}

} // namespace

int main() {
    test_hand_computable_case();
    test_batches_positions_and_non_square_dimensions();
    test_zero_and_negative_values_with_zero_bias();
    test_shape_mismatches_throw();
    test_zero_sized_dimensions();
    test_inputs_are_not_modified();
    std::puts("test_linear passed");
    return 0;
}
