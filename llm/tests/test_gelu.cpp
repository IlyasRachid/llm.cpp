#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>

#include "gelu.hpp"

namespace {

constexpr float kEps = 1e-5f;

bool nearly_equal(float actual, float expected, float tolerance = kEps) {
    return std::fabs(actual - expected) <= tolerance;
}

void test_zero_positive_and_negative_values() {
    Tensor<float, 1> input({3});
    input(0) = 0.0f;
    input(1) = 1.0f;
    input(2) = -1.0f;

    const auto output = gelu(input);

    assert(output(0) == 0.0f);
    assert(output(1) > 0.0f);
    assert(output(1) < input(1));
    assert(output(2) < 0.0f);
    assert(output(2) > input(2));
}

void test_approximate_reference_values() {
    Tensor<float, 1> input({4});
    input(0) = 1.0f;
    input(1) = -1.0f;
    input(2) = 2.0f;
    input(3) = -2.0f;

    const auto output = gelu(input);

    // Reference values for GELU(x) = x * Phi(x).
    assert(nearly_equal(output(0), 0.84134475f));
    assert(nearly_equal(output(1), -0.15865525f));
    assert(nearly_equal(output(2), 1.95449974f));
    assert(nearly_equal(output(3), -0.04550026f));
}

void test_symmetry_relation() {
    Tensor<float, 1> input({3});
    input(0) = 0.25f;
    input(1) = 1.5f;
    input(2) = 3.0f;

    for (size_t i = 0; i < input.numel(); ++i) {
        Tensor<float, 1> negated({1});
        negated(0) = -input(i);
        // Evaluate both signs from explicit one-element tensors so this checks
        // GELU(x) - GELU(-x) = x, not a property of shared storage.
        Tensor<float, 1> positive_input({1});
        positive_input(0) = input(i);
        const auto positive_output = gelu(positive_input);
        const auto negative_output = gelu(negated);
        assert(nearly_equal(positive_output(0) - negative_output(0), input(i)));
    }
}

void test_zero_sized_tensors() {
    Tensor<float, 3> input({2, 0, 4});
    const auto output = gelu(input);

    assert(output.shape() == (std::array<size_t, 3>{2, 0, 4}));
    assert(output.numel() == 0);
}

} // namespace

int main() {
    test_zero_positive_and_negative_values();
    test_approximate_reference_values();
    test_symmetry_relation();
    test_zero_sized_tensors();
    std::puts("test_gelu passed");
    return 0;
}
