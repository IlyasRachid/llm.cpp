#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "tensor.hpp"
#include "layerNorm.hpp"

namespace {

constexpr float kEps = 1e-4f; // loose enough for float accumulation, tight enough to catch real bugs

bool nearly_equal(float a, float b, float tol = kEps) {
    return std::fabs(a - b) < tol;
}

Tensor<float, 1> make_vec(std::initializer_list<float> vals) {
    Tensor<float, 1> t({vals.size()});
    size_t i = 0;
    for (float v : vals) t(i++) = v;
    return t;
}

// Builds a [1, 1, C] input tensor from a single row, for convenience in the
// hand-calculated cases below.
Tensor<float, 3> make_single_row(std::initializer_list<float> vals) {
    Tensor<float, 3> t({1, 1, vals.size()});
    size_t i = 0;
    for (float v : vals) t(0, 0, i++) = v;
    return t;
}

} // namespace

// ---- constant rows ----------------------------------------------------

void test_constant_row_gamma1_beta0() {
    auto input = make_single_row({4.0f, 4.0f, 4.0f});
    auto gamma = make_vec({1.0f, 1.0f, 1.0f});
    auto beta = make_vec({0.0f, 0.0f, 0.0f});

    auto out = layerNorm(input, gamma, beta);

    for (size_t c = 0; c < 3; c++)
        assert(nearly_equal(out(0, 0, c), 0.0f));
    std::puts("test_constant_row_gamma1_beta0 passed");
}

// Constant row, nonzero gamma/beta: normalized value is 0 everywhere,
// so output must equal beta exactly (gamma has no effect).
// Reference (Python): [1.0, 1.0, 1.0]
void test_constant_row_nonzero_gamma_beta() {
    auto input = make_single_row({5.0f, 5.0f, 5.0f});
    auto gamma = make_vec({2.0f, 2.0f, 2.0f});
    auto beta = make_vec({1.0f, 1.0f, 1.0f});

    auto out = layerNorm(input, gamma, beta);

    float expected[3] = {1.0f, 1.0f, 1.0f};
    for (size_t c = 0; c < 3; c++)
        assert(nearly_equal(out(0, 0, c), expected[c]));
    std::puts("test_constant_row_nonzero_gamma_beta passed");
}

// ---- manually calculated small values ----------------------------------

// row = [2, 4, 6], gamma=1, beta=0. mean=4, variance=8/3.
// Reference (Python): [-1.2247425750014138, 0.0, 1.2247425750014138]
void test_manual_small_values() {
    auto input = make_single_row({2.0f, 4.0f, 6.0f});
    auto gamma = make_vec({1.0f, 1.0f, 1.0f});
    auto beta = make_vec({0.0f, 0.0f, 0.0f});

    auto out = layerNorm(input, gamma, beta);

    float expected[3] = {-1.2247425750014138f, 0.0f, 1.2247425750014138f};
    for (size_t c = 0; c < 3; c++)
        assert(nearly_equal(out(0, 0, c), expected[c]));
    std::puts("test_manual_small_values passed");
}

// ---- nonzero gamma and beta ---------------------------------------------

// Same row [2, 4, 6], distinct per-channel gamma/beta.
// Reference (Python): [-1.4494851500028276, -1.0, 1.7247425750014138]
void test_nonzero_gamma_and_beta() {
    auto input = make_single_row({2.0f, 4.0f, 6.0f});
    auto gamma = make_vec({2.0f, 0.5f, 1.0f});
    auto beta = make_vec({1.0f, -1.0f, 0.5f});

    auto out = layerNorm(input, gamma, beta);

    float expected[3] = {-1.4494851500028276f, -1.0f, 1.7247425750014138f};
    for (size_t c = 0; c < 3; c++)
        assert(nearly_equal(out(0, 0, c), expected[c]));
    std::puts("test_nonzero_gamma_and_beta passed");
}

// ---- negative values -----------------------------------------------------

// row = [-3, 1, 5], gamma=1, beta=0.
// Reference (Python): [-1.2247442972928344, 0.0, 1.2247442972928344]
void test_negative_values() {
    auto input = make_single_row({-3.0f, 1.0f, 5.0f});
    auto gamma = make_vec({1.0f, 1.0f, 1.0f});
    auto beta = make_vec({0.0f, 0.0f, 0.0f});

    auto out = layerNorm(input, gamma, beta);

    float expected[3] = {-1.2247442972928344f, 0.0f, 1.2247442972928344f};
    for (size_t c = 0; c < 3; c++)
        assert(nearly_equal(out(0, 0, c), expected[c]));
    std::puts("test_negative_values passed");
}

// ---- C = 1 -----------------------------------------------------------

// Single channel -> mean == the value -> variance always 0 -> normalized
// term always 0. Output must equal beta exactly regardless of gamma.
// Reference (Python): [2.0]
void test_C_equals_1() {
    Tensor<float, 3> input({1, 1, 1});
    input(0, 0, 0) = 7.0f;
    auto gamma = make_vec({3.0f});
    auto beta = make_vec({2.0f});

    auto out = layerNorm(input, gamma, beta);

    assert(nearly_equal(out(0, 0, 0), 2.0f));
    std::puts("test_C_equals_1 passed");
}

// ---- zero-sized dimensions -------------------------------------------

void test_zero_leading_dimension() {
    Tensor<float, 3> input({0, 4, 3});
    auto gamma = make_vec({1.0f, 1.0f, 1.0f});
    auto beta = make_vec({0.0f, 0.0f, 0.0f});

    auto out = layerNorm(input, gamma, beta);

    assert(out.shape()[0] == 0);
    assert(out.shape()[1] == 4);
    assert(out.shape()[2] == 3);
    assert(out.numel() == 0);
    std::puts("test_zero_leading_dimension passed");
}

void test_zero_middle_dimension() {
    Tensor<float, 3> input({2, 0, 3});
    auto gamma = make_vec({1.0f, 1.0f, 1.0f});
    auto beta = make_vec({0.0f, 0.0f, 0.0f});

    auto out = layerNorm(input, gamma, beta);

    assert(out.shape()[0] == 2);
    assert(out.shape()[1] == 0);
    assert(out.shape()[2] == 3);
    assert(out.numel() == 0);
    std::puts("test_zero_middle_dimension passed");
}

// C == 0 must throw rather than divide by zero.
void test_zero_channel_dimension_throws() {
    Tensor<float, 3> input({2, 3, 0});
    Tensor<float, 1> gamma({0});
    Tensor<float, 1> beta({0});

    bool threw = false;
    try {
        auto out = layerNorm(input, gamma, beta);
        (void)out;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::puts("test_zero_channel_dimension_throws passed");
}

// ---- shape mismatches --------------------------------------------------

void test_gamma_shape_mismatch_throws() {
    Tensor<float, 3> input({2, 3, 4});
    Tensor<float, 1> gamma({5}); // wrong: should be 4
    Tensor<float, 1> beta({4});

    bool threw = false;
    try {
        auto out = layerNorm(input, gamma, beta);
        (void)out;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::puts("test_gamma_shape_mismatch_throws passed");
}

void test_beta_shape_mismatch_throws() {
    Tensor<float, 3> input({2, 3, 4});
    Tensor<float, 1> gamma({4});
    Tensor<float, 1> beta({3}); // wrong: should be 4

    bool threw = false;
    try {
        auto out = layerNorm(input, gamma, beta);
        (void)out;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::puts("test_beta_shape_mismatch_throws passed");
}

// ---- bonus: eps precondition -----------------------------------------

void test_eps_must_be_positive() {
    Tensor<float, 3> input({1, 1, 3});
    auto gamma = make_vec({1.0f, 1.0f, 1.0f});
    auto beta = make_vec({0.0f, 0.0f, 0.0f});

    bool threw = false;
    try {
        auto out = layerNorm(input, gamma, beta, 0.0f);
        (void)out;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::puts("test_eps_must_be_positive passed");
}

// ---- numerical agreement with hand calculation across a full batch -----

void test_numerical_agreement_multi_row() {
    Tensor<float, 3> input({2, 2, 3});
    // row(0,0)=[2,4,6], row(0,1)=[-3,1,5], row(1,0)=[5,5,5] (constant), row(1,1)=[2,4,6]
    float vals[2][2][3] = {
        {{2.0f, 4.0f, 6.0f}, {-3.0f, 1.0f, 5.0f}},
        {{5.0f, 5.0f, 5.0f}, {2.0f, 4.0f, 6.0f}}
    };
    for (size_t b = 0; b < 2; b++)
        for (size_t t = 0; t < 2; t++)
            for (size_t c = 0; c < 3; c++)
                input(b, t, c) = vals[b][t][c];

    auto gamma = make_vec({1.0f, 1.0f, 1.0f});
    auto beta = make_vec({0.0f, 0.0f, 0.0f});

    auto out = layerNorm(input, gamma, beta);

    float expected_row00[3] = {-1.2247425750014138f, 0.0f, 1.2247425750014138f};
    float expected_row01[3] = {-1.2247442972928344f, 0.0f, 1.2247442972928344f};
    float expected_row10[3] = {0.0f, 0.0f, 0.0f};
    float expected_row11[3] = {-1.2247425750014138f, 0.0f, 1.2247425750014138f};

    for (size_t c = 0; c < 3; c++) {
        assert(nearly_equal(out(0, 0, c), expected_row00[c]));
        assert(nearly_equal(out(0, 1, c), expected_row01[c]));
        assert(nearly_equal(out(1, 0, c), expected_row10[c]));
        assert(nearly_equal(out(1, 1, c), expected_row11[c]));
    }
    std::puts("test_numerical_agreement_multi_row passed");
}

int main() {
    test_constant_row_gamma1_beta0();
    test_constant_row_nonzero_gamma_beta();
    test_manual_small_values();
    test_nonzero_gamma_and_beta();
    test_negative_values();
    test_C_equals_1();
    test_zero_leading_dimension();
    test_zero_middle_dimension();
    test_zero_channel_dimension_throws();
    test_gamma_shape_mismatch_throws();
    test_beta_shape_mismatch_throws();
    test_eps_must_be_positive();
    test_numerical_agreement_multi_row();

    std::puts("All layerNorm tests passed.");
    return 0;
}