#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "projected_multi_head_attention.hpp"

namespace {

constexpr float kEps = 1e-5f;

bool nearly_equal(float actual, float expected) {
    return std::fabs(actual - expected) <= kEps;
}

void test_hand_computable_end_to_end_case() {
    Tensor<float, 3> qkv({1, 2, 3}); // C = H = D = 1
    Tensor<float, 2> projection({2, 1});
    Tensor<float, 1> bias({2});
    // Per token: [Q, K, V].
    qkv(0, 0, 0) = 1.0f; qkv(0, 0, 1) = 3.0f; qkv(0, 0, 2) = 10.0f;
    qkv(0, 1, 0) = 2.0f; qkv(0, 1, 1) = 4.0f; qkv(0, 1, 2) = 20.0f;
    projection(0, 0) = 2.0f;
    projection(1, 0) = -3.0f;
    bias(0) = 1.0f;
    bias(1) = -2.0f;

    const auto output = projected_multi_head_attention(qkv, 1, projection, bias);
    const float first_key_probability = 1.0f / (1.0f + std::exp(2.0f));
    const float attended_second =
        first_key_probability * 10.0f + (1.0f - first_key_probability) * 20.0f;

    assert(output.shape() == (std::array<size_t, 3>{1, 2, 2}));
    assert(nearly_equal(output(0, 0, 0), 21.0f));
    assert(nearly_equal(output(0, 0, 1), -32.0f));
    assert(nearly_equal(output(0, 1, 0), 2.0f * attended_second + 1.0f));
    assert(nearly_equal(output(0, 1, 1), -3.0f * attended_second - 2.0f));
}

void test_multiple_batches_and_heads() {
    Tensor<float, 3> qkv({2, 2, 6}); // C = 2, H = 2, D = 1
    Tensor<float, 2> projection({2, 2});
    Tensor<float, 1> bias({2});
    projection(0, 0) = 1.0f; projection(0, 1) = 0.0f;
    projection(1, 0) = 0.0f; projection(1, 1) = 1.0f;

    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t time = 0; time < 2; ++time) {
            // Q and K are zero, so each causal row averages its available V values.
            qkv(batch, time, 0) = 0.0f;
            qkv(batch, time, 1) = 0.0f;
            qkv(batch, time, 2) = 0.0f;
            qkv(batch, time, 3) = 0.0f;
            qkv(batch, time, 4) = static_cast<float>(100 * batch + 10 * time + 1);
            qkv(batch, time, 5) = static_cast<float>(100 * batch + 10 * time + 2);
        }
    }

    const auto output = projected_multi_head_attention(qkv, 2, projection, bias);

    assert(output.shape() == (std::array<size_t, 3>{2, 2, 2}));
    for (size_t batch = 0; batch < 2; ++batch) {
        assert(nearly_equal(output(batch, 0, 0), static_cast<float>(100 * batch + 1)));
        assert(nearly_equal(output(batch, 0, 1), static_cast<float>(100 * batch + 2)));
        assert(nearly_equal(output(batch, 1, 0), static_cast<float>(100 * batch + 6)));
        assert(nearly_equal(output(batch, 1, 1), static_cast<float>(100 * batch + 7)));
    }
}

void test_head_width_to_different_output_dimension() {
    Tensor<float, 3> qkv({1, 1, 12}); // C = 4, H = 2, D = 2
    Tensor<float, 2> projection({3, 4});
    Tensor<float, 1> bias({3});
    // Q/K are zero; with T=1 the attended result is exactly V.
    for (size_t channel = 0; channel < 4; ++channel) {
        qkv(0, 0, channel) = 0.0f;
        qkv(0, 0, 4 + channel) = 0.0f;
        qkv(0, 0, 8 + channel) = static_cast<float>(channel + 1);
    }
    const float weight_values[] = {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 0.0f, -1.0f, 0.0f,
        0.0f, 2.0f, 0.0f, -2.0f,
    };
    const float bias_values[] = {1.0f, -1.0f, 2.0f};
    for (size_t i = 0; i < projection.numel(); ++i) {
        projection.data_ptr()[i] = weight_values[i];
    }
    for (size_t i = 0; i < bias.numel(); ++i) {
        bias.data_ptr()[i] = bias_values[i];
    }

    const auto output = projected_multi_head_attention(qkv, 2, projection, bias);

    assert(output.shape() == (std::array<size_t, 3>{1, 1, 3}));
    assert(nearly_equal(output(0, 0, 0), 11.0f));
    assert(nearly_equal(output(0, 0, 1), -3.0f));
    assert(nearly_equal(output(0, 0, 2), -2.0f));
}

void test_shape_validation() {
    Tensor<float, 3> invalid_packed({1, 1, 5});
    Tensor<float, 3> channels_not_divisible_by_heads({1, 1, 9}); // C = 3
    Tensor<float, 3> valid_qkv({1, 1, 6}); // C = 2
    Tensor<float, 2> valid_projection({3, 2});
    Tensor<float, 1> valid_bias({3});
    Tensor<float, 2> wrong_projection({3, 3});
    Tensor<float, 1> wrong_bias({2});

    bool packed_threw = false;
    try {
        const auto output = projected_multi_head_attention(
            invalid_packed, 1, valid_projection, valid_bias);
        (void)output;
    } catch (const std::invalid_argument&) {
        packed_threw = true;
    }
    assert(packed_threw);

    bool heads_threw = false;
    try {
        const auto output = projected_multi_head_attention(
            channels_not_divisible_by_heads, 2, valid_projection, valid_bias);
        (void)output;
    } catch (const std::invalid_argument&) {
        heads_threw = true;
    }
    assert(heads_threw);

    bool projection_threw = false;
    try {
        const auto output = projected_multi_head_attention(
            valid_qkv, 1, wrong_projection, valid_bias);
        (void)output;
    } catch (const std::invalid_argument&) {
        projection_threw = true;
    }
    assert(projection_threw);

    bool bias_threw = false;
    try {
        const auto output = projected_multi_head_attention(
            valid_qkv, 1, valid_projection, wrong_bias);
        (void)output;
    } catch (const std::invalid_argument&) {
        bias_threw = true;
    }
    assert(bias_threw);
}

void test_inputs_and_projection_parameters_remain_unchanged() {
    Tensor<float, 3> qkv({1, 2, 6});
    Tensor<float, 2> projection({3, 2});
    Tensor<float, 1> bias({3});
    for (size_t i = 0; i < qkv.numel(); ++i) {
        qkv.data_ptr()[i] = static_cast<float>(i) - 6.0f;
    }
    for (size_t i = 0; i < projection.numel(); ++i) {
        projection.data_ptr()[i] = static_cast<float>(2 * i) - 3.0f;
    }
    for (size_t i = 0; i < bias.numel(); ++i) {
        bias.data_ptr()[i] = static_cast<float>(i) + 1.0f;
    }
    const auto qkv_before = qkv.clone();
    const auto projection_before = projection.clone();
    const auto bias_before = bias.clone();

    const auto output = projected_multi_head_attention(qkv, 2, projection, bias);
    (void)output;

    for (size_t i = 0; i < qkv.numel(); ++i) {
        assert(qkv.data_ptr()[i] == qkv_before.data_ptr()[i]);
    }
    for (size_t i = 0; i < projection.numel(); ++i) {
        assert(projection.data_ptr()[i] == projection_before.data_ptr()[i]);
    }
    for (size_t i = 0; i < bias.numel(); ++i) {
        assert(bias.data_ptr()[i] == bias_before.data_ptr()[i]);
    }
}

} // namespace

int main() {
    test_hand_computable_end_to_end_case();
    test_multiple_batches_and_heads();
    test_head_width_to_different_output_dimension();
    test_shape_validation();
    test_inputs_and_projection_parameters_remain_unchanged();

    std::puts("test_projected_multi_head_attention passed");
    return 0;
}
