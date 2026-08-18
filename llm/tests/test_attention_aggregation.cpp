#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "attention_aggregation.hpp"

namespace {

constexpr float kEps = 1e-5f;

bool nearly_equal(float actual, float expected) {
    return std::fabs(actual - expected) <= kEps;
}

void test_hand_computable_values() {
    Tensor<float, 3> attention({1, 2, 2});
    Tensor<float, 3> values({1, 2, 2});
    attention(0, 0, 0) = 0.2f;
    attention(0, 0, 1) = 0.8f;
    attention(0, 1, 0) = 0.5f;
    attention(0, 1, 1) = 0.5f;
    values(0, 0, 0) = 1.0f;
    values(0, 0, 1) = 2.0f;
    values(0, 1, 0) = 3.0f;
    values(0, 1, 1) = 4.0f;

    const auto output = aggregate(attention, values);

    assert(output.shape() == (std::array<size_t, 3>{1, 2, 2}));
    assert(nearly_equal(output(0, 0, 0), 2.6f));
    assert(nearly_equal(output(0, 0, 1), 3.6f));
    assert(nearly_equal(output(0, 1, 0), 2.0f));
    assert(nearly_equal(output(0, 1, 1), 3.0f));
}

void test_multiple_batches() {
    Tensor<float, 3> attention({2, 2, 2});
    Tensor<float, 3> values({2, 2, 1});
    const float attention_values[] = {
        1.0f, 0.0f, 0.0f, 1.0f,
        0.25f, 0.75f, 0.5f, 0.5f,
    };
    const float value_values[] = {10.0f, 20.0f, -4.0f, 8.0f};
    for (size_t i = 0; i < attention.numel(); ++i) {
        attention.data_ptr()[i] = attention_values[i];
    }
    for (size_t i = 0; i < values.numel(); ++i) {
        values.data_ptr()[i] = value_values[i];
    }

    const auto output = aggregate(attention, values);

    assert(nearly_equal(output(0, 0, 0), 10.0f));
    assert(nearly_equal(output(0, 1, 0), 20.0f));
    assert(nearly_equal(output(1, 0, 0), 5.0f));
    assert(nearly_equal(output(1, 1, 0), 2.0f));
}

void test_time_and_channel_dimensions_can_differ() {
    Tensor<float, 3> attention({1, 3, 3});
    Tensor<float, 3> values({1, 3, 2});
    const float attention_values[] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };
    const float value_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    for (size_t i = 0; i < attention.numel(); ++i) {
        attention.data_ptr()[i] = attention_values[i];
    }
    for (size_t i = 0; i < values.numel(); ++i) {
        values.data_ptr()[i] = value_values[i];
    }

    const auto output = aggregate(attention, values);

    assert(output.shape() == (std::array<size_t, 3>{1, 3, 2}));
    for (size_t i = 0; i < output.numel(); ++i) {
        assert(nearly_equal(output.data_ptr()[i], value_values[i]));
    }
}

void test_one_hot_attention_rows_select_values() {
    Tensor<float, 3> attention({1, 3, 3});
    Tensor<float, 3> values({1, 3, 2});
    const float attention_values[] = {
        0.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
    };
    const float value_values[] = {10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f};
    for (size_t i = 0; i < attention.numel(); ++i) {
        attention.data_ptr()[i] = attention_values[i];
    }
    for (size_t i = 0; i < values.numel(); ++i) {
        values.data_ptr()[i] = value_values[i];
    }

    const auto output = aggregate(attention, values);

    assert(nearly_equal(output(0, 0, 0), 30.0f));
    assert(nearly_equal(output(0, 0, 1), 31.0f));
    assert(nearly_equal(output(0, 1, 0), 10.0f));
    assert(nearly_equal(output(0, 1, 1), 11.0f));
    assert(nearly_equal(output(0, 2, 0), 20.0f));
    assert(nearly_equal(output(0, 2, 1), 21.0f));
}

void test_uniform_attention_averages_values() {
    Tensor<float, 3> attention({1, 2, 3});
    Tensor<float, 3> values({1, 3, 2});
    for (size_t i = 0; i < attention.numel(); ++i) {
        attention.data_ptr()[i] = 1.0f / 3.0f;
    }
    const float value_values[] = {3.0f, 6.0f, 6.0f, 9.0f, 9.0f, 12.0f};
    for (size_t i = 0; i < values.numel(); ++i) {
        values.data_ptr()[i] = value_values[i];
    }

    const auto output = aggregate(attention, values);

    for (size_t time = 0; time < 2; ++time) {
        assert(nearly_equal(output(0, time, 0), 6.0f));
        assert(nearly_equal(output(0, time, 1), 9.0f));
    }
}

void test_zero_values_produce_zero_output() {
    Tensor<float, 3> attention({1, 2, 2});
    Tensor<float, 3> values({1, 2, 3});
    const float attention_values[] = {0.3f, 0.7f, 0.6f, 0.4f};
    for (size_t i = 0; i < attention.numel(); ++i) {
        attention.data_ptr()[i] = attention_values[i];
    }

    const auto output = aggregate(attention, values);

    for (size_t i = 0; i < output.numel(); ++i) {
        assert(nearly_equal(output.data_ptr()[i], 0.0f));
    }
}

void test_shape_mismatches_throw() {
    Tensor<float, 3> attention({2, 2, 2});
    Tensor<float, 3> wrong_batch_values({3, 2, 3});
    Tensor<float, 3> wrong_time_values({2, 3, 3});

    bool batch_threw = false;
    try {
        const auto output = aggregate(attention, wrong_batch_values);
        (void)output;
    } catch (const std::invalid_argument&) {
        batch_threw = true;
    }
    assert(batch_threw);

    bool time_threw = false;
    try {
        const auto output = aggregate(attention, wrong_time_values);
        (void)output;
    } catch (const std::invalid_argument&) {
        time_threw = true;
    }
    assert(time_threw);
}

void test_inputs_remain_unchanged() {
    Tensor<float, 3> attention({1, 2, 2});
    Tensor<float, 3> values({1, 2, 2});
    const float attention_values[] = {0.25f, 0.75f, 0.5f, 0.5f};
    const float value_values[] = {1.0f, -2.0f, 3.0f, -4.0f};
    for (size_t i = 0; i < attention.numel(); ++i) {
        attention.data_ptr()[i] = attention_values[i];
        values.data_ptr()[i] = value_values[i];
    }

    const auto output = aggregate(attention, values);
    (void)output;

    for (size_t i = 0; i < attention.numel(); ++i) {
        assert(attention.data_ptr()[i] == attention_values[i]);
        assert(values.data_ptr()[i] == value_values[i]);
    }
}

} // namespace

int main() {
    test_hand_computable_values();
    test_multiple_batches();
    test_time_and_channel_dimensions_can_differ();
    test_one_hot_attention_rows_select_values();
    test_uniform_attention_averages_values();
    test_zero_values_produce_zero_output();
    test_shape_mismatches_throw();
    test_inputs_remain_unchanged();

    std::puts("test_attention_aggregation passed");
    return 0;
}
