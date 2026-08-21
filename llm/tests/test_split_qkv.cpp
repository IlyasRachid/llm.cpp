#include <array>
#include <cassert>
#include <cstdio>
#include <stdexcept>

#include "split_qkv.hpp"

namespace {

void test_exact_qkv_values() {
    Tensor<float, 3> packed({1, 1, 6});
    const float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    for (size_t i = 0; i < packed.numel(); ++i) {
        packed.data_ptr()[i] = values[i];
    }

    const auto qkv = split_qkv(packed);

    assert(qkv.Q.shape() == (std::array<size_t, 3>{1, 1, 2}));
    assert(qkv.K.shape() == (std::array<size_t, 3>{1, 1, 2}));
    assert(qkv.V.shape() == (std::array<size_t, 3>{1, 1, 2}));
    assert(qkv.Q(0, 0, 0) == 1.0f);
    assert(qkv.Q(0, 0, 1) == 2.0f);
    assert(qkv.K(0, 0, 0) == 3.0f);
    assert(qkv.K(0, 0, 1) == 4.0f);
    assert(qkv.V(0, 0, 0) == 5.0f);
    assert(qkv.V(0, 0, 1) == 6.0f);
}

void test_multiple_batches_and_positions() {
    Tensor<float, 3> packed({2, 2, 6});
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t time = 0; time < 2; ++time) {
            for (size_t channel = 0; channel < 2; ++channel) {
                const float base = static_cast<float>(100 * batch + 10 * time + channel);
                packed(batch, time, channel) = base;
                packed(batch, time, 2 + channel) = base + 20.0f;
                packed(batch, time, 4 + channel) = base + 40.0f;
            }
        }
    }

    const auto qkv = split_qkv(packed);

    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t time = 0; time < 2; ++time) {
            for (size_t channel = 0; channel < 2; ++channel) {
                const float base = static_cast<float>(100 * batch + 10 * time + channel);
                assert(qkv.Q(batch, time, channel) == base);
                assert(qkv.K(batch, time, channel) == base + 20.0f);
                assert(qkv.V(batch, time, channel) == base + 40.0f);
            }
        }
    }
}

void test_negative_values() {
    Tensor<float, 3> packed({1, 1, 3});
    packed(0, 0, 0) = -1.0f;
    packed(0, 0, 1) = -2.0f;
    packed(0, 0, 2) = -3.0f;

    const auto qkv = split_qkv(packed);

    assert(qkv.Q(0, 0, 0) == -1.0f);
    assert(qkv.K(0, 0, 0) == -2.0f);
    assert(qkv.V(0, 0, 0) == -3.0f);
}

void test_single_channel() {
    Tensor<float, 3> packed({1, 2, 3});
    const float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    for (size_t i = 0; i < packed.numel(); ++i) {
        packed.data_ptr()[i] = values[i];
    }

    const auto qkv = split_qkv(packed);

    assert(qkv.Q.shape() == (std::array<size_t, 3>{1, 2, 1}));
    assert(qkv.Q(0, 0, 0) == 1.0f);
    assert(qkv.K(0, 0, 0) == 2.0f);
    assert(qkv.V(0, 0, 0) == 3.0f);
    assert(qkv.Q(0, 1, 0) == 4.0f);
    assert(qkv.K(0, 1, 0) == 5.0f);
    assert(qkv.V(0, 1, 0) == 6.0f);
}

void test_non_divisible_final_dimension_throws() {
    Tensor<float, 3> packed({1, 1, 5});

    bool threw = false;
    try {
        const auto qkv = split_qkv(packed);
        (void)qkv;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_source_tensor_remains_unchanged() {
    Tensor<float, 3> packed({1, 2, 6});
    const float values[] = {
        1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f,
        7.0f, -8.0f, 9.0f, -10.0f, 11.0f, -12.0f,
    };
    for (size_t i = 0; i < packed.numel(); ++i) {
        packed.data_ptr()[i] = values[i];
    }

    const auto qkv = split_qkv(packed);
    (void)qkv;

    for (size_t i = 0; i < packed.numel(); ++i) {
        assert(packed.data_ptr()[i] == values[i]);
    }
}

} // namespace

int main() {
    test_exact_qkv_values();
    test_multiple_batches_and_positions();
    test_negative_values();
    test_single_channel();
    test_non_divisible_final_dimension_throws();
    test_source_tensor_remains_unchanged();

    std::puts("test_split_qkv passed");
    return 0;
}
