#include <array>
#include <cassert>
#include <cstdio>
#include <stdexcept>

#include "multi_head_reshape.hpp"

namespace {

void test_hand_filled_index_mapping() {
    Tensor<float, 3> input({1, 2, 4});
    for (size_t time = 0; time < 2; ++time) {
        for (size_t channel = 0; channel < 4; ++channel) {
            input(0, time, channel) = static_cast<float>(10 * time + channel);
        }
    }

    const auto output = multi_head_reshape(input, 2);

    assert(output.shape() == (std::array<size_t, 4>{1, 2, 2, 2}));
    assert(output(0, 0, 0, 0) == 0.0f);
    assert(output(0, 0, 0, 1) == 1.0f);
    assert(output(0, 1, 0, 0) == 2.0f);
    assert(output(0, 1, 0, 1) == 3.0f);
    assert(output(0, 0, 1, 0) == 10.0f);
    assert(output(0, 0, 1, 1) == 11.0f);
    assert(output(0, 1, 1, 0) == 12.0f);
    assert(output(0, 1, 1, 1) == 13.0f);
}

void test_multiple_batches() {
    Tensor<float, 3> input({2, 2, 4});
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t time = 0; time < 2; ++time) {
            for (size_t channel = 0; channel < 4; ++channel) {
                input(batch, time, channel) =
                    static_cast<float>(100 * batch + 10 * time + channel);
            }
        }
    }

    const auto output = multi_head_reshape(input, 2);

    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t head = 0; head < 2; ++head) {
            for (size_t time = 0; time < 2; ++time) {
                for (size_t channel = 0; channel < 2; ++channel) {
                    assert(output(batch, head, time, channel) ==
                           input(batch, time, head * 2 + channel));
                }
            }
        }
    }
}

void test_multiple_heads() {
    Tensor<float, 3> input({1, 2, 6});
    for (size_t time = 0; time < 2; ++time) {
        for (size_t channel = 0; channel < 6; ++channel) {
            input(0, time, channel) = static_cast<float>(10 * time + channel);
        }
    }

    const auto output = multi_head_reshape(input, 3);

    assert(output.shape() == (std::array<size_t, 4>{1, 3, 2, 2}));
    for (size_t head = 0; head < 3; ++head) {
        for (size_t time = 0; time < 2; ++time) {
            for (size_t channel = 0; channel < 2; ++channel) {
                assert(output(0, head, time, channel) ==
                       input(0, time, head * 2 + channel));
            }
        }
    }
}

void test_single_head() {
    Tensor<float, 3> input({1, 2, 3});
    const float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    for (size_t i = 0; i < input.numel(); ++i) {
        input.data_ptr()[i] = values[i];
    }

    const auto output = multi_head_reshape(input, 1);

    assert(output.shape() == (std::array<size_t, 4>{1, 1, 2, 3}));
    for (size_t time = 0; time < 2; ++time) {
        for (size_t channel = 0; channel < 3; ++channel) {
            assert(output(0, 0, time, channel) == input(0, time, channel));
        }
    }
}

void test_channels_equal_heads() {
    Tensor<float, 3> input({1, 2, 3});
    const float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    for (size_t i = 0; i < input.numel(); ++i) {
        input.data_ptr()[i] = values[i];
    }

    const auto output = multi_head_reshape(input, 3);

    assert(output.shape() == (std::array<size_t, 4>{1, 3, 2, 1}));
    for (size_t head = 0; head < 3; ++head) {
        for (size_t time = 0; time < 2; ++time) {
            assert(output(0, head, time, 0) == input(0, time, head));
        }
    }
}

void test_non_divisible_channel_count_throws() {
    Tensor<float, 3> input({1, 2, 5});

    bool threw = false;
    try {
        const auto output = multi_head_reshape(input, 2);
        (void)output;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_source_tensor_remains_unchanged() {
    Tensor<float, 3> input({1, 2, 4});
    const float values[] = {1.0f, -2.0f, 3.0f, -4.0f,
                            5.0f, -6.0f, 7.0f, -8.0f};
    for (size_t i = 0; i < input.numel(); ++i) {
        input.data_ptr()[i] = values[i];
    }

    const auto output = multi_head_reshape(input, 2);
    (void)output;

    for (size_t i = 0; i < input.numel(); ++i) {
        assert(input.data_ptr()[i] == values[i]);
    }
}

} // namespace

int main() {
    test_hand_filled_index_mapping();
    test_multiple_batches();
    test_multiple_heads();
    test_single_head();
    test_channels_equal_heads();
    test_non_divisible_channel_count_throws();
    test_source_tensor_remains_unchanged();

    std::puts("test_multi_head_reshape passed");
    return 0;
}
