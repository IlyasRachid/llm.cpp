#include <array>
#include <cassert>
#include <cstdio>

#include "merge_multi_head.hpp"

namespace {

void test_hand_computable_mapping() {
    Tensor<float, 4> attention_heads({1, 2, 3, 2});
    for (size_t head = 0; head < 2; ++head) {
        for (size_t time = 0; time < 3; ++time) {
            for (size_t channel = 0; channel < 2; ++channel) {
                attention_heads(0, head, time, channel) =
                    static_cast<float>(100 * head + 10 * time + channel);
            }
        }
    }

    const auto output = merge_multi_head(attention_heads);

    assert(output.shape() == (std::array<size_t, 3>{1, 3, 4}));
    assert(output(0, 0, 0) == 0.0f);
    assert(output(0, 0, 1) == 1.0f);
    assert(output(0, 0, 2) == 100.0f);
    assert(output(0, 0, 3) == 101.0f);
    assert(output(0, 2, 0) == 20.0f);
    assert(output(0, 2, 1) == 21.0f);
    assert(output(0, 2, 2) == 120.0f);
    assert(output(0, 2, 3) == 121.0f);
}

void test_multiple_batches() {
    Tensor<float, 4> attention_heads({2, 2, 2, 2});
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t head = 0; head < 2; ++head) {
            for (size_t time = 0; time < 2; ++time) {
                for (size_t channel = 0; channel < 2; ++channel) {
                    attention_heads(batch, head, time, channel) =
                        static_cast<float>(1000 * batch + 100 * head +
                                           10 * time + channel);
                }
            }
        }
    }

    const auto output = merge_multi_head(attention_heads);

    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t head = 0; head < 2; ++head) {
            for (size_t time = 0; time < 2; ++time) {
                for (size_t channel = 0; channel < 2; ++channel) {
                    assert(output(batch, time, head * 2 + channel) ==
                           attention_heads(batch, head, time, channel));
                }
            }
        }
    }
}

void test_multiple_heads() {
    Tensor<float, 4> attention_heads({1, 3, 2, 2});
    for (size_t head = 0; head < 3; ++head) {
        for (size_t time = 0; time < 2; ++time) {
            for (size_t channel = 0; channel < 2; ++channel) {
                attention_heads(0, head, time, channel) =
                    static_cast<float>(100 * head + 10 * time + channel);
            }
        }
    }

    const auto output = merge_multi_head(attention_heads);

    assert(output.shape() == (std::array<size_t, 3>{1, 2, 6}));
    for (size_t head = 0; head < 3; ++head) {
        for (size_t time = 0; time < 2; ++time) {
            for (size_t channel = 0; channel < 2; ++channel) {
                assert(output(0, time, head * 2 + channel) ==
                       attention_heads(0, head, time, channel));
            }
        }
    }
}

void test_single_head() {
    Tensor<float, 4> attention_heads({1, 1, 2, 3});
    for (size_t i = 0; i < attention_heads.numel(); ++i) {
        attention_heads.data_ptr()[i] = static_cast<float>(i) + 1.0f;
    }

    const auto output = merge_multi_head(attention_heads);

    assert(output.shape() == (std::array<size_t, 3>{1, 2, 3}));
    for (size_t time = 0; time < 2; ++time) {
        for (size_t channel = 0; channel < 3; ++channel) {
            assert(output(0, time, channel) == attention_heads(0, 0, time, channel));
        }
    }
}

void test_single_head_dimension() {
    Tensor<float, 4> attention_heads({1, 2, 3, 1});
    for (size_t head = 0; head < 2; ++head) {
        for (size_t time = 0; time < 3; ++time) {
            attention_heads(0, head, time, 0) =
                static_cast<float>(10 * head + time);
        }
    }

    const auto output = merge_multi_head(attention_heads);

    assert(output.shape() == (std::array<size_t, 3>{1, 3, 2}));
    for (size_t head = 0; head < 2; ++head) {
        for (size_t time = 0; time < 3; ++time) {
            assert(output(0, time, head) == attention_heads(0, head, time, 0));
        }
    }
}

void test_source_tensor_remains_unchanged() {
    Tensor<float, 4> attention_heads({1, 2, 2, 2});
    const float values[] = {1.0f, -2.0f, 3.0f, -4.0f,
                            5.0f, -6.0f, 7.0f, -8.0f};
    for (size_t i = 0; i < attention_heads.numel(); ++i) {
        attention_heads.data_ptr()[i] = values[i];
    }

    const auto output = merge_multi_head(attention_heads);
    (void)output;

    for (size_t i = 0; i < attention_heads.numel(); ++i) {
        assert(attention_heads.data_ptr()[i] == values[i]);
    }
}

void test_shape_preservation() {
    Tensor<float, 4> attention_heads({2, 3, 4, 5});

    const auto output = merge_multi_head(attention_heads);

    assert(output.shape() == (std::array<size_t, 3>{2, 4, 15}));
    assert(output.numel() == attention_heads.numel());
}

} // namespace

int main() {
    test_hand_computable_mapping();
    test_multiple_batches();
    test_multiple_heads();
    test_single_head();
    test_single_head_dimension();
    test_source_tensor_remains_unchanged();
    test_shape_preservation();

    std::puts("test_merge_multi_head passed");
    return 0;
}
