#include "tensor.hpp"

#include <cstdio>
#include <stdexcept>

using Tensor3D = Tensor<float, 3>;

static int g_failures = 0;

#define CHECK(condition)                                                   \
    do {                                                                   \
        if (!(condition)) {                                                \
            std::printf("FAIL: %s (line %d)\n", #condition, __LINE__);   \
            ++g_failures;                                                  \
        }                                                                   \
    } while (false)

static Tensor3D make_tensor(size_t b, size_t t, size_t c) {
    return Tensor3D({b, t, c});
}

static void test_shape_and_access() {
    auto tensor = make_tensor(2, 3, 4);
    CHECK(tensor.numel() == 24);
    CHECK(tensor.shape()[0] == 2);
    CHECK(tensor.shape()[1] == 3);
    CHECK(tensor.shape()[2] == 4);

    for (size_t b = 0; b < 2; ++b) {
        for (size_t t = 0; t < 3; ++t) {
            for (size_t c = 0; c < 4; ++c) {
                tensor(b, t, c) = static_cast<float>(b * 100 + t * 10 + c);
            }
        }
    }

    CHECK(tensor(0, 0, 0) == 0.0f);
    CHECK(tensor(1, 2, 3) == 123.0f);
    const Tensor3D& view = tensor;
    CHECK(view(1, 2, 3) == 123.0f);
}

static void test_row_major_layout() {
    auto tensor = make_tensor(2, 3, 4);
    for (size_t b = 0; b < 2; ++b) {
        for (size_t t = 0; t < 3; ++t) {
            for (size_t c = 0; c < 4; ++c) {
                tensor(b, t, c) = static_cast<float>((b * 3 + t) * 4 + c);
            }
        }
    }

    const float* raw = tensor.data_ptr();
    for (size_t i = 0; i < tensor.numel(); ++i) {
        CHECK(raw[i] == static_cast<float>(i));
    }
}

static void test_slice() {
    auto tensor = make_tensor(2, 3, 4);
    for (size_t b = 0; b < 2; ++b) {
        for (size_t t = 0; t < 3; ++t) {
            for (size_t c = 0; c < 4; ++c) {
                tensor(b, t, c) = static_cast<float>(b * 100 + t * 10 + c);
            }
        }
    }

    auto sliced = tensor.slice({{1, 2}, {1, 3}, {1, 4}});
    CHECK(sliced.shape()[0] == 1);
    CHECK(sliced.shape()[1] == 2);
    CHECK(sliced.shape()[2] == 3);
    CHECK(sliced.numel() == 6);
    CHECK(sliced(0, 0, 0) == 111.0f);
    CHECK(sliced(0, 1, 2) == 123.0f);

    bool threw = false;
    try {
        tensor.slice({{0, 3}, {0, 3}, {0, 4}});
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_zero_sized_slice() {
    auto tensor = make_tensor(2, 3, 4);
    auto sliced = tensor.slice({{0, 0}, {0, 3}, {0, 4}});
    CHECK(sliced.numel() == 0);
    CHECK(sliced.shape()[0] == 0);
}

static void test_rank1() {
    Tensor<float, 1> tensor({5});
    for (size_t i = 0; i < 5; ++i) tensor(i) = static_cast<float>(i * 2);
    CHECK(tensor.numel() == 5);
    CHECK(tensor(3) == 6.0f);

    auto sliced = tensor.slice({{1, 4}});
    CHECK(sliced.shape()[0] == 3);
    CHECK(sliced(0) == 2.0f);
    CHECK(sliced(2) == 6.0f);
}

static void test_rank2() {
    Tensor<float, 2> tensor({2, 3});
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            tensor(i, j) = static_cast<float>(i * 10 + j);
    CHECK(tensor.numel() == 6);
    CHECK(tensor(1, 2) == 12.0f);

    auto sliced = tensor.slice({{0, 2}, {1, 3}});
    CHECK(sliced.shape()[0] == 2);
    CHECK(sliced.shape()[1] == 2);
    CHECK(sliced(0, 0) == 1.0f);
    CHECK(sliced(1, 1) == 12.0f);
}

static void test_rank4() {
    Tensor<float, 4> tensor({2, 2, 2, 3});
    for (size_t a = 0; a < 2; ++a)
        for (size_t b = 0; b < 2; ++b)
            for (size_t c = 0; c < 2; ++c)
                for (size_t d = 0; d < 3; ++d)
                    tensor(a, b, c, d) =
                        static_cast<float>(a * 1000 + b * 100 + c * 10 + d);
    CHECK(tensor.numel() == 24);
    CHECK(tensor(1, 0, 1, 2) == 1012.0f);

    auto sliced = tensor.slice({{1, 2}, {0, 2}, {0, 2}, {1, 3}});
    CHECK(sliced.shape()[0] == 1);
    CHECK(sliced.shape()[1] == 2);
    CHECK(sliced.shape()[2] == 2);
    CHECK(sliced.shape()[3] == 2);
    CHECK(sliced(0, 0, 0, 0) == 1001.0f);
    CHECK(sliced(0, 1, 1, 1) == 1112.0f);
}

int main() {
    test_shape_and_access();
    test_row_major_layout();
    test_slice();
    test_zero_sized_slice();
    test_rank1();
    test_rank2();
    test_rank4();
    return g_failures == 0 ? 0 : 1;
}
