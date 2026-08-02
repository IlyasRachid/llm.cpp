#include "tensor.hpp"

#include <cstdio>
#include <limits>
#include <stdexcept>

using Tensor3D = Tensor<float, 3>;
using Tensor2D = Tensor<float, 2>;

static int g_failures = 0;

#define CHECK(condition)                                                   \
    do {                                                                   \
        if (!(condition)) {                                                \
            std::printf("FAIL: %s (line %d)\n", #condition, __LINE__);   \
            ++g_failures;                                                  \
        }                                                                   \
    } while (false)

static Tensor3D make_tensor(size_t B, size_t T, size_t C) {
    return Tensor3D({B, T, C});
}

static void test_read_And_write() {
    Tensor3D tensor = make_tensor(2, 3, 4);
    for (size_t b = 0; b < 2; ++b) {
        for (size_t t = 0; t < 3; ++t) {
            for (size_t c = 0; c < 4; ++c) {
                tensor(b, t, c) = static_cast<float>(b * 100 + t * 10 + c);
            }
        }
    }

    tensor.save("tensor_test.bin");

    Tensor3D loaded = Tensor3D::load("tensor_test.bin");

    CHECK(loaded.shape() == tensor.shape());
    CHECK(loaded.numel() == tensor.numel());
    for (size_t b = 0; b < 2; ++b) {
        for (size_t t = 0; t < 3; ++t) {
            for (size_t c = 0; c < 4; ++c) {
                CHECK(loaded(b, t, c) == tensor(b, t, c));
            }
        }
    }   
}

static void check_addition() {
    Tensor3D tensor1 = make_tensor(2, 2, 2);
    Tensor3D tensor2 = make_tensor(2, 2, 2);
    
    tensor1(0, 0, 0) = 1.0f;
    tensor2(0, 0, 0) = 2.0f;

    Tensor3D result = tensor1 + tensor2;

    CHECK(result.shape()[0] == 2);
    CHECK(result.shape()[1] == 2);
    CHECK(result.shape()[2] == 2);
    CHECK(result.numel() == 8);
    CHECK(result(0, 0, 0) == 3.0f);

}

static void check_moving_operation() {
    Tensor3D tensor = make_tensor(2, 3, 4);
    tensor(0, 0, 0) = 1.0f;

    Tensor3D moved = std::move(tensor);
    CHECK(moved(0, 0, 0) == 1.0f);
}

static void check_indexing() {
    Tensor3D tensor = make_tensor(3,5,4);

    // catch the error when at least one index is not valid
    bool threw = false;
    try {
        tensor(-1, -1, -2);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_rank_zero() {
    Tensor<float, 0> tensor0d = Tensor<float, 0>({});
    CHECK(tensor0d.numel() == 1);
    tensor0d() = 42.0f;
    CHECK(tensor0d.item() == 42.0f);

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

static void test_rank0_scalar_policy() {
    // Rank 0 represents a scalar: it has exactly one element and no axes.
    Tensor<int, 0> scalar({});
    CHECK(scalar.numel() == 1);

    scalar() = 42;
    CHECK(scalar() == 42);

    const Tensor<int, 0>& view = scalar;
    CHECK(view() == 42);

    const auto copy = scalar.slice(std::array<Tensor<int, 0>::Slice, 0>{});
    CHECK(copy.numel() == 1);
    CHECK(copy() == 42);
}

static void test_int_tensor() {
    Tensor<int, 2> left({2, 2});
    Tensor<int, 2> right({2, 2});

    left(0, 0) = 1;
    left(0, 1) = 2;
    left(1, 0) = 3;
    left(1, 1) = 4;
    right(0, 0) = 10;
    right(0, 1) = 20;
    right(1, 0) = 30;
    right(1, 1) = 40;

    const auto sum = left + right;
    CHECK(sum(0, 0) == 11);
    CHECK(sum(0, 1) == 22);
    CHECK(sum(1, 0) == 33);
    CHECK(sum(1, 1) == 44);
}

static void test_shape_product_overflow() {
    bool threw = false;
    try {
        Tensor<float, 2> tensor({std::numeric_limits<size_t>::max(), 2});
    } catch (const std::overflow_error&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_addition_shape_mismatch() {
    Tensor<float, 2> left({2, 3});
    Tensor<float, 2> right({3, 2});

    bool threw = false;
    try {
        const auto sum = left + right;
        static_cast<void>(sum);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_rank2() {
    Tensor2D tensor({2, 3});
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

static void test_2d_vectors() {
    // Treat each row as a four-element vector.
    Tensor2D vectors({3, 4});
    for (size_t row = 0; row < 3; ++row) {
        for (size_t column = 0; column < 4; ++column) {
            vectors(row, column) =
                static_cast<float>(row * 10 + column);
        }
    }

    CHECK(vectors.numel() == 12);
    CHECK(vectors(0, 0) == 0.0f);
    CHECK(vectors(1, 2) == 12.0f);
    CHECK(vectors(2, 3) == 23.0f);

    // Rows are contiguous because Tensor uses row-major storage.
    const float* raw = vectors.data_ptr();
    CHECK(raw[0] == 0.0f);
    CHECK(raw[4] == 10.0f);
    CHECK(raw[11] == 23.0f);

    const Tensor2D& view = vectors;
    CHECK(view(2, 1) == 21.0f);

    bool threw = false;
    try {
        vectors(3, 0);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        vectors(0, -1);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);

    // Extract the middle two columns from every vector.
    const auto middle = vectors.slice({{0, 3}, {1, 3}});
    CHECK(middle.shape()[0] == 3);
    CHECK(middle.shape()[1] == 2);
    CHECK(middle(0, 0) == 1.0f);
    CHECK(middle(1, 1) == 12.0f);
    CHECK(middle(2, 0) == 21.0f);
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
    test_read_And_write();
    test_rank_zero();
    check_addition();
    check_moving_operation();
    check_indexing();
    test_shape_and_access();
    test_row_major_layout();
    test_slice();
    test_zero_sized_slice();
    test_rank1();
    test_rank0_scalar_policy();
    test_int_tensor();
    test_shape_product_overflow();
    test_addition_shape_mismatch();
    test_rank2();
    test_2d_vectors();
    test_rank4();
    return g_failures == 0 ? 0 : 1;
}
