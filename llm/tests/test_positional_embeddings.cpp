#include <cassert>
#include <cmath>
#include <cstdio>

#include "embeddings.hpp"

namespace {

constexpr float kEps = 1e-6f;

bool nearly_equal(float a, float b) {
    return std::fabs(a - b) < kEps;
}

Tensor<float, 3> make_token_embedding(size_t B, size_t T, size_t C) {
    Tensor<float, 3> t({B, T, C});
    for (size_t b = 0; b < B; b++)
        for (size_t i = 0; i < T; i++)
            for (size_t c = 0; c < C; c++)
                t(b, i, c) = static_cast<float>(100 * b + 10 * i + c);
    return t;
}

Tensor<float, 2> make_positional_embedding(size_t T, size_t C) {
    Tensor<float, 2> p({T, C});
    for (size_t i = 0; i < T; i++)
        for (size_t c = 0; c < C; c++)
            p(i, c) = static_cast<float>(1000 + 10 * i + c);
    return p;
}

Tensor<float, 3> make_filled(size_t B, size_t T, size_t C, float value) {
    Tensor<float, 3> t({B, T, C});
    for (size_t b = 0; b < B; b++)
        for (size_t i = 0; i < T; i++)
            for (size_t c = 0; c < C; c++)
                t(b, i, c) = value;
    return t;
}

Tensor<float, 2> make_filled2(size_t T, size_t C, float value) {
    Tensor<float, 2> t({T, C});
    for (size_t i = 0; i < T; i++)
        for (size_t c = 0; c < C; c++)
            t(i, c) = value;
    return t;
}

} // namespace

void test_output_shape_matches_token_embedding() {
    auto tok = make_token_embedding(2, 3, 5); // T != C on purpose
    auto pos = make_positional_embedding(3, 5);
    auto out = positional_embedding(tok, pos);
    assert(out.shape()[0] == 2);
    assert(out.shape()[1] == 3);
    assert(out.shape()[2] == 5);
    assert(out.numel() == 2 * 3 * 5);
    std::puts("test_output_shape_matches_token_embedding passed");
}

void test_all_channels_written_when_C_greater_than_T() {
    size_t B = 1, T = 2, C = 5;
    auto tok = make_token_embedding(B, T, C);
    auto pos = make_positional_embedding(T, C);
    auto out = positional_embedding(tok, pos);
    for (size_t t = 0; t < T; t++)
        for (size_t c = 0; c < C; c++) {
            float expected = tok(0, t, c) + pos(t, c);
            assert(nearly_equal(out(0, t, c), expected));
        }
    std::puts("test_all_channels_written_when_C_greater_than_T passed");
}

void test_correct_when_C_less_than_T() {
    size_t B = 1, T = 5, C = 2;
    auto tok = make_token_embedding(B, T, C);
    auto pos = make_positional_embedding(T, C);
    auto out = positional_embedding(tok, pos);
    for (size_t t = 0; t < T; t++)
        for (size_t c = 0; c < C; c++) {
            float expected = tok(0, t, c) + pos(t, c);
            assert(nearly_equal(out(0, t, c), expected));
        }
    std::puts("test_correct_when_C_less_than_T passed");
}

void test_elementwise_correctness_multi_batch() {
    size_t B = 3, T = 4, C = 6;
    auto tok = make_token_embedding(B, T, C);
    auto pos = make_positional_embedding(T, C);
    auto out = positional_embedding(tok, pos);
    for (size_t b = 0; b < B; b++)
        for (size_t t = 0; t < T; t++)
            for (size_t c = 0; c < C; c++) {
                float expected = tok(b, t, c) + pos(t, c);
                assert(nearly_equal(out(b, t, c), expected));
            }
    std::puts("test_elementwise_correctness_multi_batch passed");
}

void test_positional_embedding_broadcasts_identically_across_batches() {
    size_t B = 4, T = 3, C = 3;
    auto tok = make_token_embedding(B, T, C);
    auto pos = make_positional_embedding(T, C);
    auto out = positional_embedding(tok, pos);
    for (size_t t = 0; t < T; t++)
        for (size_t c = 0; c < C; c++) {
            float added_first = out(0, t, c) - tok(0, t, c);
            for (size_t b = 1; b < B; b++) {
                float added_b = out(b, t, c) - tok(b, t, c);
                assert(nearly_equal(added_first, added_b));
            }
        }
    std::puts("test_positional_embedding_broadcasts_identically_across_batches passed");
}

void test_zero_positional_embedding_is_identity() {
    size_t B = 2, T = 3, C = 4;
    auto tok = make_token_embedding(B, T, C);
    auto pos = make_filled2(T, C, 0.0f);
    auto out = positional_embedding(tok, pos);
    for (size_t b = 0; b < B; b++)
        for (size_t t = 0; t < T; t++)
            for (size_t c = 0; c < C; c++)
                assert(nearly_equal(out(b, t, c), tok(b, t, c)));
    std::puts("test_zero_positional_embedding_is_identity passed");
}

void test_zero_token_embedding_yields_broadcast_positional() {
    size_t B = 3, T = 2, C = 2;
    auto tok = make_filled(B, T, C, 0.0f);
    auto pos = make_positional_embedding(T, C);
    auto out = positional_embedding(tok, pos);
    for (size_t b = 0; b < B; b++)
        for (size_t t = 0; t < T; t++)
            for (size_t c = 0; c < C; c++)
                assert(nearly_equal(out(b, t, c), pos(t, c)));
    std::puts("test_zero_token_embedding_yields_broadcast_positional passed");
}

void test_minimal_shape() {
    Tensor<float, 3> tok({1, 1, 1});
    tok(0, 0, 0) = 2.5f;
    Tensor<float, 2> pos({1, 1});
    pos(0, 0) = 1.5f;
    auto out = positional_embedding(tok, pos);
    assert(out.shape()[0] == 1 && out.shape()[1] == 1 && out.shape()[2] == 1);
    assert(nearly_equal(out(0, 0, 0), 4.0f));
    std::puts("test_minimal_shape passed");
}

void test_inputs_are_not_modified() {
    size_t B = 2, T = 3, C = 3;
    auto tok = make_token_embedding(B, T, C);
    auto pos = make_positional_embedding(T, C);
    Tensor<float, 3> tok_before = make_token_embedding(B, T, C);
    Tensor<float, 2> pos_before = make_positional_embedding(T, C);

    auto out = positional_embedding(tok, pos);
    (void)out;

    for (size_t b = 0; b < B; b++)
        for (size_t t = 0; t < T; t++)
            for (size_t c = 0; c < C; c++)
                assert(nearly_equal(tok(b, t, c), tok_before(b, t, c)));
    for (size_t t = 0; t < T; t++)
        for (size_t c = 0; c < C; c++)
            assert(nearly_equal(pos(t, c), pos_before(t, c)));
    std::puts("test_inputs_are_not_modified passed");
}

// Rank == 2: token_embedding is [B, C], positional_embedding is [C].
void test_rank2_generic() {
    size_t B = 3, C = 5;
    Tensor<float, 2> tok({B, C});
    for (size_t b = 0; b < B; b++)
        for (size_t c = 0; c < C; c++)
            tok(b, c) = static_cast<float>(100 * b + c);

    Tensor<float, 1> pos({C});
    for (size_t c = 0; c < C; c++)
        pos(c) = static_cast<float>(1000 + c);

    auto out = positional_embedding(tok, pos);

    assert(out.shape()[0] == B && out.shape()[1] == C);
    for (size_t b = 0; b < B; b++)
        for (size_t c = 0; c < C; c++)
            assert(nearly_equal(out(b, c), tok(b, c) + pos(c)));
    std::puts("test_rank2_generic passed");
}

// Rank == 4: token_embedding is [B, T, H, C], positional_embedding is [T, H, C].
void test_rank4_generic() {
    size_t B = 2, T = 2, H = 3, C = 2;
    Tensor<float, 4> tok({B, T, H, C});
    for (size_t b = 0; b < B; b++)
        for (size_t t = 0; t < T; t++)
            for (size_t h = 0; h < H; h++)
                for (size_t c = 0; c < C; c++)
                    tok(b, t, h, c) = static_cast<float>(1000 * b + 100 * t + 10 * h + c);

    Tensor<float, 3> pos({T, H, C});
    for (size_t t = 0; t < T; t++)
        for (size_t h = 0; h < H; h++)
            for (size_t c = 0; c < C; c++)
                pos(t, h, c) = static_cast<float>(10000 + 100 * t + 10 * h + c);

    auto out = positional_embedding(tok, pos);

    assert(out.shape()[0] == B && out.shape()[1] == T &&
           out.shape()[2] == H && out.shape()[3] == C);
    for (size_t b = 0; b < B; b++)
        for (size_t t = 0; t < T; t++)
            for (size_t h = 0; h < H; h++)
                for (size_t c = 0; c < C; c++)
                    assert(nearly_equal(out(b, t, h, c),
                                         tok(b, t, h, c) + pos(t, h, c)));
    std::puts("test_rank4_generic passed");
}

void test_zero_batch_dimension() {
    size_t B = 0, T = 3, C = 4;
    Tensor<float, 3> tok({B, T, C});
    auto pos = make_positional_embedding(T, C);

    auto out = positional_embedding(tok, pos);

    assert(out.shape()[0] == 0);
    assert(out.shape()[1] == T);
    assert(out.shape()[2] == C);
    assert(out.numel() == 0);
    std::puts("test_zero_batch_dimension passed");
}

void test_zero_inner_dimension() {
    size_t B = 3, T = 2, C = 0;
    Tensor<float, 3> tok({B, T, C});
    Tensor<float, 2> pos({T, C});

    auto out = positional_embedding(tok, pos);

    assert(out.shape()[0] == B);
    assert(out.shape()[1] == T);
    assert(out.shape()[2] == 0);
    assert(out.numel() == 0);
    std::puts("test_zero_inner_dimension passed");
}

void test_zero_middle_dimension() {
    size_t B = 2, T = 0, C = 5;
    Tensor<float, 3> tok({B, T, C});
    Tensor<float, 2> pos({T, C});

    auto out = positional_embedding(tok, pos);

    assert(out.shape()[0] == B);
    assert(out.shape()[1] == 0);
    assert(out.shape()[2] == C);
    assert(out.numel() == 0);
    std::puts("test_zero_middle_dimension passed");
}

// Fully empty tensor: B=T=C=0.
void test_all_dimensions_zero() {
    Tensor<float, 3> tok({0, 0, 0});
    Tensor<float, 2> pos({0, 0});

    auto out = positional_embedding(tok, pos);

    assert(out.numel() == 0);
    std::puts("test_all_dimensions_zero passed");
}

int main() {
    test_output_shape_matches_token_embedding();
    test_all_channels_written_when_C_greater_than_T();
    test_correct_when_C_less_than_T();
    test_elementwise_correctness_multi_batch();
    test_positional_embedding_broadcasts_identically_across_batches();
    test_zero_positional_embedding_is_identity();
    test_zero_token_embedding_yields_broadcast_positional();
    test_minimal_shape();
    test_inputs_are_not_modified();
    test_rank2_generic();
    test_rank4_generic();
    test_zero_batch_dimension();
    test_zero_inner_dimension();
    test_zero_middle_dimension();
    test_all_dimensions_zero();

    std::puts("All positional_embedding tests passed.");
    return 0;
}