#include "tensor.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << __FILE__ << ':' << __LINE__                           \
                      << ": CHECK failed: " #condition << '\n';               \
            ++failures;                                                        \
        }                                                                       \
    } while (false)

template <typename Function>
void check_throws_out_of_range(Function&& function) {
    try {
        function();
        CHECK(false && "Expected std::out_of_range");
    } catch (const std::out_of_range&) {
    } catch (const std::exception& error) {
        std::cerr << "Expected std::out_of_range, got: " << error.what() << '\n';
        ++failures;
    }
}

void test_2d_token_embedding_lookup() {
    Tensor<float, 2> embedding_table({3, 2});
    embedding_table(0, 0) = 0.1f;
    embedding_table(0, 1) = 0.2f;
    embedding_table(1, 0) = 1.1f;
    embedding_table(1, 1) = 1.2f;
    embedding_table(2, 0) = 2.1f;
    embedding_table(2, 1) = 2.2f;

    const Tensor<float, 2>& wte = embedding_table;
    const std::vector<float> values_before(wte.data_ptr(),
                                           wte.data_ptr() + wte.numel());

    const size_t vocab_size = wte.shape()[0];
    const size_t embedding_size = wte.shape()[1];

    Tensor<int, 2> tokens({2, 3});
    tokens(0, 0) = 2;
    tokens(0, 1) = 0;
    tokens(0, 2) = 2;
    tokens(1, 0) = 1;
    tokens(1, 1) = 2;
    tokens(1, 2) = 0;

    const size_t batch_size = tokens.shape()[0];
    const size_t sequence_length = tokens.shape()[1];
    const auto out = wte[tokens];

    CHECK(out.shape()[0] == batch_size);
    CHECK(out.shape()[1] == sequence_length);
    CHECK(out.shape()[2] == embedding_size);

    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t token = 0; token < sequence_length; ++token) {
            for (size_t channel = 0; channel < embedding_size; ++channel) {
                CHECK(out(batch, token, channel) == wte(tokens(batch, token), channel));
            }
        }
    }

    CHECK(vocab_size == wte.shape()[0]);
    for (size_t i = 0; i < wte.numel(); ++i) {
        CHECK(wte.data_ptr()[i] == values_before[i]);
    }
}

void test_invalid_token_ids_are_rejected() {
    Tensor<float, 2> embedding_table({3, 2});
    const Tensor<float, 2>& wte = embedding_table;
    const size_t vocab_size = wte.shape()[0];

    Tensor<int, 2> negative_tokens({1, 1});
    negative_tokens(0, 0) = -1;
    check_throws_out_of_range([&]() {
        const auto ignored = wte[negative_tokens];
        (void)ignored;
    });

    Tensor<int, 2> tokens_at_vocab_size({1, 1});
    tokens_at_vocab_size(0, 0) = static_cast<int>(vocab_size);
    check_throws_out_of_range([&]() {
        const auto ignored = wte[tokens_at_vocab_size];
        (void)ignored;
    });
}

}  // namespace

int main() {
    test_2d_token_embedding_lookup();
    test_invalid_token_ids_are_rejected();

    if (failures != 0) {
        std::cerr << failures << " token embedding test(s) failed\n";
    }

    return failures == 0 ? 0 : 1;
}
