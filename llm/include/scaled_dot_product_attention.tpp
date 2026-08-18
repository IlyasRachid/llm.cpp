#include <stdexcept>
#include <cmath>

template <typename T, std::size_t Rank>
Tensor<T, Rank> scaledDotProductScores(const Tensor<T, Rank>& query, const Tensor<T, Rank>& key) {
    static_assert(Rank >= 2, "Rank must be at least 2");

    std::size_t C = query.shape()[Rank - 1];
    if (C != key.shape()[Rank - 1]) {
        throw std::invalid_argument("query/key channel dims must match");
    }
    if (C == 0) {
        throw std::domain_error("Division by zero is not allowed");
    }

    Tensor<T, Rank> scores = query.matmul(key, /*transpose_b=*/true);
    scores.scale_(T(1) / std::sqrt(static_cast<T>(C)));
    return scores;
}