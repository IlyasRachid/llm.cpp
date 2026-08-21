#include <array>

template <typename T>
QKV<T> split_qkv(const Tensor<T, 3>& packed) {
    const auto& shape = packed.shape();
    std::size_t B = shape[0];
    std::size_t T_ = shape[1];
    std::size_t C3 = shape[2];

    if (C3 % 3 != 0) {
        throw std::invalid_argument("split_qkv: last dim must be divisible by 3");
    }
    std::size_t C = C3 / 3;

    Tensor<T,3> Q = packed.slice({ {0,B}, {0,T_}, {0,   C} });
    Tensor<T,3> K = packed.slice({ {0,B}, {0,T_}, {C,  2*C} });
    Tensor<T,3> V = packed.slice({ {0,B}, {0,T_}, {2*C, 3*C} });

    return {std::move(Q), std::move(K), std::move(V)};
}