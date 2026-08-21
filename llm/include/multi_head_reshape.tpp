#include <stdexcept>

template <typename T, std::size_t Rank>
Tensor<T, Rank + 1> multi_head_reshape(const Tensor<T, Rank>& input, size_t H) {
    static_assert(Rank >= 2, "multi_head_reshape requires at least a sequence and channel dimension");

    if (H == 0) {
        throw std::invalid_argument("H must be nonzero");
    }

    size_t C = input.shape()[Rank-1];

    if (C % H != 0) {
        throw std::invalid_argument("number of heads must be a divisor of the channel dimension");
    }
    size_t D = C / H;
    std::array<std::size_t, 2> new_shape{H, D};
    Tensor<T, Rank + 1> reshaped = input.template reshape<2>(Rank-1, new_shape);
    Tensor<T, Rank + 1> transposed = reshaped.transpose(Rank-2, Rank-1);

    return transposed;
}