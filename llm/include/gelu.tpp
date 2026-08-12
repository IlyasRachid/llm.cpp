#include <cmath>

template <typename T, std::size_t Rank>
Tensor<T, Rank> gelu(const Tensor<T, Rank>& input) {
    size_t num_elems = input.numel();
    Tensor<T, Rank> output(input.shape());

    const T* in_ptr = input.data_ptr();
    T* out_ptr = output.data_ptr();

    for (size_t i = 0; i < num_elems; i++) {
        T x = in_ptr[i];
        out_ptr[i] = static_cast<T>(0.5) * x * (static_cast<T>(1) + std::erf(x / std::sqrt(static_cast<T>(2))));
    }

    return output;

}