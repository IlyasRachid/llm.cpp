#pragma once

#include <vector>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <utility>

template <typename T, std::size_t Rank>
class Tensor {
public:
    explicit Tensor(const std::array<size_t, Rank>& shape);
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    using Slice = std::pair<size_t, size_t>;

    template <typename... Indices,
              std::enable_if_t<
                  sizeof...(Indices) == Rank &&
                  (std::is_integral_v<std::decay_t<Indices>> && ...),
                  int> = 0>
    T& operator()(Indices... indices) noexcept;

    template <typename... Indices,
              std::enable_if_t<
                  sizeof...(Indices) == Rank &&
                  (std::is_integral_v<std::decay_t<Indices>> && ...),
                  int> = 0>
    const T& operator()(Indices... indices) const noexcept;

    Tensor slice(const std::array<Slice, Rank>& ranges) const;
    Tensor slice(std::initializer_list<Slice> ranges) const;

    Tensor operator+(const Tensor& other) const;

    const std::array<size_t, Rank>& shape() const;
    size_t numel() const;

    const T* data_ptr() const;

private:
    size_t size_;
    std::vector<T> data_;
    std::array<size_t, Rank> shape_;
};

#include "tensor.tpp"
