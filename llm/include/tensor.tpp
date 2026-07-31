#pragma once

#include <algorithm> 
#include <limits>
#include <stdexcept>

template <typename T, std::size_t Rank>
Tensor<T, Rank>::Tensor(const std::array<size_t, Rank>& shape)
    : size_(1), shape_(shape) {
    for (size_t dimension : shape_) {
        if (dimension != 0 && size_ > std::numeric_limits<size_t>::max() / dimension) {
            throw std::overflow_error("Tensor shape size overflows size_t");
        }
        size_ *= dimension;
    }
    data_.resize(size_);
}

template <typename T, std::size_t Rank>
Tensor<T, Rank>::~Tensor() = default;

template <typename T, std::size_t Rank>
template <typename... Indices,
          std::enable_if_t<
              sizeof...(Indices) == Rank &&
              (std::is_integral_v<std::decay_t<Indices>> && ...),
              int>>
T& Tensor<T, Rank>::operator()(Indices... indices) noexcept {
    const std::array<size_t, Rank> index{
        static_cast<size_t>(indices)...
    };
    size_t offset = index[0];
    for (size_t dimension = 1; dimension < Rank; ++dimension) {
        offset = offset * shape_[dimension] + index[dimension];
    }
    return data_[offset];
}

template <typename T, std::size_t Rank>
template <typename... Indices,
          std::enable_if_t<
              sizeof...(Indices) == Rank &&
              (std::is_integral_v<std::decay_t<Indices>> && ...),
              int>>
const T& Tensor<T, Rank>::operator()(Indices... indices) const noexcept {
    const std::array<size_t, Rank> index{
        static_cast<size_t>(indices)...
    };
    size_t offset = index[0];
    for (size_t dimension = 1; dimension < Rank; ++dimension) {
        offset = offset * shape_[dimension] + index[dimension];
    }
    return data_[offset];
}

template <typename T, std::size_t Rank>
Tensor<T, Rank> Tensor<T, Rank>::slice(
    const std::array<Slice, Rank>& ranges) const {
    std::array<size_t, Rank> sliced_shape{};
    for (size_t dimension = 0; dimension < Rank; ++dimension) {
        const auto [begin, end] = ranges[dimension];
        if (begin > end || end > shape_[dimension]) {
            throw std::out_of_range("Tensor slice out of range");
        }
        sliced_shape[dimension] = end - begin;
    }

    Tensor result(sliced_shape);
    if (result.size_ == 0) {
        return result;
    }

    std::array<size_t, Rank> source_stride{};
    source_stride[Rank - 1] = 1;
    for (size_t dimension = Rank - 1; dimension != 0; --dimension) {
        source_stride[dimension - 1] =
            source_stride[dimension] * shape_[dimension];
    }

    const size_t block_size = sliced_shape[Rank - 1];
    const size_t block_count = result.size_ / block_size;
    for (size_t block = 0; block < block_count; ++block) {
        size_t remaining = block;
        size_t source_offset = ranges[Rank - 1].first;
        for (size_t dimension = Rank - 1; dimension != 0; --dimension) {
            const size_t outer_dimension = dimension - 1;
            const size_t coordinate =
                remaining % sliced_shape[outer_dimension];
            remaining /= sliced_shape[outer_dimension];
            source_offset +=
                (ranges[outer_dimension].first + coordinate) *
                source_stride[outer_dimension];
        }
        std::copy_n(data_.data() + source_offset, block_size,
                    result.data_.data() + block * block_size);
    }
    return result;
}

template <typename T, std::size_t Rank>
Tensor<T, Rank> Tensor<T, Rank>::slice(
    std::initializer_list<Slice> ranges) const {
    if (ranges.size() != Rank) {
        throw std::invalid_argument(
            "Tensor slice must specify one range per dimension");
    }
    std::array<Slice, Rank> range_array{};
    std::copy(ranges.begin(), ranges.end(), range_array.begin());
    return slice(range_array);
}

template <typename T, std::size_t Rank>
const std::array<size_t, Rank>& Tensor<T, Rank>::shape() const {
    return shape_;
}

template <typename T, std::size_t Rank>
size_t Tensor<T, Rank>::numel() const {
    return size_;
}

template <typename T, std::size_t Rank>
const T* Tensor<T, Rank>::data_ptr() const {
    return data_.data();
}
