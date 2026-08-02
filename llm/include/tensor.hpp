#pragma once

#include <vector>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <filesystem>


enum class DType : uint32_t {
    Float32 = 0,
    Int32 = 1,
    Int64 = 2,
    UInt8 = 3,
    Int8 = 4,
    Float64 = 5
};

// Enables coordinate-access overloads only when there is one integral index
// for every tensor dimension.
template <std::size_t Rank, typename... Indices>
using enable_indices_t = std::enable_if_t<
    sizeof...(Indices) == Rank &&
    (std::is_integral_v<std::decay_t<Indices>> && ...),
    int>;
template <typename T, std::size_t Rank>
class Tensor {
public:
    // Construction and ownership. Rank 0 is a scalar tensor with one element.
    explicit Tensor(const std::array<size_t, Rank>& shape);
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // A half-open range [begin, end) used by slice().
    using Slice = std::pair<size_t, size_t>;

    // Element access. The number of coordinates must equal Rank.
    template <typename... Indices, enable_indices_t<Rank, Indices...> = 0>
    T& operator()(Indices... indices);

    template <typename... Indices,
              enable_indices_t<Rank, Indices...> = 0>
    const T& operator()(Indices... indices) const;

    template <typename IndexType, std::size_t IndexRank>
    Tensor<T, Rank - 1 + IndexRank> operator[](const Tensor<IndexType, IndexRank>& indices) const;

    // Tensor operations
    Tensor slice(const std::array<Slice, Rank>& ranges) const;
    Tensor slice(std::initializer_list<Slice> ranges) const;

    Tensor operator+(const Tensor& other) const;

    // Tensor metadata and storage access
    const std::array<size_t, Rank>& shape() const;
    size_t numel() const;

    const T& item() const;

    T* data_ptr();
    const T* data_ptr() const;

    void save(const std::filesystem::path& path) const;

    static Tensor load(const std::filesystem::path& path);

private:
    struct Header {
        uint32_t magic = 0x54454E53; // "TENS"
        uint32_t version = 1;
        DType dtype;
        uint32_t rank;
    };

    Header header_;
    size_t size_;
    std::vector<T> data_;
    std::array<size_t, Rank> shape_;
    std::array<size_t, Rank> stride_;
};

#include "tensor.tpp"
