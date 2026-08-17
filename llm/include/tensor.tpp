#pragma once

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <fstream>



// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

template <typename... Indices, std::size_t Rank>
static void check_index_range(const std::array<size_t, Rank>& shape, Indices... indices) {
    // checking the integers are not negative
    ((indices >= 0
        ? void()
        : throw std::out_of_range("Tensor index out of range")),
    ...);

    const std::array<size_t, Rank> index{
        static_cast<size_t>(indices)...
    };

    for (size_t dimension = 0; dimension < Rank; ++dimension) {
        if (index[dimension] >= shape[dimension]) {
            throw std::out_of_range("Tensor index out of range");
        }
    }
}

template <typename T>
static constexpr DType dtype_to_DType() {
    if constexpr (std::is_same_v<T, float>) {
        return DType::Float32;
    } else if constexpr (std::is_same_v<T, double>) {
        return DType::Float64;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return DType::Int32;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return DType::Int64;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return DType::UInt8;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return DType::Int8;
    } else {
        static_assert(std::is_same_v<T, void>, "Unsupported tensor data type");
    }
}

// -----------------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------------

template <typename T, std::size_t Rank>
Tensor<T, Rank>::Tensor(const std::array<size_t, Rank>& shape): size_(1), shape_(shape) {
    // Compute the element count while detecting multiplication overflow.
    for (size_t dimension : shape_) {
        if (dimension != 0 && size_ > std::numeric_limits<size_t>::max() / dimension) {
            throw std::overflow_error("Tensor shape size overflows size_t");
        }
        size_ *= dimension;
    }
    data_.resize(size_);

    header_.dtype = dtype_to_DType<T>();
    header_.rank = static_cast<uint32_t>(Rank);

    // A scalar has no strides. Higher-rank tensors use row-major strides.
    if constexpr (Rank > 0) {
        stride_[Rank - 1] = 1;
        for (size_t dimension = Rank - 1; dimension > 0; --dimension) {
            stride_[dimension - 1] = stride_[dimension] * shape_[dimension];
        }
    }
}

template <typename T, std::size_t Rank>
Tensor<T, Rank>::~Tensor() = default;

template <typename T, std::size_t Rank>
Tensor<T, Rank>::Tensor(Tensor&& other) noexcept = default;

template <typename T, std::size_t Rank>
Tensor<T, Rank>& Tensor<T, Rank>::operator=(Tensor&& other) noexcept = default;

// -----------------------------------------------------------------------------
// Element access
// -----------------------------------------------------------------------------

template <typename T, std::size_t Rank>
template <typename... Indices, enable_indices_t<Rank, Indices...>>
T& Tensor<T, Rank>::operator()(Indices... indices) {
    // Validate each coordinate while building its row-major flat offset.
    check_index_range(shape_, indices...);
    const std::array<size_t, Rank> index{
        static_cast<size_t>(indices)...
    };

    size_t offset = 0;

    for (size_t dimension = 0; dimension < Rank; ++dimension) {
        offset += index[dimension] * stride_[dimension];
    }

    return data_[offset];
}

template <typename T, std::size_t Rank>
template <typename... Indices, enable_indices_t<Rank, Indices...>>
const T& Tensor<T, Rank>::operator()(Indices... indices) const {
    // Keep the same validation and row-major addressing as mutable access.
    check_index_range(shape_, indices...);
    const std::array<size_t, Rank> index{
        static_cast<size_t>(indices)...
    };

    size_t offset = 0;

    for (size_t dimension = 0; dimension < Rank; ++dimension) {
        offset += index[dimension] * stride_[dimension];
    }

    return data_[offset];
}

template <typename T, std::size_t Rank>
template <typename IndexType, std::size_t IndexRank>
Tensor<T, Rank - 1 + IndexRank> Tensor<T, Rank>::operator[](const Tensor<IndexType, IndexRank>& indices) const {
    static_assert(std::is_integral_v<IndexType>, "Index tensor must have integral type");
    static_assert(Rank >= 1, "Tensor rank must be at least 1 for indexing");
    static_assert(IndexRank >= 1, "Index tensor must have rank greater than 0");

    // The output tensor has rank Rank - 1 + IndexRank.
    std::array<size_t, Rank - 1 + IndexRank> output_shape{};

    std::copy(indices.shape().begin(), indices.shape().end(), output_shape.begin());
    std::copy(shape_.begin() + 1, shape_.end(), output_shape.begin() + IndexRank);

    Tensor<T, Rank - 1 + IndexRank> output(output_shape);

    size_t slice_size = this->numel() / shape_[0];

    const IndexType* index_data = indices.data_ptr();

    for (size_t i = 0; i < indices.numel(); ++i) {
        const IndexType raw_index = index_data[i];
        if constexpr (std::is_signed_v<IndexType>) {
            if (raw_index < 0) {
                throw std::out_of_range("Index tensor contains out-of-bounds index");
            }
        }

        if (static_cast<size_t>(raw_index) >= shape_[0]) {
            throw std::out_of_range("Index tensor contains out-of-bounds index");
        }

        size_t index = static_cast<size_t>(raw_index);

        const T* src_ptr = this->data_ptr() + index * slice_size;
        T* dst_ptr = output.data_ptr() + i * slice_size;

        std::copy_n(src_ptr, slice_size, dst_ptr);
    }

    return output;
}


// -----------------------------------------------------------------------------
// Slicing
// -----------------------------------------------------------------------------

template <typename T, std::size_t Rank>
Tensor<T, Rank> Tensor<T, Rank>::slice(const std::array<Slice, Rank>& ranges) const {
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

    if constexpr (Rank == 0) {
        // Slicing a scalar with no ranges returns an independent scalar.
        result.data_[0] = data_[0];
    } else {
        // Copy contiguous runs in the final dimension for each outer coordinate.
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
                    stride_[outer_dimension];
            }
            std::copy_n(data_.data() + source_offset, block_size,
                        result.data_.data() + block * block_size);
        }
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

// -----------------------------------------------------------------------------
// Addition
// -----------------------------------------------------------------------------

template <typename T, std::size_t Rank>
Tensor<T, Rank> Tensor<T,Rank>::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument(
            "Tensor addition requires matching shapes");
    }

    Tensor result(shape_);
    for (size_t i = 0; i < size_; i++) {
        result.data_[i] = data_[i] + other.data_[i]; 
    }

    return result;
}

// -----------------------------------------------------------------------------
// Multiplication
// -----------------------------------------------------------------------------

template <typename T, std::size_t Rank>
Tensor<T, Rank> Tensor<T, Rank>::matmul(const Tensor& other, bool transpose_b) const {
    static_assert(Rank >= 2, "Rank must be at least 2");

    auto shape = this->shape();
    auto other_shape = other.shape();

    for (std::size_t i = 0; i < Rank - 2; i++) {
        if (shape[i] != other_shape[i]) {
            throw std::invalid_argument("batch shapes are not equal");
        }
    }

    std::size_t M = shape[Rank - 2];
    std::size_t K = shape[Rank - 1];
    
    // if transpose_b, "other" is logically [..., N, K] and we read it as such
    std::size_t N   = transpose_b ? other_shape[Rank - 2] : other_shape[Rank - 1];
    std::size_t K2  = transpose_b ? other_shape[Rank - 1] : other_shape[Rank - 2];

    if (K != K2) {
        throw std::invalid_argument("inner dimensions must match");
    }

    auto out_shape = shape;
    out_shape[Rank - 1] = N;
    Tensor<T, Rank> output(out_shape);

    size_t batch_size = 1;
    for (size_t i = 0; i < Rank-2; ++i) { batch_size *= shape[i]; }

    const auto& a_strides = this->stride();
    const auto& b_strides = other.stride();
    const auto& c_strides = output.stride();

    const T* a_data = this->data_ptr();
    const T* b_data = other.data_ptr();
    T* c_data = output.data_ptr();

    std::array<std::size_t, Rank> curr_idx{};

    for (size_t b = 0; b < batch_size; b++) {
        std::size_t a_batch_off = 0, b_batch_off = 0, c_batch_off = 0;
        size_t rem = b;
        for (size_t d = Rank-2; d-- > 0; ) {
            curr_idx[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }

        for (size_t i = 0; i < Rank-2; i++) {
            a_batch_off += curr_idx[i] * a_strides[i];
            b_batch_off += curr_idx[i] * b_strides[i];
            c_batch_off += curr_idx[i] * c_strides[i];
        }

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                T total = 0;
                for (size_t k = 0; k < K; k++) {
                    size_t a_off = a_batch_off + i * a_strides[Rank-2] + k * a_strides[Rank-1];
                    size_t b_off = transpose_b 
                    ? b_batch_off + j * b_strides[Rank-2] + k * b_strides[Rank-1]
                    : b_batch_off + k * b_strides[Rank-2] + j * b_strides[Rank-1];
                    total += a_data[a_off] * b_data[b_off];
                }
                size_t c_off = c_batch_off + i * c_strides[Rank-2] + j * c_strides[Rank-1];
                c_data[c_off] = total;
            }
        }
    }
    return output;
}

// -----------------------------------------------------------------------------
// Metadata
// -----------------------------------------------------------------------------

template <typename T, std::size_t Rank>
const std::array<size_t, Rank>& Tensor<T, Rank>::shape() const {
    return shape_;
}

template <typename T, std::size_t Rank>
size_t Tensor<T, Rank>::numel() const {
    return size_;
}

template <typename T, std::size_t Rank>
T* Tensor<T, Rank>::data_ptr() {
    return data_.data();
}

template <typename T, std::size_t Rank>
const T* Tensor<T, Rank>::data_ptr() const {
    return data_.data();
}

template <typename T, std::size_t Rank>
const T& Tensor<T, Rank>::item() const {
    if constexpr (Rank != 0) {
        throw std::logic_error("item() is only valid for rank 0 tensors");
    }
    return data_.data()[0]; 
}

template <typename T, std::size_t Rank>
void Tensor<T, Rank>::save(const std::filesystem::path& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file");
    }
    // write header
    file.write(
        reinterpret_cast<const char*>(&header_),
        sizeof(Header)
    );
    // write shape
    file.write(
        reinterpret_cast<const char*>(shape_.data()),
        Rank * sizeof(size_t)
    );
    // write data
    file.write(
        reinterpret_cast<const char*>(data_.data()),
        size_ * sizeof(T)
    );

    if (!file) {
        throw std::runtime_error("Failed to write tensor to file");
    }

}

template <typename T, std::size_t Rank>
Tensor<T, Rank> Tensor<T, Rank>::load(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Failed to open file");
    }

    Header header;
    file.read(
        reinterpret_cast<char*>(&header),
        sizeof(Header)
    );

    if (!file) {
        throw std::runtime_error("Failed to read tensor header from file");
    }

    if (header.magic != 0x54454E53) {
        throw std::runtime_error("Invalid tensor file format");
    }

    if (header.rank != Rank) {
        throw std::runtime_error("Tensor rank mismatch");
    }

    std::array<size_t, Rank> shape;
    file.read(
        reinterpret_cast<char*>(shape.data()),
        Rank * sizeof(size_t)
    );

    if (!file) {
        throw std::runtime_error("Failed to read tensor shape from file");
    }

    Tensor<T, Rank> tensor(shape);

    file.read(
        reinterpret_cast<char*>(tensor.data_.data()),
        tensor.size_ * sizeof(T)
    );

    if (!file) {
        throw std::runtime_error("Failed to read tensor data from file");
    }

    return tensor;
}
