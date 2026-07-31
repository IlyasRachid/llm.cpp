#include <cassert>
#include <iostream>
#include <vector>

template <typename T>
int test_indexing(const std::vector<T>& data, std::size_t index_C, std::size_t flattened_index) {
    assert(flattened_index < data.size());
    assert(data[index_C] == data[flattened_index]);
    return 0;
}

template <typename T>
int test_indexing(const std::vector<std::vector<T>>& data, const std::vector<T>& flattened_data, std::size_t index_B, std::size_t index_C, std::size_t flattened_index) {
    assert(flattened_index < flattened_data.size());
    assert(data[index_B][index_C] == flattened_data[flattened_index]);
    return 0;
}

template <typename T>
int test_indexing(const std::vector<std::vector<std::vector<T>>>& data, const std::vector<T>& flattened_data, std::size_t index_B, std::size_t index_T, std::size_t index_C, std::size_t flattened_index) {
    assert(flattened_index < flattened_data.size());
    assert(data[index_B][index_T][index_C] == flattened_data[flattened_index]);
    return 0;
}



void test(int dim0, int dim1, int dim2) {

    std::vector<int> data1D(dim0);

    std::vector<std::vector<int>> data2D(
        dim0,
        std::vector<int>(dim1)
    );

    std::vector<std::vector<std::vector<int>>> data3D(
        dim0,
        std::vector<std::vector<int>>(
            dim1,
            std::vector<int>(dim2)
        )
    );

    std::vector<int> flat1D(dim0);
    std::vector<int> flat2D(dim0 * dim1);
    std::vector<int> flat3D(dim0 * dim1 * dim2);

    // Fill both structures with the same values
    for (int i = 0; i < dim0; ++i) {

        int value1D = i;
            data1D[i] = value1D;

            int flat_index_1D = i;
            flat1D[flat_index_1D] = value1D;

        for (int j = 0; j < dim1; ++j) {

            int value2D = 10 * i + j;
            data2D[i][j] = value2D;

            int flat_index_2D = i*dim1 + j;
            flat2D[flat_index_2D] = value2D;

            for (int k = 0; k < dim2; ++k) {
                int value3D = 100 * i + 10 * j + k;

                data3D[i][j][k] = value3D;

                int flat_index_3D = i * dim1 * dim2
                               + j * dim2
                               + k;

                flat3D[flat_index_3D] = value3D;
            }
        }
    }

    

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            for (int k = 0; k < dim2; ++k) {
                int flat_index_1D = i;
                int flat_index_2D = i*dim1 + j;
                int flat_index_3D = i * dim1 * dim2
                            + j * dim2
                            + k;
                test_indexing(data1D, i, flat_index_1D);
                test_indexing(data2D, flat2D, i, j, flat_index_2D);
                test_indexing(data3D, flat3D, i, j, k, flat_index_3D);
            }
        }
    }
}

int main() {
    for (int dim0 = 0; dim0 < 4; dim0++) {
        for (int dim1 = 0; dim1 < 4; dim1++) {
            for (int dim2 = 0; dim2 < 4; dim2++) {
                test(dim0, dim1, dim2);
            }
        }
    }
    return 0;
}