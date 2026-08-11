This file records the progress made throughout this project

# ------------------------------------------------------------------------
Milestone 1: establish a reproducible C++ project skeleton

1 - Configure out-of-source: cmake -S llm -B llm/build : `SUCCESS`
2 - Build: cmake --build llm/build : `SUCCESS`
3 - Run CTest: ctest --test-dir llm/build --output-on-failure : `SUCCESS`
4 - Run the executable directly : `SUCCESS`
5 - Repeat from a newly deleted llm/build directory to demonstrate no hidden generated inputs : `SUCCESS`

- Concepts learned: Writing CMakeLists file, building cpp projects
- Files changed: ./src/main.cpp, ./CMakeLists.txt, ./README.md, ./tests/CMakeLists.txt, ./.gitignore
- Tests added: AppRunsSuccessfully
- Known limitations: NONE
- Deferred improvements: NONE
- Next milestone: formalize tensor shape and indexing conventions

-> STATUS : `PASSED`


# ------------------------------------------------------------------------
Milestone 2: formalize tensor shape and indexing conventions

1 - Add docs/tensor_conventions.md, covering rank-1 [C], rank-2 [B, C], and rank-3 [B, T, C] tensors : `SUCCESS`
2 - Write a test file for checking the correctness of the flatened index: `SUCCESS`

- Concepts learned: inline indexing
- Files changed: docs/tensor_conventions.md, tests/test_indexing.cpp, ./tests/CMakeLists.txt
- Tests added: TensorConventions
- Known limitations: NONE
- Deferred improvements: NONE
- Next milestone: CPU rank-3 tensor storage and access

-> STATUS : `PASSED`


# ------------------------------------------------------------------------
Milestone 3: CPU rank-3 tensor storage and access

1 - Implement an owning CPU tensor for float data shaped [B, T, C] : `SUCCESS`
2 - Implement a small Tensor3D type with shape accessors, size(), a contiguous std::vector<float>, and const/non-const element access: `SUCCESS`
3 - Check shape/size, write/read every element for multiple shapes, validate flat-buffer row-major order, test singleton and zero-sized shapes, and test your chosen invalid-index behavior : `SUCCESS`
4 - Document ownership/indexing decisions, add the test to CTest, and show a clean build plus passing tests : `SUCCESS`

- Concepts learned: NONE
- Files changed: ./include/tensor.hpp, ./src/tensor.cpp, tests/test_tensor3D.cpp, ./tests/CMakeLists.txt, ./CMakeLists.txt
- Tests added: Tensor
- Known limitations: The tensor has a fixed shape of 3
- Deferred improvements: create a generic tensor TensorND.
- Next milestone:

-> STATUS : `PASSED`


# ------------------------------------------------------------------------
Milestone 4: CPU elementwise addition for [B,T,C]

1 - Add two same-shaped Tensor3D tensors elementwise : `SUCCESS`
2 - Apply the operation independently at every flat offset i: out[i] = a[i] + b[i] : `SUCCESS`
3 - preserve input tensors; define and document behavior if any tensor has a zero-sized dimension. Reject shape mismatches predictably : `SUCCESS`
4 - Implement: A narrowly scoped addition function, with a documented signature and exception policy : `SUCCESS`
5 - Register the test with CTest and provide the clean configure/build/CTest result : `SUCCESS`

- Concepts learned: dangling references, move semantics
- Files changed: ./include/tensor.hpp, ./src/tensor.cpp, tests/test_tensor3D.cpp
- Tests added: Tensor
- Known limitations: the two operands tensors must have the same shape
- Deferred improvements: Add broadcasting and parallelism with CUDA 
- Next milestone: Generic tensor

-> STATUS : `PASSED`

# ------------------------------------------------------------------------
Milestone 5: Generic tensor

1 - Refactor Tensor3D into Tensor<T, Rank> with contiguous allocation, checked shape product, row-major strides, and checked coordinate access : `SUCCESS`
2 - Preserve/regression-test all existing Tensor3D behavior : `SUCCESS`

- Concepts learned: mixed-radix conversion, template files
- Files changed: ./include/tensor.hpp, ./include/tensor.tpp, tests/test_tensor.cpp, ./CMakeLists.txt, tests/CMakeLists.txt
- Tests added: Tensor
- Known limitations: NONE
- Deferred improvements: NONE
- Next milestone: CPU token-embedding forward pass

-> STATUS : `PASSED`

# ------------------------------------------------------------------------
Milestone 6: CPU token-embedding forward pass

1 - Implement a CPU function that maps token IDs to embedding vector : `SUCCESS`
2 - Add tests using a tiny hand-filled table where each output value is random float: `SUCCESS`

- Concepts learned: token embedding, embedding lookup
- Files changed: ./include/tensor.hpp, ./include/tensor.tpp, tests/test_token_embeddings.cpp, tests/CMakeLists.txt
- Tests added: TokenEmbeddings
- Known limitations: Embedding lookup is implemented through generic tensor indexing, not a dedicated embedding API
- Deferred improvements: Having a Tensor method for embedding lookup, and a more efficient implementation using a single contiguous buffer for the embedding table 
- Next milestone:

-> STATUS : `PASSED`

# ------------------------------------------------------------------------
Milestone 7: positional embedding forward pass

1 - Implement CPU positional embeddings for GPT-style input representations : `SUCCESS`
2 - Add tests using a tiny hand-filled table where each output value is random float: `SUCCESS`

- Concepts learned: Positional embedding
- Files changed: ./include/embeddings.hpp, ./include/embeddings.tpp, tests/test_positional_embeddings.cpp, tests/CMakeLists.txt
- Tests added: PositionEmbedding
- Known limitations: NONE
- Deferred improvements: NONE
- Next milestone:

-> STATUS : `PASSED`