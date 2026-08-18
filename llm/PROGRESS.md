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

# ------------------------------------------------------------------------
Milestone 8: CPU LayerNorm forward pass

1 - Implement layer normalization over the final tensor dimension : `SUCCESS`
2 - Use a fixed small epsilon, such as 1e-4f, and document it : `SUCCESS`

- Concepts learned: layer_norm
- Files changed: ./include/layerNorm.hpp, ./include/layerNorm.tpp, tests/test_layernorm.cpp, tests/CMakeLists.txt
- Tests added: TestLayerNorm
- Known limitations: NONE
- Deferred improvements: NONE
- Next milestone:

-> STATUS : `PASSED`

# ------------------------------------------------------------------------
Milestone 9: CPU linear layer forward pass

1 - Implement a fully connected layer that transforms the final dimension of an input tensor : `SUCCESS`
2 - Start with this simple version. Do not optimize or parallelize it yet : `SUCCESS`

- Concepts learned: Linear forward pass
- Files changed: ./include/linear.hpp, ./include/linear.tpp, tests/test_linear.cpp, tests/CMakeLists.txt
- Tests added: Linear
- Known limitations: NONE
- Deferred improvements: NONE
- Next milestone: GELU forward pass

-> STATUS : `PASSED`

# ------------------------------------------------------------------------
Milestone 10: GELU forward pass

1 - Implement the Gaussian Error Linear Unit elementwise over a tensor : `SUCCESS`
2 - Test zero, positive, negative, symmetry-related values, approximate reference values, and zero-sized tensors : `SUCCESS`

- Concepts learned: gelu function
- Files changed: ./include/gelu.hpp, ./include/gelu.tpp, tests/test_gelu.cpp, tests/CMakeLists.txt
- Tests added: Gelu
- Known limitations: NONE
- Deferred improvements: NONE
- Next milestone:stable softmax over the final dimension

-> STATUS : `PASSED`

# ------------------------------------------------------------------------
Milestone 11: stable softmax over the final dimension

1 - Implement CPU softmax for a rank-n tensor [B,T,C], normalizing independently over each [B,T] row: `SUCCESS`

- Concepts learned: softmax function
- Files changed: ./include/softmax.hpp, ./include/softmax.tpp, tests/test_softmax.cpp, tests/CMakeLists.txt
- Tests added: Softmax
- Known limitations: NONE
- Deferred improvements: NONE
- Next milestone:

-> STATUS : `PASSED`

# ------------------------------------------------------------------------
Milestone 12: Causal masking

1 - Implement a CPU causal mask for attention scores shaped [B,T,T] : `SUCCESS`

- Concepts learned: causal masking
- Files changed: ./include/causal_masking.hpp, ./include/causal_masking.tpp, tests/test_causal_masking.cpp, tests/CMakeLists.txt
- Tests added: CausalMasking
- Known limitations: NONE
- Deferred improvements: NONE
- Next milestone:

-> STATUS : `PASSED`

# ------------------------------------------------------------------------
Milestone 13: Scaled dot-product attention scores

1 - Implement only the score calculation : `SUCCESS`

- Concepts learned: scaled dot-product attention scores
- Files changed: ./include/scaled_dot_product_attention.hpp, ./include/scaled_dot_product_attention.tpp, tests/test_scaled_dot_product_attention.cpp, tests/CMakeLists.txt
- Tests added: ScaledDotProductAttention
- Known limitations: NONE
- Deferred improvements: NONE
- Next milestone: attention probability matrix

-> STATUS : `PASSED`

# ------------------------------------------------------------------------
Milestone 14: attention probability matrix

1 - Implement only the score calculation : `SUCCESS`

- Concepts learned: scaled dot-product attention scores
- Files changed: ./include/causal_masking.hpp, ./include/causal_masking.tpp, tests/test_causal_masking.cpp, tests/CMakeLists.txt
- Tests added: 
- Known limitations: NONE
- Deferred improvements: NONE
- Next milestone:

-> STATUS :