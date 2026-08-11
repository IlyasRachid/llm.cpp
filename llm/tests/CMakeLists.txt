set(TEST_WARNING_FLAGS -Wall -Wextra -Wpedantic)

add_executable(testIndex test_indexing.cpp)
add_executable(testTensor test_tensor.cpp)
add_executable(testTokenEmbeddings test_token_embeddings.cpp)
add_executable(testPositionalEmbedding test_positional_embeddings.cpp)

target_compile_options(testIndex PRIVATE ${TEST_WARNING_FLAGS})

target_link_libraries(testTensor PRIVATE tensor_lib)
target_compile_options(testTensor PRIVATE ${TEST_WARNING_FLAGS})

target_link_libraries(testTokenEmbeddings PRIVATE tensor_lib)
target_compile_options(testTokenEmbeddings PRIVATE ${TEST_WARNING_FLAGS})

target_link_libraries(testPositionalEmbedding PRIVATE tensor_lib)
target_compile_options(testPositionalEmbedding PRIVATE ${TEST_WARNING_FLAGS})

add_test(NAME AppRunsSuccessfully COMMAND llm)
add_test(NAME TensorConventions COMMAND testIndex)
add_test(NAME Tensor COMMAND testTensor)
add_test(NAME TokenEmbeddings COMMAND testTokenEmbeddings)
add_test(NAME PositionEmbedding COMMAND testPositionalEmbedding)