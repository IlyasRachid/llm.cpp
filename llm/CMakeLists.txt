cmake_minimum_required(VERSION 3.16)

project(llm VERSION 1.0 DESCRIPTION "GPT2 from scratch" LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Tensor is implemented by templates in the public headers.
add_library(tensor_lib INTERFACE)


# Add header search paths
target_include_directories(tensor_lib INTERFACE ${CMAKE_SOURCE_DIR}/include)

target_compile_options(tensor_lib INTERFACE -Wall -Wextra -Wpedantic)


# Create an executable
add_executable(llm src/main.cpp)


target_link_libraries(llm PRIVATE tensor_lib)


# Compile warnings
target_compile_options(llm PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic> $<$<CXX_COMPILER_ID:MSVC>:/W4>)


enable_testing()
add_subdirectory(tests)
