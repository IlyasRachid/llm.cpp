This project bundles a end-to-end implementation of GPT2 from scratch using C++ and cuda. It's is mainly inspired from Andrej Karpathy's llm.c repo.

In order to make this project conform and scalable, we have to define the grounding rules and conventions that carry how operations are performed.

The standard tensor representation is: Tensor<T, Rank> where T is the data type and Rank is the number of dimensions. The tensor in question will have the properties:
- Zero based indexing
- row-major contiguous


# the commands are executed from the source dir "llm" 
to build the project, follow these steps:
``` cmake -S . -B ./build ```
``` cmake --build ./build ```

to build and run the tests, run:
``` ctest --test-dir ./build --output-on-failure ```

to run the executable, use the command:
``` ./build/llm ```

