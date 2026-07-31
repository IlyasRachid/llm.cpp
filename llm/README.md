This project bundles a end-to-end implementation of GPT2 from scratch using C++ and cuda. It's is mainly inspired from Andrej Karpathy's llm.c repo.

In order to make this project conform and scalable, we have to define the grounding rules and conventions that carry how operations are performed.

The standard tensor representation is that used in llm.c repo: (B,T,C) where B is the batch of input data, T is the sequence length (number of tokens per batch input data point), and C is the size of the token (Channel)

the tensor in question will have the properties:
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

