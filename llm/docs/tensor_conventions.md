# This is a compact design note defining the tensor conventions we will enforce before implementing any tensor storage or operations.

Required definitions:

- All tensors are contiguous, row-major, zero-based.
- The standard tensor representation is: Tensor<T, Rank> where T is the data type and Rank is the number of dimensions. 
- The tensor will have a shape defined by an array of size_t of length Rank.
- The tensor will have a data pointer of type T* that points to the contiguous memory block storing the tensor data.
- The tensor will have a numel() method that returns the total number of elements in the tensor, which is the product of the dimensions in the shape array.
- The tensor will have an item() method that returns a reference to the single element in a rank 0 tensor. If the tensor is not rank 0, calling item() will throw a logic_error exception.


Tensor saving format: (binary + metadata)
+---------------------------+
| Magic Number (4 bytes)    |
+---------------------------+
| Version (4 bytes)         |
+---------------------------+
| Rank (4 bytes)            |
+---------------------------+
| Element Size (4 bytes)    |
+---------------------------+
| Shape (rank * uint64_t)   |
+---------------------------+
| Raw tensor data           |
+---------------------------+


final layout:

+--------------------------+
| Magic = "TENS"           |
+--------------------------+
| Version                  |
+--------------------------+
| DType                    |
+--------------------------+
| Rank                     |
+--------------------------+
| Reserved                 |
+--------------------------+
| Shape[0]                 |
+--------------------------+
| Shape[1]                 |
+--------------------------+
| ...                      |
+--------------------------+
| Shape[Rank-1]            |
+--------------------------+
| Raw tensor bytes         |
+--------------------------+