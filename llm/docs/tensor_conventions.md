# This is a compact design note defining the tensor conventions we will enforce before implementing any tensor storage or operations.

Required definitions:

- All tensors are contiguous, row-major, zero-based.
- Valid indexes:
batch diemnsion B: 0 <= x < B (B >= 0)
Time_Sequence dimension T: 0 <= x < T (T >= 0)
Channel C: 0 <= x < C (C >= 0)
- Zero-sized tensors are permmitted.
- The final dimension is contiguous and has stride 1.

1) - Rank-1 tensors (C):
offset(c) = c;

2) - Rank-2 tensors (B,C):
offset(b, c) = b*C + c;
``` Example: ```
``` L = [[1,2,3], [4,5,6], [7,8,9]]; offset(1, 2) = 1 * 3 + 2```

3) - Rank-3 tensors (B,T,C):
offset(b, t, c) = b * (T*C) + (t*C) + c = (b*T + t)*C + c;
