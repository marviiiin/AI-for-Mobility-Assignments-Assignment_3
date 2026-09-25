"""
Question 2.1: Convolution by Hand
4x4 input matrix with a 2x2 kernel (filter)
Stride = 1, No padding
Output size: (4-2+1) x (4-2+1) = 3x3
"""

import numpy as np

print("=" * 70)
print("QUESTION 2.1: CONVOLUTION BY HAND")
print("=" * 70)

# Define input matrix (4x4)
X = np.array([
    [1, 2, 3, 0],
    [0, 1, 2, 1],
    [1, 0, 1, 2],
    [2, 1, 0, 1]
])

# Define kernel/filter (2x2)
K = np.array([
    [1, 0],
    [-1, 1]
])

print("\nInput Matrix (4x4):")
print(f"  X = | {X[0,0]}  {X[0,1]}  {X[0,2]}  {X[0,3]} |")
print(f"      | {X[1,0]}  {X[1,1]}  {X[1,2]}  {X[1,3]} |")
print(f"      | {X[2,0]}  {X[2,1]}  {X[2,2]}  {X[2,3]} |")
print(f"      | {X[3,0]}  {X[3,1]}  {X[3,2]}  {X[3,3]} |")

print(f"\nKernel/Filter (2x2):")
print(f"  K = | {K[0,0]}   {K[0,1]} |")
print(f"      | {K[1,0]}   {K[1,1]} |")

print(f"\nStride = 1, No padding")
print(f"Output size = (4 - 2 + 1) x (4 - 2 + 1) = 3 x 3")

print("\n" + "=" * 70)
print("STEP-BY-STEP CONVOLUTION (element-wise multiply and sum)")
print("=" * 70)

output = np.zeros((3, 3))

for i in range(3):
    for j in range(3):
        # Extract the 2x2 patch
        patch = X[i:i+2, j:j+2]

        # Element-wise multiply and sum
        products = patch * K
        result = np.sum(products)
        output[i, j] = result

        print(f"\n--- Position ({i},{j}): rows [{i},{i+1}], cols [{j},{j+1}] ---")
        print(f"  Patch from X:     | {patch[0,0]}  {patch[0,1]} |")
        print(f"                    | {patch[1,0]}  {patch[1,1]} |")
        print(f"  Element-wise multiply with kernel:")
        print(f"    {patch[0,0]}*({K[0,0]}) + {patch[0,1]}*({K[0,1]}) + {patch[1,0]}*({K[1,0]}) + {patch[1,1]}*({K[1,1]})")
        print(f"    = {patch[0,0]*K[0,0]} + {patch[0,1]*K[0,1]} + ({patch[1,0]*K[1,0]}) + {patch[1,1]*K[1,1]}")
        print(f"    = {int(result)}")

print("\n" + "=" * 70)
print("OUTPUT FEATURE MAP (3x3)")
print("=" * 70)
print(f"\n  Output = | {int(output[0,0]):2d}  {int(output[0,1]):2d}  {int(output[0,2]):2d} |")
print(f"           | {int(output[1,0]):2d}  {int(output[1,1]):2d}  {int(output[1,2]):2d} |")
print(f"           | {int(output[2,0]):2d}  {int(output[2,1]):2d}  {int(output[2,2]):2d} |")

# Verify with scipy
print("\n" + "=" * 70)
print("VERIFICATION WITH SCIPY")
print("=" * 70)
from scipy.signal import correlate2d
output_verify = correlate2d(X, K, mode='valid')
print(f"\nscipy correlate2d result:")
print(output_verify)
print(f"\nHand calculation matches: {np.allclose(output, output_verify)}")

print("\n" + "=" * 70)
print("VISUAL SUMMARY FOR PAPER")
print("=" * 70)
print("""
Write on paper:

1. Draw the 4x4 input matrix X
2. Draw the 2x2 kernel K
3. Show the sliding window at each of the 9 positions
4. For each position, write:
   - The 2x2 patch extracted from X
   - The element-wise products with K
   - The sum = output value
5. Write the final 3x3 output feature map

Key formula: Output(i,j) = SUM over (m,n) of X(i+m, j+n) * K(m,n)
where m,n in {0,1} for a 2x2 kernel
""")
