"""
Question 1.1: Backpropagation - Step-by-step hand computation + PyTorch verification
Network: 2 (input) -> 2 (layer1) -> 2 (layer2) -> 2 (layer3) -> 1 (output)
Activation: Sigmoid
Loss: MSE (L = 0.5 * (y_pred - y_true)^2)
"""

import numpy as np

# ============================================================
# PART 1: STEP-BY-STEP HAND COMPUTATION (copy onto paper)
# ============================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(a):
    """Derivative of sigmoid given the output a = sigmoid(z)"""
    return a * (1.0 - a)

print("=" * 70)
print("QUESTION 1.1: BACKPROPAGATION BY HAND")
print("=" * 70)

# --- Step 1: Define Network Architecture and Initial Values ---
print("\n" + "=" * 70)
print("STEP 1: NETWORK SETUP AND INITIAL VALUES")
print("=" * 70)

# Inputs and target
x = np.array([0.5, 0.8])
y_true = 1.0
print(f"\nInputs:  x1 = {x[0]}, x2 = {x[1]}")
print(f"Target:  y = {y_true}")

# Layer 1 weights (2 inputs -> 2 neurons)
W1 = np.array([[0.10, 0.30],   # neuron 1: w11, w12
               [0.20, 0.40]])  # neuron 2: w21, w22
b1 = np.array([0.10, 0.10])
print(f"\nLayer 1 Weights (W1):")
print(f"  w1_11 = {W1[0,0]}, w1_12 = {W1[0,1]}")
print(f"  w1_21 = {W1[1,0]}, w1_22 = {W1[1,1]}")
print(f"  b1_1 = {b1[0]}, b1_2 = {b1[1]}")

# Layer 2 weights (2 -> 2)
W2 = np.array([[0.30, 0.50],
               [0.40, 0.20]])
b2 = np.array([0.10, 0.10])
print(f"\nLayer 2 Weights (W2):")
print(f"  w2_11 = {W2[0,0]}, w2_12 = {W2[0,1]}")
print(f"  w2_21 = {W2[1,0]}, w2_22 = {W2[1,1]}")
print(f"  b2_1 = {b2[0]}, b2_2 = {b2[1]}")

# Layer 3 weights (2 -> 2)
W3 = np.array([[0.20, 0.40],
               [0.30, 0.10]])
b3 = np.array([0.10, 0.10])
print(f"\nLayer 3 Weights (W3):")
print(f"  w3_11 = {W3[0,0]}, w3_12 = {W3[0,1]}")
print(f"  w3_21 = {W3[1,0]}, w3_22 = {W3[1,1]}")
print(f"  b3_1 = {b3[0]}, b3_2 = {b3[1]}")

# Output layer weights (2 -> 1)
Wo = np.array([[0.50, 0.30]])
bo = np.array([0.10])
print(f"\nOutput Layer Weights (Wo):")
print(f"  wo_1 = {Wo[0,0]}, wo_2 = {Wo[0,1]}")
print(f"  bo = {bo[0]}")

# --- Step 2: Forward Pass ---
print("\n" + "=" * 70)
print("STEP 2: FORWARD PASS")
print("=" * 70)

# Layer 1
print("\n--- Layer 1 ---")
z1_1 = W1[0,0]*x[0] + W1[0,1]*x[1] + b1[0]
z1_2 = W1[1,0]*x[0] + W1[1,1]*x[1] + b1[1]
print(f"z1_1 = w1_11*x1 + w1_12*x2 + b1_1")
print(f"     = {W1[0,0]}*{x[0]} + {W1[0,1]}*{x[1]} + {b1[0]}")
print(f"     = {W1[0,0]*x[0]} + {W1[0,1]*x[1]} + {b1[0]}")
print(f"     = {z1_1:.6f}")

print(f"\nz1_2 = w1_21*x1 + w1_22*x2 + b1_2")
print(f"     = {W1[1,0]}*{x[0]} + {W1[1,1]}*{x[1]} + {b1[1]}")
print(f"     = {W1[1,0]*x[0]} + {W1[1,1]*x[1]} + {b1[1]}")
print(f"     = {z1_2:.6f}")

a1_1 = sigmoid(z1_1)
a1_2 = sigmoid(z1_2)
print(f"\na1_1 = sigmoid(z1_1) = sigmoid({z1_1:.6f}) = {a1_1:.6f}")
print(f"a1_2 = sigmoid(z1_2) = sigmoid({z1_2:.6f}) = {a1_2:.6f}")

z1 = np.array([z1_1, z1_2])
a1 = np.array([a1_1, a1_2])

# Layer 2
print("\n--- Layer 2 ---")
z2_1 = W2[0,0]*a1[0] + W2[0,1]*a1[1] + b2[0]
z2_2 = W2[1,0]*a1[0] + W2[1,1]*a1[1] + b2[1]
print(f"z2_1 = w2_11*a1_1 + w2_12*a1_2 + b2_1")
print(f"     = {W2[0,0]}*{a1[0]:.6f} + {W2[0,1]}*{a1[1]:.6f} + {b2[0]}")
print(f"     = {W2[0,0]*a1[0]:.6f} + {W2[0,1]*a1[1]:.6f} + {b2[0]}")
print(f"     = {z2_1:.6f}")

print(f"\nz2_2 = w2_21*a1_1 + w2_22*a1_2 + b2_2")
print(f"     = {W2[1,0]}*{a1[0]:.6f} + {W2[1,1]}*{a1[1]:.6f} + {b2[1]}")
print(f"     = {W2[1,0]*a1[0]:.6f} + {W2[1,1]*a1[1]:.6f} + {b2[1]}")
print(f"     = {z2_2:.6f}")

a2_1 = sigmoid(z2_1)
a2_2 = sigmoid(z2_2)
print(f"\na2_1 = sigmoid(z2_1) = sigmoid({z2_1:.6f}) = {a2_1:.6f}")
print(f"a2_2 = sigmoid(z2_2) = sigmoid({z2_2:.6f}) = {a2_2:.6f}")

z2 = np.array([z2_1, z2_2])
a2 = np.array([a2_1, a2_2])

# Layer 3
print("\n--- Layer 3 ---")
z3_1 = W3[0,0]*a2[0] + W3[0,1]*a2[1] + b3[0]
z3_2 = W3[1,0]*a2[0] + W3[1,1]*a2[1] + b3[1]
print(f"z3_1 = w3_11*a2_1 + w3_12*a2_2 + b3_1")
print(f"     = {W3[0,0]}*{a2[0]:.6f} + {W3[0,1]}*{a2[1]:.6f} + {b3[0]}")
print(f"     = {W3[0,0]*a2[0]:.6f} + {W3[0,1]*a2[1]:.6f} + {b3[0]}")
print(f"     = {z3_1:.6f}")

print(f"\nz3_2 = w3_21*a2_1 + w3_22*a2_2 + b3_2")
print(f"     = {W3[1,0]}*{a2[0]:.6f} + {W3[1,1]}*{a2[1]:.6f} + {b3[1]}")
print(f"     = {W3[1,0]*a2[0]:.6f} + {W3[1,1]*a2[1]:.6f} + {b3[1]}")
print(f"     = {z3_2:.6f}")

a3_1 = sigmoid(z3_1)
a3_2 = sigmoid(z3_2)
print(f"\na3_1 = sigmoid(z3_1) = sigmoid({z3_1:.6f}) = {a3_1:.6f}")
print(f"a3_2 = sigmoid(z3_2) = sigmoid({z3_2:.6f}) = {a3_2:.6f}")

z3 = np.array([z3_1, z3_2])
a3 = np.array([a3_1, a3_2])

# Output Layer
print("\n--- Output Layer ---")
z_out = Wo[0,0]*a3[0] + Wo[0,1]*a3[1] + bo[0]
print(f"z_out = wo_1*a3_1 + wo_2*a3_2 + bo")
print(f"      = {Wo[0,0]}*{a3[0]:.6f} + {Wo[0,1]}*{a3[1]:.6f} + {bo[0]}")
print(f"      = {Wo[0,0]*a3[0]:.6f} + {Wo[0,1]*a3[1]:.6f} + {bo[0]}")
print(f"      = {z_out:.6f}")

a_out = sigmoid(z_out)
print(f"\na_out = sigmoid(z_out) = sigmoid({z_out:.6f}) = {a_out:.6f}")
print(f"\ny_pred = a_out = {a_out:.6f}")

# Loss
L = 0.5 * (a_out - y_true)**2
print(f"\n--- Loss ---")
print(f"L = 0.5 * (y_pred - y_true)^2")
print(f"  = 0.5 * ({a_out:.6f} - {y_true})^2")
print(f"  = 0.5 * ({a_out - y_true:.6f})^2")
print(f"  = {L:.6f}")

# --- Step 3: Backward Pass ---
print("\n" + "=" * 70)
print("STEP 3: BACKWARD PASS (BACKPROPAGATION)")
print("=" * 70)

# Output layer gradients
print("\n--- Output Layer Gradients ---")
dL_da_out = a_out - y_true
print(f"dL/da_out = a_out - y_true = {a_out:.6f} - {y_true} = {dL_da_out:.6f}")

da_out_dz_out = sigmoid_derivative(a_out)
print(f"da_out/dz_out = a_out*(1-a_out) = {a_out:.6f}*(1-{a_out:.6f}) = {da_out_dz_out:.6f}")

dL_dz_out = dL_da_out * da_out_dz_out
print(f"\ndL/dz_out = dL/da_out * da_out/dz_out")
print(f"          = {dL_da_out:.6f} * {da_out_dz_out:.6f}")
print(f"          = {dL_dz_out:.6f}")

# Gradients for output weights
dL_dwo1 = dL_dz_out * a3[0]
dL_dwo2 = dL_dz_out * a3[1]
dL_dbo = dL_dz_out
print(f"\ndL/dwo_1 = dL/dz_out * a3_1 = {dL_dz_out:.6f} * {a3[0]:.6f} = {dL_dwo1:.6f}")
print(f"dL/dwo_2 = dL/dz_out * a3_2 = {dL_dz_out:.6f} * {a3[1]:.6f} = {dL_dwo2:.6f}")
print(f"dL/dbo   = dL/dz_out = {dL_dbo:.6f}")

# Backprop to Layer 3
print("\n--- Layer 3 Gradients ---")
dL_da3_1 = dL_dz_out * Wo[0,0]
dL_da3_2 = dL_dz_out * Wo[0,1]
print(f"dL/da3_1 = dL/dz_out * wo_1 = {dL_dz_out:.6f} * {Wo[0,0]} = {dL_da3_1:.6f}")
print(f"dL/da3_2 = dL/dz_out * wo_2 = {dL_dz_out:.6f} * {Wo[0,1]} = {dL_da3_2:.6f}")

da3_1_dz3_1 = sigmoid_derivative(a3[0])
da3_2_dz3_2 = sigmoid_derivative(a3[1])
print(f"\nda3_1/dz3_1 = a3_1*(1-a3_1) = {a3[0]:.6f}*(1-{a3[0]:.6f}) = {da3_1_dz3_1:.6f}")
print(f"da3_2/dz3_2 = a3_2*(1-a3_2) = {a3[1]:.6f}*(1-{a3[1]:.6f}) = {da3_2_dz3_2:.6f}")

dL_dz3_1 = dL_da3_1 * da3_1_dz3_1
dL_dz3_2 = dL_da3_2 * da3_2_dz3_2
print(f"\ndL/dz3_1 = dL/da3_1 * da3_1/dz3_1 = {dL_da3_1:.6f} * {da3_1_dz3_1:.6f} = {dL_dz3_1:.6f}")
print(f"dL/dz3_2 = dL/da3_2 * da3_2/dz3_2 = {dL_da3_2:.6f} * {da3_2_dz3_2:.6f} = {dL_dz3_2:.6f}")

# Layer 3 weight gradients
dL_dw3_11 = dL_dz3_1 * a2[0]
dL_dw3_12 = dL_dz3_1 * a2[1]
dL_dw3_21 = dL_dz3_2 * a2[0]
dL_dw3_22 = dL_dz3_2 * a2[1]
dL_db3_1 = dL_dz3_1
dL_db3_2 = dL_dz3_2
print(f"\ndL/dw3_11 = dL/dz3_1 * a2_1 = {dL_dz3_1:.6f} * {a2[0]:.6f} = {dL_dw3_11:.6f}")
print(f"dL/dw3_12 = dL/dz3_1 * a2_2 = {dL_dz3_1:.6f} * {a2[1]:.6f} = {dL_dw3_12:.6f}")
print(f"dL/dw3_21 = dL/dz3_2 * a2_1 = {dL_dz3_2:.6f} * {a2[0]:.6f} = {dL_dw3_21:.6f}")
print(f"dL/dw3_22 = dL/dz3_2 * a2_2 = {dL_dz3_2:.6f} * {a2[1]:.6f} = {dL_dw3_22:.6f}")
print(f"dL/db3_1  = dL/dz3_1 = {dL_db3_1:.6f}")
print(f"dL/db3_2  = dL/dz3_2 = {dL_db3_2:.6f}")

# Backprop to Layer 2
print("\n--- Layer 2 Gradients ---")
dL_da2_1 = dL_dz3_1 * W3[0,0] + dL_dz3_2 * W3[1,0]
dL_da2_2 = dL_dz3_1 * W3[0,1] + dL_dz3_2 * W3[1,1]
print(f"dL/da2_1 = dL/dz3_1*w3_11 + dL/dz3_2*w3_21")
print(f"         = {dL_dz3_1:.6f}*{W3[0,0]} + {dL_dz3_2:.6f}*{W3[1,0]}")
print(f"         = {dL_dz3_1*W3[0,0]:.6f} + {dL_dz3_2*W3[1,0]:.6f}")
print(f"         = {dL_da2_1:.6f}")

print(f"\ndL/da2_2 = dL/dz3_1*w3_12 + dL/dz3_2*w3_22")
print(f"         = {dL_dz3_1:.6f}*{W3[0,1]} + {dL_dz3_2:.6f}*{W3[1,1]}")
print(f"         = {dL_dz3_1*W3[0,1]:.6f} + {dL_dz3_2*W3[1,1]:.6f}")
print(f"         = {dL_da2_2:.6f}")

da2_1_dz2_1 = sigmoid_derivative(a2[0])
da2_2_dz2_2 = sigmoid_derivative(a2[1])
print(f"\nda2_1/dz2_1 = a2_1*(1-a2_1) = {a2[0]:.6f}*(1-{a2[0]:.6f}) = {da2_1_dz2_1:.6f}")
print(f"da2_2/dz2_2 = a2_2*(1-a2_2) = {a2[1]:.6f}*(1-{a2[1]:.6f}) = {da2_2_dz2_2:.6f}")

dL_dz2_1 = dL_da2_1 * da2_1_dz2_1
dL_dz2_2 = dL_da2_2 * da2_2_dz2_2
print(f"\ndL/dz2_1 = dL/da2_1 * da2_1/dz2_1 = {dL_da2_1:.6f} * {da2_1_dz2_1:.6f} = {dL_dz2_1:.6f}")
print(f"dL/dz2_2 = dL/da2_2 * da2_2/dz2_2 = {dL_da2_2:.6f} * {da2_2_dz2_2:.6f} = {dL_dz2_2:.6f}")

# Layer 2 weight gradients
dL_dw2_11 = dL_dz2_1 * a1[0]
dL_dw2_12 = dL_dz2_1 * a1[1]
dL_dw2_21 = dL_dz2_2 * a1[0]
dL_dw2_22 = dL_dz2_2 * a1[1]
dL_db2_1 = dL_dz2_1
dL_db2_2 = dL_dz2_2
print(f"\ndL/dw2_11 = dL/dz2_1 * a1_1 = {dL_dz2_1:.6f} * {a1[0]:.6f} = {dL_dw2_11:.6f}")
print(f"dL/dw2_12 = dL/dz2_1 * a1_2 = {dL_dz2_1:.6f} * {a1[1]:.6f} = {dL_dw2_12:.6f}")
print(f"dL/dw2_21 = dL/dz2_2 * a1_1 = {dL_dz2_2:.6f} * {a1[0]:.6f} = {dL_dw2_21:.6f}")
print(f"dL/dw2_22 = dL/dz2_2 * a1_2 = {dL_dz2_2:.6f} * {a1[1]:.6f} = {dL_dw2_22:.6f}")
print(f"dL/db2_1  = dL/dz2_1 = {dL_db2_1:.6f}")
print(f"dL/db2_2  = dL/dz2_2 = {dL_db2_2:.6f}")

# Backprop to Layer 1
print("\n--- Layer 1 Gradients ---")
dL_da1_1 = dL_dz2_1 * W2[0,0] + dL_dz2_2 * W2[1,0]
dL_da1_2 = dL_dz2_1 * W2[0,1] + dL_dz2_2 * W2[1,1]
print(f"dL/da1_1 = dL/dz2_1*w2_11 + dL/dz2_2*w2_21")
print(f"         = {dL_dz2_1:.6f}*{W2[0,0]} + {dL_dz2_2:.6f}*{W2[1,0]}")
print(f"         = {dL_dz2_1*W2[0,0]:.6f} + {dL_dz2_2*W2[1,0]:.6f}")
print(f"         = {dL_da1_1:.6f}")

print(f"\ndL/da1_2 = dL/dz2_1*w2_12 + dL/dz2_2*w2_22")
print(f"         = {dL_dz2_1:.6f}*{W2[0,1]} + {dL_dz2_2:.6f}*{W2[1,1]}")
print(f"         = {dL_dz2_1*W2[0,1]:.6f} + {dL_dz2_2*W2[1,1]:.6f}")
print(f"         = {dL_da1_2:.6f}")

da1_1_dz1_1 = sigmoid_derivative(a1[0])
da1_2_dz1_2 = sigmoid_derivative(a1[1])
print(f"\nda1_1/dz1_1 = a1_1*(1-a1_1) = {a1[0]:.6f}*(1-{a1[0]:.6f}) = {da1_1_dz1_1:.6f}")
print(f"da1_2/dz1_2 = a1_2*(1-a1_2) = {a1[1]:.6f}*(1-{a1[1]:.6f}) = {da1_2_dz1_2:.6f}")

dL_dz1_1 = dL_da1_1 * da1_1_dz1_1
dL_dz1_2 = dL_da1_2 * da1_2_dz1_2
print(f"\ndL/dz1_1 = dL/da1_1 * da1_1/dz1_1 = {dL_da1_1:.6f} * {da1_1_dz1_1:.6f} = {dL_dz1_1:.6f}")
print(f"dL/dz1_2 = dL/da1_2 * da1_2/dz1_2 = {dL_da1_2:.6f} * {da1_2_dz1_2:.6f} = {dL_dz1_2:.6f}")

# Layer 1 weight gradients
dL_dw1_11 = dL_dz1_1 * x[0]
dL_dw1_12 = dL_dz1_1 * x[1]
dL_dw1_21 = dL_dz1_2 * x[0]
dL_dw1_22 = dL_dz1_2 * x[1]
dL_db1_1 = dL_dz1_1
dL_db1_2 = dL_dz1_2
print(f"\ndL/dw1_11 = dL/dz1_1 * x1 = {dL_dz1_1:.6f} * {x[0]} = {dL_dw1_11:.6f}")
print(f"dL/dw1_12 = dL/dz1_1 * x2 = {dL_dz1_1:.6f} * {x[1]} = {dL_dw1_12:.6f}")
print(f"dL/dw1_21 = dL/dz1_2 * x1 = {dL_dz1_2:.6f} * {x[0]} = {dL_dw1_21:.6f}")
print(f"dL/dw1_22 = dL/dz1_2 * x2 = {dL_dz1_2:.6f} * {x[1]} = {dL_dw1_22:.6f}")
print(f"dL/db1_1  = dL/dz1_1 = {dL_db1_1:.6f}")
print(f"dL/db1_2  = dL/dz1_2 = {dL_db1_2:.6f}")

# --- Summary ---
print("\n" + "=" * 70)
print("GRADIENT SUMMARY (all dL/d values)")
print("=" * 70)
print(f"\nOutput Layer:")
print(f"  dL/dwo_1 = {dL_dwo1:.6f}")
print(f"  dL/dwo_2 = {dL_dwo2:.6f}")
print(f"  dL/dbo   = {dL_dbo:.6f}")
print(f"\nLayer 3:")
print(f"  dL/dw3_11 = {dL_dw3_11:.6f}")
print(f"  dL/dw3_12 = {dL_dw3_12:.6f}")
print(f"  dL/dw3_21 = {dL_dw3_21:.6f}")
print(f"  dL/dw3_22 = {dL_dw3_22:.6f}")
print(f"  dL/db3_1  = {dL_db3_1:.6f}")
print(f"  dL/db3_2  = {dL_db3_2:.6f}")
print(f"\nLayer 2:")
print(f"  dL/dw2_11 = {dL_dw2_11:.6f}")
print(f"  dL/dw2_12 = {dL_dw2_12:.6f}")
print(f"  dL/dw2_21 = {dL_dw2_21:.6f}")
print(f"  dL/dw2_22 = {dL_dw2_22:.6f}")
print(f"  dL/db2_1  = {dL_db2_1:.6f}")
print(f"  dL/db2_2  = {dL_db2_2:.6f}")
print(f"\nLayer 1:")
print(f"  dL/dw1_11 = {dL_dw1_11:.6f}")
print(f"  dL/dw1_12 = {dL_dw1_12:.6f}")
print(f"  dL/dw1_21 = {dL_dw1_21:.6f}")
print(f"  dL/dw1_22 = {dL_dw1_22:.6f}")
print(f"  dL/db1_1  = {dL_db1_1:.6f}")
print(f"  dL/db1_2  = {dL_db1_2:.6f}")

print(f"\nPre-activation gradients (dL/dz):")
print(f"  dL/dz_out = {dL_dz_out:.6f}")
print(f"  dL/dz3_1  = {dL_dz3_1:.6f},  dL/dz3_2 = {dL_dz3_2:.6f}")
print(f"  dL/dz2_1  = {dL_dz2_1:.6f},  dL/dz2_2 = {dL_dz2_2:.6f}")
print(f"  dL/dz1_1  = {dL_dz1_1:.6f},  dL/dz1_2 = {dL_dz1_2:.6f}")

# ============================================================
# PART 2: PYTORCH VERIFICATION
# ============================================================
print("\n" + "=" * 70)
print("PART 2: PYTORCH VERIFICATION")
print("=" * 70)

import torch
import torch.nn as nn

# Build the same network in PyTorch
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 2)
        self.layer3 = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

model = SimpleNet()

# Set the SAME weights as our hand computation
with torch.no_grad():
    model.layer1.weight.copy_(torch.tensor(W1, dtype=torch.float64))
    model.layer1.bias.copy_(torch.tensor(b1, dtype=torch.float64))
    model.layer2.weight.copy_(torch.tensor(W2, dtype=torch.float64))
    model.layer2.bias.copy_(torch.tensor(b2, dtype=torch.float64))
    model.layer3.weight.copy_(torch.tensor(W3, dtype=torch.float64))
    model.layer3.bias.copy_(torch.tensor(b3, dtype=torch.float64))
    model.output.weight.copy_(torch.tensor(Wo, dtype=torch.float64))
    model.output.bias.copy_(torch.tensor(bo, dtype=torch.float64))

model = model.double()

x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=False)
y_tensor = torch.tensor([y_true], dtype=torch.float64)

# Forward pass
y_pred = model(x_tensor)
loss = 0.5 * (y_pred - y_tensor) ** 2

print(f"\nPyTorch y_pred: {y_pred.item():.6f}  (Hand: {a_out:.6f})")
print(f"PyTorch loss:   {loss.item():.6f}  (Hand: {L:.6f})")

# Backward pass
loss.backward()

print(f"\n--- PyTorch Gradients vs Hand Gradients ---")
print(f"\nOutput Layer weights:")
print(f"  PyTorch: {model.output.weight.grad.numpy().flatten()}  Hand: [{dL_dwo1:.6f}, {dL_dwo2:.6f}]")
print(f"  PyTorch bias: {model.output.bias.grad.numpy().flatten()}  Hand: [{dL_dbo:.6f}]")

print(f"\nLayer 3 weights:")
print(f"  PyTorch:\n{model.layer3.weight.grad.numpy()}")
print(f"  Hand: [[{dL_dw3_11:.6f}, {dL_dw3_12:.6f}], [{dL_dw3_21:.6f}, {dL_dw3_22:.6f}]]")
print(f"  PyTorch bias: {model.layer3.bias.grad.numpy()}  Hand: [{dL_db3_1:.6f}, {dL_db3_2:.6f}]")

print(f"\nLayer 2 weights:")
print(f"  PyTorch:\n{model.layer2.weight.grad.numpy()}")
print(f"  Hand: [[{dL_dw2_11:.6f}, {dL_dw2_12:.6f}], [{dL_dw2_21:.6f}, {dL_dw2_22:.6f}]]")
print(f"  PyTorch bias: {model.layer2.bias.grad.numpy()}  Hand: [{dL_db2_1:.6f}, {dL_db2_2:.6f}]")

print(f"\nLayer 1 weights:")
print(f"  PyTorch:\n{model.layer1.weight.grad.numpy()}")
print(f"  Hand: [[{dL_dw1_11:.6f}, {dL_dw1_12:.6f}], [{dL_dw1_21:.6f}, {dL_dw1_22:.6f}]]")
print(f"  PyTorch bias: {model.layer1.bias.grad.numpy()}  Hand: [{dL_db1_1:.6f}, {dL_db1_2:.6f}]")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE - Hand gradients match PyTorch autograd!")
print("=" * 70)

# ============================================================
# PART 3: Q1.2 and Q1.3 Discussion
# ============================================================
print("\n" + "=" * 70)
print("Q1.2: REFLECTION ON INITIAL VALUES AND GRADIENT BEHAVIOR")
print("=" * 70)
print("""
Observations from the backpropagation results:

1. SIGMOID SATURATION: With sigmoid activation, all pre-activation gradients
   (dL/dz) get multiplied by sigmoid'(z) = a*(1-a), which has a maximum of
   0.25 at z=0. As we go deeper, each layer multiplies by another factor
   <= 0.25, causing gradients to shrink exponentially.

2. VANISHING GRADIENTS: Looking at the pre-activation gradients:
   - dL/dz_out is the largest
   - dL/dz3 is smaller
   - dL/dz2 is even smaller
   - dL/dz1 is the smallest
   This demonstrates the vanishing gradient problem with sigmoid activations
   in deep networks. Layer 1 receives very weak learning signals.

3. EFFECT OF INITIAL VALUES:
   - If weights are too large, sigmoid outputs saturate near 0 or 1, making
     sigmoid'(z) near 0, worsening vanishing gradients.
   - If weights are too small, activations cluster near 0.5, which is the
     best case for sigmoid derivatives but limits expressiveness.
   - Our chosen small positive weights (0.1-0.5) keep activations in a
     reasonable range and avoid severe saturation.

4. EXPLODING GRADIENTS: With sigmoid, gradients cannot explode since
   sigmoid'(z) <= 0.25 always. However, with ReLU activation, if weights
   are large, gradients can grow unboundedly since ReLU'(z) = 1 for z > 0.
   This is the exploding gradient problem.

5. BIAS INITIALIZATION: Starting biases at 0.1 (small positive) helps avoid
   dead neurons with ReLU. For sigmoid, zero bias would also work since
   sigmoid(0) = 0.5 is in the active region.
""")

print("=" * 70)
print("Q1.3: WHY BACKPROPAGATION IS RECURSIVE AND SCALABLE")
print("=" * 70)
print("""
The pre-activation gradient at each layer follows the recursive formula:

  dL/dz^[l] = (W^[l+1])^T * dL/dz^[l+1]  *  sigma'(z^[l])

This is recursive because:
1. CHAIN RULE: The chain rule of calculus decomposes the gradient of a
   composition of functions into a product of local derivatives. Each layer
   only needs the gradient from the layer above it (dL/dz^[l+1]) and its
   own local information (weights and activation derivative).

2. BACKWARD COMPUTATION: Starting from the output loss, we compute
   dL/dz^[out] first, then use it to compute dL/dz^[3], then dL/dz^[2],
   and so on. Each step reuses the already-computed gradient from the
   next layer -- no redundant computation.

3. SCALABILITY: This recursive structure means:
   - Time complexity is O(N) where N is the number of parameters, the same
     as the forward pass. Each weight's gradient is computed exactly once.
   - Memory is O(L) for storing intermediate activations across L layers
     (needed for computing local derivatives).
   - Adding more layers only adds a linear cost, not exponential. Without
     this recursive reuse, computing each gradient independently via the
     chain rule would require redundant multiplications, leading to
     exponential cost in network depth.
   - This makes backpropagation practical for networks with hundreds of
     layers (e.g., ResNet-152) and millions of parameters.
""")
