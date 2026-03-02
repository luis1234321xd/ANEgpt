#!/usr/bin/env python3
"""Test the ANE Python bridge — verifies matmul kernel runs on Neural Engine."""

import numpy as np
import sys
import os

# Add nanochat to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nanochat.ane_bridge import ANEBridge, generate_conv_mil

def test_matmul():
    """Test a simple matmul (conv) on ANE: y = W @ x"""
    bridge = ANEBridge()
    bridge.init()
    print("✓ ANE initialized")

    in_ch, out_ch, spatial = 64, 128, 16
    
    # Create random weights and input
    np.random.seed(42)
    W = np.random.randn(out_ch, in_ch).astype(np.float32) * 0.01
    x = np.random.randn(in_ch, spatial).astype(np.float32)
    
    # Expected output (CPU reference)
    y_expected = W @ x  # [out_ch, spatial]
    
    # Build weight blob
    w_blob = bridge.build_weight_blob(W)
    print(f"✓ Weight blob built: {len(w_blob)} bytes")
    
    # Generate MIL
    mil_text = generate_conv_mil(in_ch, out_ch, spatial)
    print(f"✓ MIL generated: {len(mil_text)} chars")
    
    # Compile
    in_bytes = in_ch * spatial * 4  # fp32
    out_bytes = out_ch * spatial * 4
    kernel = bridge.compile(mil_text, weight_data=w_blob,
                            input_sizes=[in_bytes], output_sizes=[out_bytes])
    print(f"✓ Kernel compiled (compile count: {bridge.compile_count})")
    
    # Write input in channel-first format [1, C, 1, S] -> just [C, S] contiguous
    x_ane = np.ascontiguousarray(x, dtype=np.float32)  # already [in_ch, spatial]
    kernel.write_input(0, x_ane)
    
    # Evaluate
    ok = kernel.eval()
    print(f"✓ Kernel evaluated: {'success' if ok else 'FAILED'}")
    
    # Read output
    y_ane = kernel.read_output(0, shape=(out_ch, spatial), dtype=np.float32)
    
    # Compare
    max_err = np.max(np.abs(y_ane - y_expected))
    mean_err = np.mean(np.abs(y_ane - y_expected))
    
    print(f"\n=== Results ===")
    print(f"  Max absolute error:  {max_err:.6f}")
    print(f"  Mean absolute error: {mean_err:.6f}")
    print(f"  y_expected[:3,:3]: {y_expected[:3,:3]}")
    print(f"  y_ane[:3,:3]:      {y_ane[:3,:3]}")
    
    # fp16 precision: errors up to ~0.01 are expected
    if max_err < 0.1:
        print(f"\n✓ TEST PASSED — ANE matmul matches CPU within fp16 precision")
    else:
        print(f"\n✗ TEST FAILED — errors too large")
        sys.exit(1)
    
    kernel.free()
    print(f"✓ Kernel freed")

def test_two_layer_training():
    """Test a simple 2-layer training loop on ANE."""
    bridge = ANEBridge()
    bridge.init()
    
    D, H, S = 64, 128, 16
    lr = 1.0
    
    np.random.seed(42)
    W1 = np.random.randn(H, D).astype(np.float32) * 0.01
    W2 = np.random.randn(D, H).astype(np.float32) * 0.01
    
    # Target: identity mapping
    x = np.sin(np.arange(S * D).reshape(S, D) * 0.1).astype(np.float32)
    y_target = x.copy()
    
    losses = []
    
    for step in range(50):
        # Forward on ANE: h = W1 @ x^T, y = W2 @ relu(h)
        mil_fwd1 = generate_conv_mil(D, H, S)
        w1_blob = bridge.build_weight_blob(W1)
        k_fwd1 = bridge.compile(mil_fwd1, w1_blob,
                                [D * S * 4], [H * S * 4])
        
        # x in channel-first: [D, S]
        x_cf = x.T.copy().astype(np.float32)
        k_fwd1.write_input(0, x_cf)
        k_fwd1.eval()
        h = k_fwd1.read_output(0, (H, S), np.float32)
        k_fwd1.free()
        
        # ReLU on CPU
        h_relu = np.maximum(h, 0)
        
        # y = W2 @ h_relu
        mil_fwd2 = generate_conv_mil(H, D, S)
        w2_blob = bridge.build_weight_blob(W2)
        k_fwd2 = bridge.compile(mil_fwd2, w2_blob,
                                [H * S * 4], [D * S * 4])
        k_fwd2.write_input(0, h_relu)
        k_fwd2.eval()
        y_cf = k_fwd2.read_output(0, (D, S), np.float32)
        k_fwd2.free()
        
        # Loss (MSE)
        y = y_cf.T  # back to [S, D]
        loss = np.mean((y - y_target) ** 2)
        losses.append(loss)
        
        # Backward on CPU (simple)
        dy = 2.0 * (y - y_target) / (S * D)  # [S, D]
        dy_cf = dy.T  # [D, S]
        
        # Backward through W2: dh_relu = W2^T @ dy_cf
        mil_bwd2 = generate_conv_mil(D, H, S)
        w2t_blob = bridge.build_weight_blob(W2.T.copy())
        k_bwd2 = bridge.compile(mil_bwd2, w2t_blob,
                                [D * S * 4], [H * S * 4])
        k_bwd2.write_input(0, dy_cf.astype(np.float32))
        k_bwd2.eval()
        dh_relu = k_bwd2.read_output(0, (H, S), np.float32)
        k_bwd2.free()
        
        # ReLU backward
        dh = dh_relu * (h > 0).astype(np.float32)
        
        # dW on CPU
        dW2 = dy_cf @ h_relu.T  # [D, H]
        dW1 = dh @ x_cf.T       # [H, D]
        
        # SGD update
        W1 -= lr * dW1
        W2 -= lr * dW2
        
        if step % 10 == 0:
            print(f"  step {step:3d}  loss={loss:.6f}  compiles={bridge.compile_count}")
        
        # Check compile budget
        if bridge.compile_count > 90:
            print(f"  ⚠ Approaching compile limit, stopping early")
            break
    
    if len(losses) > 1 and losses[-1] < losses[0]:
        print(f"\n✓ TRAINING TEST PASSED — loss decreased: {losses[0]:.6f} → {losses[-1]:.6f}")
    else:
        print(f"\n✗ TRAINING TEST FAILED — loss did not decrease")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 50)
    print("ANE Bridge Test Suite")
    print("=" * 50)
    
    print("\n--- Test 1: Simple Matmul ---")
    test_matmul()
    
    print("\n--- Test 2: Two-Layer Training ---")
    test_two_layer_training()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
