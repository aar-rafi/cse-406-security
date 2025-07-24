"""
author: 20005039 
Usage:
    python 2005039_ecc.py --generate 128
    python 2005039_ecc.py --measure
"""

import sys
import time
import argparse
import os
import json
from typing import Dict, List
import statistics

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ecc import (ECDH, generate_curve_params, simulate_ecdh_exchange)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Elliptic Curve Diffie-Hellman Key Exchange Implementation"
    )
    
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--generate", type=int, choices=[128, 192, 256], help="Generate curve parameters and simulate key exchange (specify bit length)")
    action.add_argument("--measure", action="store_true", help="Measure performance for different key sizes")
    action.add_argument("--simulate", action="store_true", help="Simulate a complete ECDH exchange")

    parser.add_argument("--trials", type=int, default=5, help="Number of trials for performance measurement (default: 5)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    
    return parser.parse_args()

def generate_and_display(key_size: int, verbose: bool = False):
    print(f"Generating {key_size}-bit Elliptic Curve Parameters")
    print("-" * 60)

    results = simulate_ecdh_exchange(key_size)
    

    print("Curve Parameters:")
    print(f"  a = {results['curve_params']['a']}")
    print(f"  b = {results['curve_params']['b']}")
    print(f"  p = {results['curve_params']['p']} ({results['curve_params']['p'].bit_length()} bits)")
    print(f"  G = {results['curve_params']['g']}")
    
    if verbose:
        print("\nDetailed Information:")
        print("Alice's Key Pair:")
        print(f"  Private Key: {results['alice_private']}")
        print(f"  Public Key: {results['alice_public']}")
        
        print("\nBob's Key Pair:")
        print(f"  Private Key: {results['bob_private']}")
        print(f"  Public Key: {results['bob_public']}")
    
    print("\nShared Secret:")
    print(f"  {results['shared_secret']}")
    print(f"  (Matching: {'Yes' if results['shared_secrets_match'] else 'No'})")
    
    print("\nPerformance:")
    print(f"  Parameter Generation: {results['params_generation_time']:.6f} ms")
    print(f"  Alice's Key Pair Generation: {results['alice_keypair_time']:.6f} ms")
    print(f"  Bob's Key Pair Generation: {results['bob_keypair_time']:.6f} ms")
    print(f"  Alice's Shared Secret Computation: {results['alice_shared_time']:.6f} ms")
    print(f"  Bob's Shared Secret Computation: {results['bob_shared_time']:.6f} ms")

def measure_performance(num_trials: int = 5):

    key_sizes = [128, 192, 256]
    results = {}
    
    print("Measuring ECDH Performance")
    print("=" * 60)
    
    for key_size in key_sizes:
        print(f"\nTesting with {key_size}-bit keys ({num_trials} trials)...")
        
        param_times = []
        key_gen_times = []
        shared_secret_times = []
        
        for i in range(num_trials):
            print(f"  Trial {i+1}/{num_trials}...", end="", flush=True)
            
            # meas parameter generation
            start_time = time.time()
            params = generate_curve_params(key_size)
            param_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            param_times.append(param_time)
            
            alice = ECDH(params, key_size)
            bob = ECDH(params, key_size)
            
            # meas key generation
            start_time = time.time()
            alice_private, alice_public = alice.generate_keypair()
            key_gen_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            key_gen_times.append(key_gen_time)
            
            bob_private, bob_public = bob.generate_keypair()
            
            # meas shared secret computation
            start_time = time.time()
            alice_shared = alice.compute_shared_secret(bob_public)
            shared_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            shared_secret_times.append(shared_time)
            
            bob_shared = bob.compute_shared_secret(alice_public)
            
            # verify
            if alice_shared != bob_shared:
                print(" ERROR: Shared secrets don't match!")
            else:
                print(" Done")
        
        # Just store the mean values which are used for the summary table
        results[key_size] = {
            'param_generation': statistics.mean(param_times),
            'key_generation': statistics.mean(key_gen_times),
            'shared_secret': statistics.mean(shared_secret_times)
        }
    
    # Create the simplified table for Task 2 requirement
    print("\nSummary Table for Task 2")
    print("=" * 60)
    print(f"{'k':^8} | {'Computation Time For':^40}")
    print(f"{' ':^8} | {'-'*40}")
    print(f"{' ':^8} | {'A':^12} | {'B':^12} | {'shared key R':^12}")
    print("-" * 60)
    
    for key_size in key_sizes:
        key_time = results[key_size]['key_generation']
        shared_time = results[key_size]['shared_secret']
        print(f"{key_size:^8} | {key_time:^12.3f} | {key_time:^12.3f} | {shared_time:^12.3f}")
    
    print("=" * 60)
    print("All times in milliseconds, averaged over multiple trials")

def simulate_exchange():
    """Simulate a complete ECDH exchange between two parties."""
    key_size = 128  # Default to 128-bit
    
    print("Simulating Complete ECDH Exchange")
    print("=" * 60)
    print(f"Using {key_size}-bit parameters")
    
    # Generate curve parameters
    print("\nStep 1: Generate curve parameters (a, b, p, G)...")
    start_time = time.time()
    params = generate_curve_params(key_size)
    param_time = (time.time() - start_time) * 1000
    
    print(f"  Generated parameters in {param_time:.3f} ms")
    print(f"  a = {params['a']}")
    print(f"  b = {params['b']}")
    print(f"  p = {params['p']}")
    print(f"  G = {params['g']}")
    
    # Initialize Alice and Bob with the same parameters
    print("\nStep 2: Alice and Bob initialize with parameters...")
    alice = ECDH(params, key_size)
    bob = ECDH(params, key_size)
    
    # Alice generates her key pair
    print("\nStep 3: Alice generates private key k_a and public key A = k_a * G...")
    start_time = time.time()
    alice_private, alice_public = alice.generate_keypair()
    alice_time = (time.time() - start_time) * 1000
    
    print(f"  Generated Alice's key pair in {alice_time:.3f} ms")
    print(f"  k_a = {alice_private}")
    print(f"  A = {alice_public}")
    
    # Bob generates his key pair
    print("\nStep 4: Bob generates private key k_b and public key B = k_b * G...")
    start_time = time.time()
    bob_private, bob_public = bob.generate_keypair()
    bob_time = (time.time() - start_time) * 1000
    
    print(f"  Generated Bob's key pair in {bob_time:.3f} ms")
    print(f"  k_b = {bob_private}")
    print(f"  B = {bob_public}")
    
    # Alice computes the shared secret
    print("\nStep 5: Alice computes shared secret using her private key and Bob's public key...")
    start_time = time.time()
    alice_shared = alice.compute_shared_secret(bob_public)
    alice_shared_time = (time.time() - start_time) * 1000
    
    print(f"  Computed shared secret in {alice_shared_time:.3f} ms")
    print(f"  R_alice = k_a * B = {alice_shared.hex()}")
    
    # Bob computes the shared secret
    print("\nStep 6: Bob computes shared secret using his private key and Alice's public key...")
    start_time = time.time()
    bob_shared = bob.compute_shared_secret(alice_public)
    bob_shared_time = (time.time() - start_time) * 1000
    
    print(f"  Computed shared secret in {bob_shared_time:.3f} ms")
    print(f"  R_bob = k_b * A = {bob_shared.hex()}")
    
    # Verify the shared secrets match
    print("\nStep 7: Verify the shared secrets match...")
    if alice_shared == bob_shared:
        print("  SUCCESS: Shared secrets match! k_a * B = k_b * A")
    else:
        print("  ERROR: Shared secrets don't match!")
    
    print("\nSummary:")
    print(f"  Parameter generation: {param_time:.3f} ms")
    print(f"  Alice's key generation: {alice_time:.3f} ms")
    print(f"  Bob's key generation: {bob_time:.3f} ms")
    print(f"  Alice's shared secret computation: {alice_shared_time:.3f} ms")
    print(f"  Bob's shared secret computation: {bob_shared_time:.3f} ms")

def main():
    args = parse_args()
    
    if args.generate:
        generate_and_display(args.generate, args.verbose)
    
    elif args.measure:
        measure_performance(args.trials)
    
    elif args.simulate:
        simulate_exchange()

if __name__ == "__main__":
    main()