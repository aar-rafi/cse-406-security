#!/usr/bin/env python3
"""
Elliptic Curve Cryptography Implementation

This module implements elliptic curve cryptography operations and the Diffie-Hellman
key exchange protocol.
"""

import os
import time
import random
import hashlib
import math
from typing import Tuple, Optional, Dict, List, Union

Point = Tuple[int, int]  # Type alias for EC Point (x, y)
# Special point to represent point at infinity (identity element)
POINT_AT_INFINITY = None

class EllipticCurve:
    """
    Represents an elliptic curve over a finite field.
    
    The curve has the form: y^2 = x^3 + ax + b (mod p)
    """
    
    def __init__(self, a: int, b: int, p: int, g: Point, n: Optional[int] = None, h: int = 1, name: str = "custom"):
        """
        Initialize an elliptic curve with the given parameters.
        
        Args:
            a: Coefficient 'a' in the curve equation
            b: Coefficient 'b' in the curve equation
            p: Prime modulus defining the finite field
            g: Generator/base point on the curve
            n: Order of the generator point (optional)
            h: Cofactor (optional, default is 1)
            name: Name of the curve (optional)
        """
        # Validate parameters
        if not self._is_prime(p):
            raise ValueError("Modulus p must be a prime number")
        
        # Check that 4a^3 + 27b^2 ≠ 0 (mod p) to ensure the curve is non-singular
        discriminant = (4 * pow(a, 3, p) + 27 * pow(b, 2, p)) % p
        if discriminant == 0:
            raise ValueError("The curve is singular (4a^3 + 27b^2 = 0 mod p)")
        
        # Check that the generator point is on the curve
        if g is not None and not self._is_on_curve(g[0], g[1], a, b, p):
            raise ValueError("Generator point G is not on the curve")
        
        self.a = a
        self.b = b
        self.p = p
        self.g = g
        self.n = n  # Order of the base point
        self.h = h  # Cofactor
        self.name = name
    
    @staticmethod
    def _is_prime(n: int) -> bool:
        """
        Check if a number is prime using the Miller-Rabin primality test.
        
        Args:
            n: Number to check
            
        Returns:
            True if the number is (probably) prime, False otherwise
        """
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0:
            return False
        
        # Miller-Rabin primality test
        # Express n-1 as 2^r * d where d is odd
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        # Witness loop
        for _ in range(40):  # Number of iterations for high confidence
            a = random.randint(2, n - 2)
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        
        return True
    
    @staticmethod
    def _is_on_curve(x: int, y: int, a: int, b: int, p: int) -> bool:
        """
        Check if a point (x, y) lies on the elliptic curve defined by y^2 = x^3 + ax + b (mod p).
        
        Args:
            x: x-coordinate of the point
            y: y-coordinate of the point
            a: Coefficient 'a' in the curve equation
            b: Coefficient 'b' in the curve equation
            p: Prime modulus
            
        Returns:
            True if the point is on the curve, False otherwise
        """
        if x is None or y is None:  # Point at infinity
            return True
        
        # Check if the point satisfies the curve equation: y^2 = x^3 + ax + b (mod p)
        left = (y * y) % p
        right = (pow(x, 3, p) + a * x + b) % p
        return left == right
    
    def is_on_curve(self, point: Optional[Point]) -> bool:
        """
        Check if a point is on this elliptic curve.
        
        Args:
            point: The point to check, or None for point at infinity
            
        Returns:
            True if the point is on the curve, False otherwise
        """
        if point is POINT_AT_INFINITY:
            return True
        
        if not isinstance(point, tuple) or len(point) != 2:
            return False
        
        x, y = point
        return self._is_on_curve(x, y, self.a, self.b, self.p)
    
    def add_points(self, p1: Optional[Point], p2: Optional[Point]) -> Optional[Point]:
        """
        Add two points on the elliptic curve.
        
        Args:
            p1: First point (or None for point at infinity)
            p2: Second point (or None for point at infinity)
            
        Returns:
            The sum of the two points
        """
        # Handle point at infinity
        if p1 is POINT_AT_INFINITY:
            return p2
        if p2 is POINT_AT_INFINITY:
            return p1
        
        # Extract coordinates
        x1, y1 = p1
        x2, y2 = p2
        
        # P + (-P) = O (point at infinity)
        if x1 == x2 and (y1 + y2) % self.p == 0:
            return POINT_AT_INFINITY
        
        # Calculate the slope of the line through p1 and p2
        if x1 == x2:  # Point doubling
            # λ = (3x₁² + a) / (2y₁) mod p
            numerator = (3 * pow(x1, 2, self.p) + self.a) % self.p
            denominator = (2 * y1) % self.p
            # Modular inverse of denominator
            denominator_inv = pow(denominator, self.p - 2, self.p)  # Fermat's little theorem
            slope = (numerator * denominator_inv) % self.p
        else:  # Point addition
            # λ = (y₂ - y₁) / (x₂ - x₁) mod p
            numerator = (y2 - y1) % self.p
            denominator = (x2 - x1) % self.p
            # Modular inverse of denominator
            denominator_inv = pow(denominator, self.p - 2, self.p)  # Fermat's little theorem
            slope = (numerator * denominator_inv) % self.p
        
        # Calculate the coordinates of the sum
        # x₃ = λ² - x₁ - x₂ mod p
        x3 = (pow(slope, 2, self.p) - x1 - x2) % self.p
        # y₃ = λ(x₁ - x₃) - y₁ mod p
        y3 = (slope * (x1 - x3) - y1) % self.p
        
        return (x3, y3)
    
    def scalar_multiply(self, k: int, point: Optional[Point]) -> Optional[Point]:
        """
        Multiply a point by a scalar using the double-and-add algorithm.
        
        Args:
            k: Scalar multiplier
            point: Point to multiply
            
        Returns:
            Resulting point after scalar multiplication
        """
        if k == 0 or point is POINT_AT_INFINITY:
            return POINT_AT_INFINITY
        
        # Handle negative scalars
        if k < 0:
            k = -k
            # Negate the point
            x, y = point
            point = (x, (-y) % self.p)
        
        # Double-and-add algorithm for scalar multiplication
        result = POINT_AT_INFINITY
        addend = point
        
        while k:
            if k & 1:  # If least significant bit is 1
                result = self.add_points(result, addend)
            addend = self.add_points(addend, addend)  # Double the point
            k >>= 1  # Shift right (divide by 2)
        
        return result

# Utility functions for curve generation
def generate_prime(bits: int) -> int:
    """
    Generate a random prime number with the specified bit length.
    
    Args:
        bits: Number of bits in the prime
        
    Returns:
        A random prime number
    """
    # Generate a random number with the specified bit length
    while True:
        # Generate random bytes and convert to integer
        num_bytes = (bits + 7) // 8  # Convert bits to bytes, rounding up
        rand_bytes = os.urandom(num_bytes)
        
        # Convert to integer and ensure proper bit length
        p = int.from_bytes(rand_bytes, byteorder='big')
        # Set the highest and lowest bit to 1 to ensure we get a number of the right bit length
        # that is also odd (no even number can be prime except 2)
        p |= (1 << (bits - 1)) | 1
        
        # Basic primality test
        if EllipticCurve._is_prime(p):
            return p

def tonelli_shanks(n: int, p: int) -> Optional[int]:
    """
    Tonelli-Shanks algorithm to find modular square root.
    Solves the equation x^2 ≡ n (mod p) for x.
    
    Args:
        n: The number to find the square root of
        p: The prime modulus
        
    Returns:
        A square root of n modulo p, or None if n is not a quadratic residue
    """
    # Check if n is a quadratic residue using Euler's Criterion
    if pow(n, (p - 1) // 2, p) != 1:
        return None  # n is not a quadratic residue
    
    # Special case for p ≡ 3 (mod 4)
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)
    
    # Factor p-1 as 2^s * q where q is odd
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    
    # Find a quadratic non-residue z
    z = 2
    while pow(z, (p - 1) // 2, p) == 1:
        z += 1
    
    # Initialize variables
    m = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)
    
    # Main loop
    while t != 1:
        # Find the least i, 0 < i < m, such that t^(2^i) ≡ 1 (mod p)
        i, temp = 0, t
        while temp != 1:
            temp = pow(temp, 2, p)
            i += 1
            if i == m:
                return None  # No solution exists
        
        # Update values
        b = pow(c, pow(2, m - i - 1, p - 1), p)
        m = i
        c = pow(b, 2, p)
        t = (t * c) % p
        r = (r * b) % p
    
    return r

def is_quadratic_residue(n: int, p: int) -> bool:
    """
    Check if n is a quadratic residue modulo p using Euler's criterion.
    
    Args:
        n: The number to check
        p: The prime modulus
        
    Returns:
        True if n is a quadratic residue modulo p, False otherwise
    """
    return pow(n, (p - 1) // 2, p) == 1

def find_point_on_curve(a: int, b: int, p: int) -> Optional[Point]:
    """
    Find a random point on the elliptic curve y^2 = x^3 + ax + b (mod p).
    
    Args:
        a: Coefficient 'a' in the curve equation
        b: Coefficient 'b' in the curve equation
        p: Prime modulus
        
    Returns:
        A random point on the curve, or None if no point could be found
    """
    max_attempts = 100  # Limit the number of attempts
    
    for _ in range(max_attempts):
        # Generate a random x-coordinate
        x = random.randint(0, p - 1)
        
        # Calculate the right side of the equation: x^3 + ax + b (mod p)
        rhs = (pow(x, 3, p) + a * x + b) % p
        
        # Check if rhs is a perfect square in the field
        if is_quadratic_residue(rhs, p):
            # Find y such that y^2 = rhs (mod p)
            y = tonelli_shanks(rhs, p)
            
            # Verify the point is on the curve
            if EllipticCurve._is_on_curve(x, y, a, b, p):
                # Randomly choose between the two possible y values
                if random.randint(0, 1):
                    y = (-y) % p
                
                return (x, y)
    
    # If we get here, we couldn't find a point after max_attempts
    return None

def generate_curve_params(bits: int) -> Dict:
    """
    Generate parameters for an elliptic curve of the specified bit length.
    
    Args:
        bits: Bit length of the prime field (128, 192, or 256)
        
    Returns:
        Dictionary containing the curve parameters
    """
    # Generate a random prime for the field
    p = generate_prime(bits)
    
    # Choose random coefficients for the curve: y^2 = x^3 + ax + b
    a = random.randint(0, p - 1)
    b = random.randint(0, p - 1)
    
    # Ensure the curve is non-singular: 4a^3 + 27b^2 ≠ 0 (mod p)
    while (4 * pow(a, 3, p) + 27 * pow(b, 2, p)) % p == 0:
        a = random.randint(0, p - 1)
        b = random.randint(0, p - 1)
    
    # Find a point on the curve
    G = find_point_on_curve(a, b, p)
    if G is None:
        # If we couldn't find a point, adjust b and try again
        b = (b + 1) % p
        G = find_point_on_curve(a, b, p)
    
    # For a proper implementation, we should calculate the order (n) of the curve,
    # but that's computationally expensive. For this exercise, we'll assume the curve
    # has a large order approximately equal to p.
    # In a real-world implementation, you would use a proper algorithm to calculate the order.
    n = p  # This is a simplification!
    
    return {
        'a': a,
        'b': b,
        'p': p,
        'g': G,
        'n': n,
        'h': 1,  # Cofactor (simplified)
        'bits': bits
    }

def point_to_bytes(point: Optional[Point]) -> bytes:
    """
    Convert an elliptic curve point to bytes for serialization.
    
    Args:
        point: The point to serialize, or None for point at infinity
        
    Returns:
        Serialized point as bytes
    """
    if point is POINT_AT_INFINITY or point is None:
        # Special value for point at infinity
        return b'\x00' * 65  # 1 byte type + 32 bytes each for x and y (all zeros)
    
    x, y = point
    
    # Convert to bytes with fixed length (32 bytes each for x and y)
    x_bytes = x.to_bytes(32, byteorder='big')
    y_bytes = y.to_bytes(32, byteorder='big')
    
    # Format: 0x04 (uncompressed point) + x + y
    return b'\x04' + x_bytes + y_bytes

def bytes_to_point(data: bytes) -> Optional[Point]:
    """
    Convert serialized bytes back to an elliptic curve point.
    
    Args:
        data: The serialized point data
        
    Returns:
        Deserialized point, or None for point at infinity
    """
    if data is None or len(data) < 65:
        raise ValueError("Invalid point data: too short")
    
    # Check if it's the point at infinity
    if data[0] == 0 and all(b == 0 for b in data[1:]):
        return POINT_AT_INFINITY
    
    # Check that it's an uncompressed point
    if data[0] != 0x04:
        raise ValueError(f"Invalid point format: expected 0x04, got 0x{data[0]:02x}")
    
    # Extract the x and y coordinates
    x = int.from_bytes(data[1:33], byteorder='big')
    y = int.from_bytes(data[33:65], byteorder='big')
    
    return (x, y)

class ECDH:
    """
    Elliptic Curve Diffie-Hellman key exchange implementation.
    """
    
    def __init__(self, curve: Union[EllipticCurve, Dict] = None, key_size: int = 128):
        """
        Initialize the ECDH implementation with curve parameters.
        
        Args:
            curve: EllipticCurve instance or dictionary containing curve parameters, or None to generate
            key_size: Key size in bits (128, 192, or 256) if generating parameters
        """
        if curve is None:
            # Generate curve parameters
            curve_params = generate_curve_params(key_size)
            
            # Initialize the elliptic curve
            self.curve = EllipticCurve(
                a=curve_params['a'],
                b=curve_params['b'],
                p=curve_params['p'],
                g=curve_params['g'],
                n=curve_params['n'],
                h=curve_params.get('h', 1),
                name=curve_params.get('name', f"custom-{key_size}")
            )
        elif isinstance(curve, dict):
            # Initialize from parameters dictionary
            self.curve = EllipticCurve(
                a=curve['a'],
                b=curve['b'],
                p=curve['p'],
                g=curve['g'],
                n=curve['n'],
                h=curve.get('h', 1),
                name=curve.get('name', f"custom-{key_size}")
            )
        else:
            # Use the provided EllipticCurve instance
            self.curve = curve
        
        self.key_size = key_size
        self.private_key = None
        self.public_key = None
    
    def generate_keypair(self) -> Tuple[int, Point]:
        """
        Generate a private and public key pair.
        
        Returns:
            Tuple containing (private_key, public_key)
        """
        # Generate a random private key
        if self.curve.n is not None:
            # Private key should be in range [1, n-1]
            self.private_key = random.randint(1, self.curve.n - 1)
        else:
            # If order is not known, use a random integer with appropriate bit length
            self.private_key = random.randint(1, 2**self.key_size - 1)
        
        # Compute the public key
        self.public_key = self.curve.scalar_multiply(self.private_key, self.curve.g)
        
        return self.private_key, self.public_key
    
    def compute_shared_secret(self, other_public_key: Point) -> bytes:
        """
        Compute the shared secret using this instance's private key and the other party's public key.
        
        Args:
            other_public_key: The other party's public key
            
        Returns:
            The shared secret as bytes
        """
        if not self.curve.is_on_curve(other_public_key):
            raise ValueError("The provided public key is not on the curve")
        
        if self.private_key is None:
            raise ValueError("Private key not generated yet. Call generate_keypair() first")
        
        # Compute the shared point
        shared_point = self.curve.scalar_multiply(self.private_key, other_public_key)
        
        if shared_point is POINT_AT_INFINITY:
            raise ValueError("Shared point is the point at infinity")
        
        # Use the x-coordinate of the shared point as the raw shared secret
        x_coord = shared_point[0]
        
        # Format as bytes with proper length
        raw_secret = x_coord.to_bytes((x_coord.bit_length() + 7) // 8, byteorder='big')
        
        # Use a key derivation function to derive the final key
        return self.kdf(raw_secret, length=self.key_size // 8)
    
    def kdf(self, input_key_material: bytes, length: int) -> bytes:
        """
        Simple key derivation function based on SHA-256.
        
        Args:
            input_key_material: The raw input key material
            length: Desired length of the derived key in bytes
            
        Returns:
            The derived key
        """
        # Use SHA-256 as a basic KDF
        key = hashlib.sha256(input_key_material).digest()
        
        # If we need more or less bits, adjust the output
        if len(key) == length:
            return key
        elif len(key) > length:
            return key[:length]
        else:
            # If we need more bits, we can iterate the hash (not ideal, but simple)
            result = bytearray(key)
            counter = 1
            while len(result) < length:
                # Concatenate counter to avoid duplicate outputs
                counter_bytes = counter.to_bytes(4, byteorder='big')
                additional = hashlib.sha256(input_key_material + counter_bytes).digest()
                result.extend(additional)
                counter += 1
            
            return bytes(result[:length])

def simulate_ecdh_exchange(key_size: int = 128) -> Dict:
    """
    Simulate a complete ECDH key exchange between Alice and Bob.
    
    Args:
        key_size: Key size in bits (128, 192, or 256)
        
    Returns:
        Dictionary with results and timing information
    """
    results = {}
    
    # Generate parameters
    start_time = time.time()
    curve_params = generate_curve_params(key_size)
    params_time = time.time() - start_time
    results['params_generation_time'] = params_time * 1000  # Convert to milliseconds
    
    # Initialize Alice's ECDH with the parameters
    alice_ecdh = ECDH(curve_params, key_size)
    
    # Generate Alice's keypair
    start_time = time.time()
    alice_private, alice_public = alice_ecdh.generate_keypair()
    alice_keypair_time = time.time() - start_time
    results['alice_keypair_time'] = alice_keypair_time * 1000  # Convert to milliseconds
    
    # Initialize Bob's ECDH with the same parameters
    bob_ecdh = ECDH(curve_params, key_size)
    
    # Generate Bob's keypair
    start_time = time.time()
    bob_private, bob_public = bob_ecdh.generate_keypair()
    bob_keypair_time = time.time() - start_time
    results['bob_keypair_time'] = bob_keypair_time * 1000  # Convert to milliseconds
    
    # Alice computes the shared secret
    start_time = time.time()
    alice_shared_secret = alice_ecdh.compute_shared_secret(bob_public)
    alice_shared_time = time.time() - start_time
    results['alice_shared_time'] = alice_shared_time * 1000  # Convert to milliseconds
    
    # Bob computes the shared secret
    start_time = time.time()
    bob_shared_secret = bob_ecdh.compute_shared_secret(alice_public)
    bob_shared_time = time.time() - start_time
    results['bob_shared_time'] = bob_shared_time * 1000  # Convert to milliseconds
    
    # Verify that they derived the same key
    results['shared_secrets_match'] = (alice_shared_secret == bob_shared_secret)
    
    # Include the curve parameters and keys in the results
    results['curve_params'] = curve_params
    results['alice_private'] = alice_private
    results['alice_public'] = alice_public
    results['bob_private'] = bob_private
    results['bob_public'] = bob_public
    results['shared_secret'] = alice_shared_secret.hex()
    
    return results

# Main function to demonstrate the implementation
def main():
    """Run a demonstration of the ECDH key exchange."""
    print("Elliptic Curve Diffie-Hellman Key Exchange Demonstration")
    print("=" * 60)
    print()
    
    key_sizes = [128, 192, 256]
    performance_results = {}
    
    for key_size in key_sizes:
        print(f"Testing ECDH with {key_size}-bit key")
        print("-" * 40)
        
        # Run multiple trials
        num_trials = 5
        total_a_keypair_time = 0
        total_b_keypair_time = 0
        total_a_shared_time = 0
        
        for i in range(num_trials):
            print(f"Trial {i+1}/{num_trials}...")
            results = simulate_ecdh_exchange(key_size)
            
            # Accumulate times for averaging
            total_a_keypair_time += results['alice_keypair_time']
            total_b_keypair_time += results['bob_keypair_time']
            total_a_shared_time += results['alice_shared_time']
            
            # Display results for first trial in detail
            if i == 0:
                print(f"Curve parameters:")
                print(f"  a = {results['curve_params']['a']}")
                print(f"  b = {results['curve_params']['b']}")
                print(f"  p = {results['curve_params']['p']} ({results['curve_params']['p'].bit_length()} bits)")
                print(f"  G = {results['curve_params']['g']}")
                
                print("\nKeys:")
                print(f"  Alice's private key: {results['alice_private']}")
                print(f"  Alice's public key: {results['alice_public']}")
                print(f"  Bob's private key: {results['bob_private']}")
                print(f"  Bob's public key: {results['bob_public']}")
                
                print("\nShared Secret:")
                print(f"  {results['shared_secret']}")
                
                print("\nVerification:")
                print(f"  Shared secrets match: {results['shared_secrets_match']}")
        
        # Calculate averages
        avg_a_keypair_time = total_a_keypair_time / num_trials
        avg_b_keypair_time = total_b_keypair_time / num_trials
        avg_shared_secret_time = total_a_shared_time / num_trials
        
        # Store performance results
        performance_results[key_size] = {
            'keypair_generation': avg_a_keypair_time,
            'shared_secret_computation': avg_shared_secret_time
        }
        
        print("\nPerformance (averaged over {num_trials} trials):")
        print(f"  Key pair generation: {avg_a_keypair_time:.6f} ms")
        print(f"  Shared secret computation: {avg_shared_secret_time:.6f} ms")
        print()
    
    # Print performance comparison table
    print("\nPerformance Comparison Table")
    print("=" * 60)
    print(f"{'Key Size':^12} | {'Key Pair Gen (ms)':^20} | {'Shared Secret (ms)':^20}")
    print("-" * 60)
    
    for size in key_sizes:
        res = performance_results[size]
        print(f"{size:^12} | {res['keypair_generation']:^20.6f} | {res['shared_secret_computation']:^20.6f}")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 