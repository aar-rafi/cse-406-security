
import os
import time
import random
import hashlib
import math
from typing import Tuple, Optional, Dict, List, Union

Point = Tuple[int, int]
POINT_AT_INFINITY = None

class EllipticCurve:
    
    def __init__(self, a: int, b: int, p: int, g: Point, n: Optional[int] = None, h: int = 1, name: str = "custom"):
        if not self._is_prime(p):
            raise ValueError("Modulus p must be a prime number")
        
        discriminant = (4 * pow(a, 3, p) + 27 * pow(b, 2, p)) % p
        if discriminant == 0:
            raise ValueError("The curve is singular (4a^3 + 27b^2 = 0 mod p)")
        
        if g is not None and not self._is_on_curve(g[0], g[1], a, b, p):
            raise ValueError("Generator point G is not on the curve")
        
        self.a = a
        self.b = b
        self.p = p
        self.g = g
        self.n = n
        self.h = h
        self.name = name
    
    @staticmethod
    def _is_prime(n: int) -> bool:
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0:
            return False
        
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        for _ in range(40):
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
        if x is None or y is None:
            return True
        
        left = (y * y) % p
        right = (pow(x, 3, p) + a * x + b) % p
        return left == right
    
    def is_on_curve(self, point: Optional[Point]) -> bool:
        if point is POINT_AT_INFINITY:
            return True
        
        if not isinstance(point, tuple) or len(point) != 2:
            return False
        
        x, y = point
        return self._is_on_curve(x, y, self.a, self.b, self.p)
    
    def add_points(self, p1: Optional[Point], p2: Optional[Point]) -> Optional[Point]:
        if p1 is POINT_AT_INFINITY:
            return p2
        if p2 is POINT_AT_INFINITY:
            return p1
        
        x1, y1 = p1
        x2, y2 = p2
        
        if x1 == x2 and (y1 + y2) % self.p == 0:
            return POINT_AT_INFINITY
        
        if x1 == x2:
            numerator = (3 * pow(x1, 2, self.p) + self.a) % self.p
            denominator = (2 * y1) % self.p
            denominator_inv = pow(denominator, self.p - 2, self.p)
            slope = (numerator * denominator_inv) % self.p
        else:
            numerator = (y2 - y1) % self.p
            denominator = (x2 - x1) % self.p
            denominator_inv = pow(denominator, self.p - 2, self.p)
            slope = (numerator * denominator_inv) % self.p
        
        x3 = (pow(slope, 2, self.p) - x1 - x2) % self.p
        y3 = (slope * (x1 - x3) - y1) % self.p
        
        return (x3, y3)
    
    def scalar_multiply(self, k: int, point: Optional[Point]) -> Optional[Point]:
        if k == 0 or point is POINT_AT_INFINITY:
            return POINT_AT_INFINITY
        
        if k < 0:
            k = -k
            x, y = point
            point = (x, (-y) % self.p)
        
        result = POINT_AT_INFINITY
        addend = point
        
        while k:
            if k & 1:
                result = self.add_points(result, addend)
            addend = self.add_points(addend, addend)
            k >>= 1
        
        return result

def generate_prime(bits: int) -> int:
    while True:
        num_bytes = (bits + 7) // 8
        rand_bytes = os.urandom(num_bytes)
        
        p = int.from_bytes(rand_bytes, byteorder='big')
        p |= (1 << (bits - 1)) | 1
        
        if EllipticCurve._is_prime(p):
            return p

def tonelli_shanks(n: int, p: int) -> Optional[int]:
    if pow(n, (p - 1) // 2, p) != 1:
        return None
    
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)
    
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    
    z = 2
    while pow(z, (p - 1) // 2, p) == 1:
        z += 1
    
    m = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)
    
    while t != 1:
        i, temp = 0, t
        while temp != 1:
            temp = pow(temp, 2, p)
            i += 1
            if i == m:
                return None
        
        b = pow(c, pow(2, m - i - 1, p - 1), p)
        m = i
        c = pow(b, 2, p)
        t = (t * c) % p
        r = (r * b) % p
    
    return r

def is_quadratic_residue(n: int, p: int) -> bool:
    return pow(n, (p - 1) // 2, p) == 1

def find_point_on_curve(a: int, b: int, p: int) -> Optional[Point]:
    max_attempts = 100
    
    for _ in range(max_attempts):
        x = random.randint(0, p - 1)
        
        rhs = (pow(x, 3, p) + a * x + b) % p
        
        if is_quadratic_residue(rhs, p):
            y = tonelli_shanks(rhs, p)
            
            if EllipticCurve._is_on_curve(x, y, a, b, p):
                if random.randint(0, 1):
                    y = (-y) % p
                
                return (x, y)
    
    return None

def generate_curve_params(bits: int) -> Dict:
    p = generate_prime(bits)
    
    a = random.randint(0, p - 1)
    b = random.randint(0, p - 1)
    
    while (4 * pow(a, 3, p) + 27 * pow(b, 2, p)) % p == 0:
        a = random.randint(0, p - 1)
        b = random.randint(0, p - 1)
    
    G = find_point_on_curve(a, b, p)
    if G is None:
        b = (b + 1) % p
        G = find_point_on_curve(a, b, p)
    
    n = p
    
    return {
        'a': a,
        'b': b,
        'p': p,
        'g': G,
        'n': n,
        'h': 1,
        'bits': bits
    }

def point_to_bytes(point: Optional[Point]) -> bytes:
    if point is POINT_AT_INFINITY or point is None:
        return b'\x00' * 65
    
    x, y = point
    
    x_bytes = x.to_bytes(32, byteorder='big')
    y_bytes = y.to_bytes(32, byteorder='big')
    
    return b'\x04' + x_bytes + y_bytes

def bytes_to_point(data: bytes) -> Optional[Point]:
    if data is None or len(data) < 65:
        raise ValueError("Invalid point data: too short")
    
    if data[0] == 0 and all(b == 0 for b in data[1:]):
        return POINT_AT_INFINITY
    
    if data[0] != 0x04:
        raise ValueError(f"Invalid point format: expected 0x04, got 0x{data[0]:02x}")
    
    x = int.from_bytes(data[1:33], byteorder='big')
    y = int.from_bytes(data[33:65], byteorder='big')
    
    return (x, y)

class ECDH:
    
    def __init__(self, curve: Union[EllipticCurve, Dict] = None, key_size: int = 128):
        if curve is None:
            curve_params = generate_curve_params(key_size)
            
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
            self.curve = curve
        
        self.key_size = key_size
        self.private_key = None
        self.public_key = None
    
    def generate_keypair(self) -> Tuple[int, Point]:
        if self.curve.n is not None:
            self.private_key = random.randint(1, self.curve.n - 1)
        else:
            self.private_key = random.randint(1, 2**self.key_size - 1)
        
        self.public_key = self.curve.scalar_multiply(self.private_key, self.curve.g)
        
        return self.private_key, self.public_key
    
    def compute_shared_secret(self, other_public_key: Point) -> bytes:
        if not self.curve.is_on_curve(other_public_key):
            raise ValueError("The provided public key is not on the curve")
        
        if self.private_key is None:
            raise ValueError("Private key not generated yet. Call generate_keypair() first")
        
        shared_point = self.curve.scalar_multiply(self.private_key, other_public_key)
        
        if shared_point is POINT_AT_INFINITY:
            raise ValueError("Shared point is the point at infinity")
        
        x_coord = shared_point[0]
        
        raw_secret = x_coord.to_bytes((x_coord.bit_length() + 7) // 8, byteorder='big')
        
        return self.kdf(raw_secret, length=self.key_size // 8)
    
    def kdf(self, input_key_material: bytes, length: int) -> bytes:
        key = hashlib.sha256(input_key_material).digest()
        
        if len(key) == length:
            return key
        elif len(key) > length:
            return key[:length]
        else:
            result = bytearray(key)
            counter = 1
            while len(result) < length:
                counter_bytes = counter.to_bytes(4, byteorder='big')
                additional = hashlib.sha256(input_key_material + counter_bytes).digest()
                result.extend(additional)
                counter += 1
            
            return bytes(result[:length])

def simulate_ecdh_exchange(key_size: int = 128) -> Dict:
    results = {}
    
    start_time = time.time()
    curve_params = generate_curve_params(key_size)
    params_time = time.time() - start_time
    results['params_generation_time'] = params_time * 1000
    
    alice_ecdh = ECDH(curve_params, key_size)
    
    start_time = time.time()
    alice_private, alice_public = alice_ecdh.generate_keypair()
    alice_keypair_time = time.time() - start_time
    results['alice_keypair_time'] = alice_keypair_time * 1000
    
    bob_ecdh = ECDH(curve_params, key_size)
    
    start_time = time.time()
    bob_private, bob_public = bob_ecdh.generate_keypair()
    bob_keypair_time = time.time() - start_time
    results['bob_keypair_time'] = bob_keypair_time * 1000
    
    start_time = time.time()
    alice_shared_secret = alice_ecdh.compute_shared_secret(bob_public)
    alice_shared_time = time.time() - start_time
    results['alice_shared_time'] = alice_shared_time * 1000
    
    start_time = time.time()
    bob_shared_secret = bob_ecdh.compute_shared_secret(alice_public)
    bob_shared_time = time.time() - start_time
    results['bob_shared_time'] = bob_shared_time * 1000
    
    results['shared_secrets_match'] = (alice_shared_secret == bob_shared_secret)
    
    results['curve_params'] = curve_params
    results['alice_private'] = alice_private
    results['alice_public'] = alice_public
    results['bob_private'] = bob_private
    results['bob_public'] = bob_public
    results['shared_secret'] = alice_shared_secret.hex()
    
    return results

def main():
    print("Elliptic Curve Diffie-Hellman Key Exchange Demonstration")
    print("=" * 60)
    print()
    
    key_sizes = [128, 192, 256]
    performance_results = {}
    
    for key_size in key_sizes:
        print(f"Testing ECDH with {key_size}-bit key")
        print("-" * 40)
        
        num_trials = 5
        total_a_keypair_time = 0
        total_b_keypair_time = 0
        total_a_shared_time = 0
        
        for i in range(num_trials):
            print(f"Trial {i+1}/{num_trials}...")
            results = simulate_ecdh_exchange(key_size)
            
            total_a_keypair_time += results['alice_keypair_time']
            total_b_keypair_time += results['bob_keypair_time']
            total_a_shared_time += results['alice_shared_time']
            
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
        
        avg_a_keypair_time = total_a_keypair_time / num_trials
        avg_b_keypair_time = total_b_keypair_time / num_trials
        avg_shared_secret_time = total_a_shared_time / num_trials
        
        performance_results[key_size] = {
            'keypair_generation': avg_a_keypair_time,
            'shared_secret_computation': avg_shared_secret_time
        }
        
        print("\nPerformance (averaged over {num_trials} trials):")
        print(f"  Key pair generation: {avg_a_keypair_time:.6f} ms")
        print(f"  Shared secret computation: {avg_shared_secret_time:.6f} ms")
        print()
    
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