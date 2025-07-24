#!/usr/bin/env python3
"""
ICMP Attack Monitor Dashboard
=============================

Real-time visual monitoring dashboard for ICMP attacks.
Shows live statistics, network activity, and attack effects.
"""

import sys
import os
import time
import threading
import subprocess
import json
from datetime import datetime, timedelta
from colorama import init, Fore, Back, Style

try:
    from scapy.all import sniff, IP, ICMP
except ImportError:
    print("Error: Scapy not installed. Please run: pip install scapy")
    sys.exit(1)

init(autoreset=True)

class AttackMonitorDashboard:
    """Real-time attack monitoring dashboard"""
    
    def __init__(self):
        self.monitoring = False
        self.stats = {
            'total_packets': 0,
            'icmp_packets': 0,
            'spoof_packets': 0,
            'redirect_packets': 0,
            'responses': 0,
            'routing_changes': 0,
            'start_time': None,
            'last_activity': None
        }
        self.recent_packets = []
        self.routing_history = {}
        self.alert_messages = []
        
    def show_banner(self):
        """Display dashboard banner"""
        print(f"{Fore.CYAN}{Style.BRIGHT}")
        print("██╗ ██████╗███╗   ███╗██████╗     ██████╗  █████╗ ███████╗██╗  ██╗██████╗  ██████╗  █████╗ ██████╗ ██╗  ██╗")
        print("██║██╔════╝████╗ ████║██╔══██╗    ██╔══██╗██╔══██╗██╔════╝██║  ██║██╔══██╗██╔═══██╗██╔══██╗██╔══██╗██║  ██║")
        print("██║██║     ██╔████╔██║██████╔╝    ██║  ██║███████║███████╗███████║██████╔╝██║   ██║███████║██████╔╝███████║")
        print("██║██║     ██║╚██╔╝██║██╔═══╝     ██║  ██║██╔══██║╚════██║██╔══██║██╔══██╗██║   ██║██╔══██║██╔══██╗██╔══██║")
        print("██║╚██████╗██║ ╚═╝ ██║██║         ██████╔╝██║  ██║███████║██║  ██║██████╔╝╚██████╔╝██║  ██║██║  ██║██║  ██║")
        print("╚═╝ ╚═════╝╚═╝     ╚═╝╚═╝         ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝")
        print(f"{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}🔥 Real-time ICMP Attack Monitoring Dashboard")
        print(f"{Fore.RED}⚠️  Live attack detection and analysis")
        print()
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear')
    
    def get_uptime(self):
        """Get monitoring uptime"""
        if self.stats['start_time']:
            uptime = datetime.now() - self.stats['start_time']
            return str(uptime).split('.')[0]  # Remove microseconds
        return "00:00:00"
    
    def add_alert(self, message, alert_type="INFO"):
        """Add alert message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = Fore.GREEN if alert_type == "SUCCESS" else Fore.RED if alert_type == "ATTACK" else Fore.YELLOW
        self.alert_messages.append(f"{color}[{timestamp}] {message}{Style.RESET_ALL}")
        
        # Keep only last 10 alerts
        if len(self.alert_messages) > 10:
            self.alert_messages.pop(0)
    
    def display_dashboard(self):
        """Display the main dashboard"""
        while self.monitoring:
            self.clear_screen()
            self.show_banner()
            
            # Statistics Panel
            print(f"{Fore.CYAN}{'='*80}")
            print(f"{Fore.CYAN}                           ATTACK STATISTICS")
            print(f"{Fore.CYAN}{'='*80}")
            
            # Main stats
            uptime = self.get_uptime()
            print(f"{Fore.GREEN}📊 MONITORING STATUS:")
            print(f"   ⏰ Uptime: {uptime}")
            print(f"   📦 Total Packets: {self.stats['total_packets']}")
            print(f"   🧊 ICMP Packets: {self.stats['icmp_packets']}")
            print(f"   🔀 Spoofed Packets: {self.stats['spoof_packets']}")
            print(f"   📤 Redirect Packets: {self.stats['redirect_packets']}")
            print(f"   📥 Responses: {self.stats['responses']}")
            print(f"   🔄 Routing Changes: {self.stats['routing_changes']}")
            
            # Activity indicator
            if self.stats['last_activity']:
                time_since = (datetime.now() - self.stats['last_activity']).seconds
                if time_since < 5:
                    activity_status = f"{Fore.RED}🔥 ACTIVE ATTACK"
                elif time_since < 30:
                    activity_status = f"{Fore.YELLOW}⚡ RECENT ACTIVITY"
                else:
                    activity_status = f"{Fore.GREEN}💤 IDLE"
            else:
                activity_status = f"{Fore.BLUE}🔍 WAITING FOR TRAFFIC"
            
            print(f"\n{Fore.YELLOW}📡 NETWORK STATUS: {activity_status}")
            
            # Recent packets panel
            print(f"\n{Fore.CYAN}{'='*80}")
            print(f"{Fore.CYAN}                         RECENT PACKET ACTIVITY")
            print(f"{Fore.CYAN}{'='*80}")
            
            if self.recent_packets:
                for packet_info in self.recent_packets[-5:]:  # Show last 5 packets
                    print(f"   {packet_info}")
            else:
                print(f"{Fore.LIGHTBLACK_EX}   No recent activity...")
            
            # Alerts panel
            print(f"\n{Fore.CYAN}{'='*80}")
            print(f"{Fore.CYAN}                            SECURITY ALERTS")
            print(f"{Fore.CYAN}{'='*80}")
            
            if self.alert_messages:
                for alert in self.alert_messages[-5:]:  # Show last 5 alerts
                    print(f"   {alert}")
            else:
                print(f"{Fore.LIGHTBLACK_EX}   No alerts...")
            
            # Real-time graphs (simple ASCII)
            print(f"\n{Fore.CYAN}{'='*80}")
            print(f"{Fore.CYAN}                         ATTACK INTENSITY GRAPH")
            print(f"{Fore.CYAN}{'='*80}")
            self.display_activity_graph()
            
            print(f"\n{Fore.YELLOW}Press Ctrl+C to stop monitoring...")
            print(f"{Fore.CYAN}{'='*80}")
            
            time.sleep(1)
    
    def display_activity_graph(self):
        """Display simple ASCII activity graph"""
        # Simple activity visualization
        max_bars = 50
        if self.stats['icmp_packets'] > 0:
            intensity = min(self.stats['icmp_packets'] // 10, max_bars)
            bar = "█" * intensity + "░" * (max_bars - intensity)
            print(f"   ICMP: {Fore.RED}{bar}{Style.RESET_ALL} [{self.stats['icmp_packets']}]")
        
        if self.stats['spoof_packets'] > 0:
            intensity = min(self.stats['spoof_packets'] // 5, max_bars)
            bar = "█" * intensity + "░" * (max_bars - intensity)
            print(f"   SPOOF: {Fore.YELLOW}{bar}{Style.RESET_ALL} [{self.stats['spoof_packets']}]")
        
        if self.stats['redirect_packets'] > 0:
            intensity = min(self.stats['redirect_packets'] // 2, max_bars)
            bar = "█" * intensity + "░" * (max_bars - intensity)
            print(f"   REDIRECT: {Fore.MAGENTA}{bar}{Style.RESET_ALL} [{self.stats['redirect_packets']}]")
    
    def packet_handler(self, packet):
        """Handle captured packets"""
        self.stats['total_packets'] += 1
        self.stats['last_activity'] = datetime.now()
        
        if packet.haslayer(ICMP):
            self.stats['icmp_packets'] += 1
            self.analyze_icmp_packet(packet)
        
        # Keep recent packets list manageable
        if len(self.recent_packets) > 20:
            self.recent_packets.pop(0)
    
    def analyze_icmp_packet(self, packet):
        """Analyze ICMP packet for attack patterns"""
        icmp_type = packet[ICMP].type
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if icmp_type == 8:  # Echo Request
            # Check for potential spoofing patterns
            if self.is_suspicious_source(src_ip):
                self.stats['spoof_packets'] += 1
                packet_info = f"{Fore.YELLOW}[{timestamp}] SUSPICIOUS PING: {src_ip} → {dst_ip}"
                self.add_alert(f"Suspicious ICMP echo from {src_ip}", "ATTACK")
            else:
                packet_info = f"{Fore.BLUE}[{timestamp}] PING: {src_ip} → {dst_ip}"
                
        elif icmp_type == 0:  # Echo Reply
            self.stats['responses'] += 1
            packet_info = f"{Fore.GREEN}[{timestamp}] PONG: {src_ip} → {dst_ip}"
            
        elif icmp_type == 5:  # Redirect
            self.stats['redirect_packets'] += 1
            gateway = getattr(packet[ICMP], 'gw', 'Unknown')
            packet_info = f"{Fore.RED}[{timestamp}] REDIRECT: {src_ip} → {dst_ip} (GW: {gateway})"
            self.add_alert(f"ICMP Redirect detected: {src_ip} → {dst_ip}", "ATTACK")
            
        elif icmp_type == 3:  # Destination Unreachable
            packet_info = f"{Fore.MAGENTA}[{timestamp}] UNREACHABLE: {src_ip} → {dst_ip}"
            
        else:
            packet_info = f"{Fore.LIGHTBLACK_EX}[{timestamp}] ICMP-{icmp_type}: {src_ip} → {dst_ip}"
        
        self.recent_packets.append(packet_info)
    
    def is_suspicious_source(self, src_ip):
        """Check if source IP looks suspicious (basic heuristics)"""
        # Simple heuristics for demo purposes
        suspicious_patterns = [
            src_ip.startswith('10.'),
            src_ip.startswith('172.'),
            src_ip.startswith('192.168.'),
            src_ip in ['127.0.0.1', '0.0.0.0']
        ]
        
        # If we see unusual private IP patterns or common spoofing IPs
        if src_ip.count('.') == 3:
            octets = src_ip.split('.')
            try:
                # Check for unusual patterns
                if all(int(octet) < 256 for octet in octets):
                    # Random pattern detection (simplified)
                    return any([
                        octets[0] in ['1', '2', '3'],  # Unusual first octets
                        src_ip.endswith('.1'),  # Gateway IPs
                        len(set(octets)) == 1,  # Same digits
                    ])
            except ValueError:
                pass
        
        return False
    
    def monitor_routing_changes(self):
        """Monitor for routing table changes"""
        while self.monitoring:
            try:
                # Get current routing table
                result = subprocess.run(['ip', 'route', 'show'], 
                                      capture_output=True, text=True, check=True)
                current_routes = result.stdout.strip()
                
                # Check for changes
                routes_hash = hash(current_routes)
                if hasattr(self, 'last_routes_hash'):
                    if routes_hash != self.last_routes_hash:
                        self.stats['routing_changes'] += 1
                        self.add_alert("Routing table change detected!", "ATTACK")
                
                self.last_routes_hash = routes_hash
                time.sleep(5)  # Check every 5 seconds
                
            except:
                time.sleep(5)
    
    def start_monitoring(self, interface=None, filter_str="icmp"):
        """Start the monitoring dashboard"""
        self.monitoring = True
        self.stats['start_time'] = datetime.now()
        
        # Start dashboard display thread
        dashboard_thread = threading.Thread(target=self.display_dashboard)
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        # Start routing monitor thread
        routing_thread = threading.Thread(target=self.monitor_routing_changes)
        routing_thread.daemon = True
        routing_thread.start()
        
        try:
            self.add_alert("Attack monitoring started", "SUCCESS")
            # Start packet sniffing
            sniff(filter=filter_str, prn=self.packet_handler, iface=interface)
            
        except KeyboardInterrupt:
            self.monitoring = False
            self.add_alert("Monitoring stopped by user", "INFO")
            print(f"\n{Fore.YELLOW}⚠️  Monitoring stopped")
            self.show_final_report()
    
    def show_final_report(self):
        """Show final attack report"""
        print(f"\n{Fore.CYAN}📊 FINAL ATTACK REPORT")
        print("=" * 50)
        print(f"⏰ Total monitoring time: {self.get_uptime()}")
        print(f"📦 Total packets captured: {self.stats['total_packets']}")
        print(f"🧊 ICMP packets: {self.stats['icmp_packets']}")
        print(f"🔀 Suspicious packets: {self.stats['spoof_packets']}")
        print(f"📤 Redirect attacks: {self.stats['redirect_packets']}")
        print(f"🔄 Routing changes: {self.stats['routing_changes']}")
        
        if self.stats['redirect_packets'] > 0 or self.stats['spoof_packets'] > 0:
            print(f"\n{Fore.RED}⚠️  ATTACK ACTIVITY DETECTED!")
            print(f"{Fore.YELLOW}   Consider investigating network security")
        else:
            print(f"\n{Fore.GREEN}✅ No attack activity detected")
        
        print("=" * 50)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ICMP Attack Monitor Dashboard")
    parser.add_argument('--interface', '-i', help='Network interface to monitor')
    parser.add_argument('--filter', '-f', default='icmp', help='Packet filter (default: icmp)')
    parser.add_argument('--targets', '-t', help='Comma-separated list of IPs to monitor')
    
    args = parser.parse_args()
    
    # Check root privileges
    if os.geteuid() != 0:
        print(f"{Fore.RED}❌ Root privileges required for packet capture")
        print(f"{Fore.YELLOW}💡 Run with: sudo python3 {sys.argv[0]}")
        sys.exit(1)
    
    # Setup filter
    filter_str = args.filter
    if args.targets:
        target_ips = args.targets.split(',')
        host_filter = ' or '.join([f'host {ip.strip()}' for ip in target_ips])
        filter_str = f"({filter_str}) and ({host_filter})"
    
    print(f"{Fore.CYAN}🔍 Filter: {filter_str}")
    if args.interface:
        print(f"{Fore.CYAN}🔌 Interface: {args.interface}")
    print()
    
    # Start monitoring
    monitor = AttackMonitorDashboard()
    monitor.start_monitoring(args.interface, filter_str)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️  Dashboard terminated")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error: {e}")
        sys.exit(1) 