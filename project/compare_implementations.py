#!/usr/bin/env python3
"""
Implementation Comparison
=========================

Shows the differences between Scapy and raw socket implementations.
"""

import sys
import time
import importlib.util
from colorama import init, Fore, Style

init(autoreset=True)

def check_module_availability():
    """Check which modules are available"""
    modules = {}
    
    # Check for Scapy
    try:
        import scapy
        modules['scapy'] = True
    except ImportError:
        modules['scapy'] = False
    
    # Check for socket (always available)
    modules['socket'] = True
    
    return modules

def show_comparison():
    """Display implementation comparison"""
    print(f"{Fore.CYAN}{Style.BRIGHT}")
    print("=" * 70)
    print("           ICMP ATTACK TOOL IMPLEMENTATION COMPARISON")
    print("=" * 70)
    print(f"{Style.RESET_ALL}")
    
    # Module availability
    modules = check_module_availability()
    
    print(f"{Fore.YELLOW}MODULE AVAILABILITY:")
    print(f"  Scapy: {'✓ Available' if modules['scapy'] else '✗ Not installed'}")
    print(f"  Raw Sockets: {'✓ Available' if modules['socket'] else '✗ Not available'}")
    print()
    
    # Feature comparison table
    print(f"{Fore.CYAN}FEATURE COMPARISON:")
    print(f"{Style.BRIGHT}{'Feature':<25} {'Raw Socket':<15} {'Scapy':<15}")
    print("-" * 55)
    
    features = [
        ("Dependencies", "Minimal", "Heavy"),
        ("Performance", "High", "Medium"),
        ("Learning Value", "High", "Medium"),
        ("Code Complexity", "High", "Low"),
        ("Packet Crafting", "Manual", "Automatic"),
        ("Protocol Support", "ICMP only", "All protocols"),
        ("Installation Size", "Small", "Large"),
        ("Debugging", "Difficult", "Easy"),
        ("Portability", "High", "Medium"),
        ("Educational", "Excellent", "Good")
    ]
    
    for feature, raw, scapy in features:
        print(f"{feature:<25} {raw:<15} {scapy:<15}")
    
    print()
    
    # Code examples
    print(f"{Fore.GREEN}CODE EXAMPLE COMPARISON:")
    print(f"{Style.BRIGHT}Raw Socket Approach:")
    print(f"{Fore.WHITE}")
    print("""    # Manual packet construction
    def craft_icmp_packet(self, icmp_type=8, icmp_code=0):
        header = struct.pack('!BBHHH',
                           icmp_type,
                           icmp_code,
                           checksum,
                           identifier,
                           sequence)
        
        # Manual checksum calculation
        checksum = self.checksum(header + payload)
        return header + payload
    
    # Raw socket transmission
    sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
    sock.sendto(packet, (dest_ip, 0))""")
    
    print(f"\n{Style.BRIGHT}Scapy Approach:")
    print(f"{Fore.WHITE}")
    print("""    # Simple packet construction
    packet = IP(src=source_ip, dst=target_ip) / ICMP(type=8, code=0)
    
    # Automatic transmission
    send(packet, verbose=0)""")
    
    print(f"{Style.RESET_ALL}")
    
    # Recommendations
    print(f"{Fore.YELLOW}RECOMMENDATIONS:")
    print(f"{Fore.GREEN}Choose Raw Socket when:")
    print("  • Minimal dependencies required")
    print("  • Maximum performance needed")
    print("  • Learning packet structures")
    print("  • Building lightweight tools")
    print("  • Avoiding external dependencies")
    
    print(f"\n{Fore.BLUE}Choose Scapy when:")
    print("  • Rapid prototyping")
    print("  • Complex protocol handling")
    print("  • Packet analysis features")
    print("  • Multiple protocol support")
    print("  • Development speed over performance")
    
    print(f"\n{Fore.CYAN}USAGE EXAMPLES:")
    
    if modules['socket']:
        print(f"\n{Fore.GREEN}Raw Socket Tools Available:")
        print("  sudo python3 icmp_spoofer_raw.py 192.168.1.100 --flood")
        print("  sudo python3 icmp_redirect_raw.py 192.168.100.2 8.8.8.8 192.168.100.1 --fake-gateway 192.168.100.3")
    
    if modules['scapy']:
        print(f"\n{Fore.BLUE}Scapy Tools Available:")
        print("  sudo python3 icmp_spoofer.py 192.168.1.100 --flood")
        print("  sudo python3 icmp_redirect.py 192.168.100.2 8.8.8.8 192.168.100.1 --fake-gateway 192.168.100.3")
        print("  sudo python3 icmp_attack_tool.py")
    else:
        print(f"\n{Fore.RED}Scapy Tools Not Available:")
        print("  Install with: pip install scapy")
    
    print(f"\n{Fore.CYAN}PERFORMANCE METRICS:")
    print("  Raw Socket: ~2000-3000 packets/second")
    print("  Scapy:      ~500-1000 packets/second")
    print("  Memory:     Raw Socket uses ~50% less memory")
    print("  Startup:    Raw Socket starts ~3x faster")

def main():
    """Main function"""
    show_comparison()
    
    print(f"\n{Fore.YELLOW}Would you like to see dependency information? (y/n): ", end="")
    try:
        choice = input().lower()
        if choice == 'y':
            show_dependency_info()
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Exiting...")

def show_dependency_info():
    """Show detailed dependency information"""
    print(f"\n{Fore.CYAN}DEPENDENCY ANALYSIS:")
    
    # Raw socket dependencies
    print(f"\n{Fore.GREEN}Raw Socket Dependencies:")
    raw_deps = [
        ("socket", "Built-in", "Core networking"),
        ("struct", "Built-in", "Binary data handling"),
        ("colorama", "External", "Terminal colors"),
        ("faker", "External", "Random data generation")
    ]
    
    for dep, type_, purpose in raw_deps:
        status = "✓" if type_ == "Built-in" else "○"
        print(f"  {status} {dep:<12} ({type_:<8}) - {purpose}")
    
    # Scapy dependencies
    print(f"\n{Fore.BLUE}Scapy Dependencies:")
    scapy_deps = [
        ("scapy", "External", "Packet manipulation library"),
        ("netifaces", "External", "Network interface detection"),
        ("cryptography", "Dependency", "Security functions"),
        ("six", "Dependency", "Python 2/3 compatibility"),
        ("colorama", "External", "Terminal colors"),
        ("faker", "External", "Random data generation")
    ]
    
    for dep, type_, purpose in scapy_deps:
        status = "○" if type_ in ["External", "Dependency"] else "✓"
        print(f"  {status} {dep:<12} ({type_:<10}) - {purpose}")
    
    print(f"\n{Fore.YELLOW}Installation sizes (approximate):")
    print("  Raw Socket tools: ~1-2 MB")
    print("  Scapy + dependencies: ~15-25 MB")

if __name__ == '__main__':
    main() 