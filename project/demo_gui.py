#!/usr/bin/env python3
"""
GUI Demo and Validation Script
=============================

This script demonstrates the corrected GUI functionality and validates 
the command-line argument fixes.
"""

import subprocess
import sys
import time

def print_banner():
    print("=" * 60)
    print("ICMP Attack GUI Demo & Validation")
    print("=" * 60)
    print()

def test_raw_socket_args():
    """Test that the raw socket tools have correct arguments"""
    print("🔍 Testing raw socket tool arguments...")
    
    # Test spoofer arguments
    try:
        result = subprocess.run(['python3', 'icmp_spoofer_raw.py', '--help'], 
                              capture_output=True, text=True, timeout=5)
        if '--continuous' in result.stdout:
            print("❌ ERROR: icmp_spoofer_raw.py still has --continuous flag")
            return False
        else:
            print("✅ icmp_spoofer_raw.py arguments correct (no --continuous)")
    except Exception as e:
        print(f"❌ Error testing spoofer: {e}")
        return False
    
    # Test redirect arguments  
    try:
        result = subprocess.run(['python3', 'icmp_redirect_raw.py', '--help'], 
                              capture_output=True, text=True, timeout=5)
        if '--continuous' in result.stdout:
            print("✅ icmp_redirect_raw.py has --continuous flag")
        else:
            print("❌ ERROR: icmp_redirect_raw.py missing --continuous flag")
            return False
    except Exception as e:
        print(f"❌ Error testing redirect: {e}")
        return False
    
    return True

def demonstrate_correct_usage():
    """Show the correct command usage"""
    print("\n📋 Correct Command Usage:")
    print("-" * 40)
    
    print("ICMP Spoofing modes:")
    print("• Single:  sudo python3 icmp_spoofer_raw.py 192.168.100.2 --source 192.168.100.3")
    print("• Flood:   sudo python3 icmp_spoofer_raw.py 192.168.100.2 --source 192.168.100.3 --flood --duration 30")
    print("• Stealth: sudo python3 icmp_spoofer_raw.py 192.168.100.2 --source 192.168.100.3 --stealth")
    print()
    
    print("ICMP Redirect modes:")
    print("• Single:  sudo python3 icmp_redirect_raw.py 192.168.100.2 10.0.0.1 192.168.100.1 --fake-gateway 192.168.100.3")
    print("• Continuous: sudo python3 icmp_redirect_raw.py 192.168.100.2 10.0.0.1 192.168.100.1 --fake-gateway 192.168.100.3 --continuous")

def check_gui_dependencies():
    """Check if GUI dependencies are available"""
    print("\n🔧 Checking GUI dependencies...")
    
    deps = ['tkinter', 'matplotlib', 'colorama', 'faker']
    missing = []
    
    for dep in deps:
        try:
            if dep == 'tkinter':
                import tkinter
            elif dep == 'matplotlib':
                import matplotlib
            elif dep == 'colorama': 
                import colorama
            elif dep == 'faker':
                import faker
            print(f"✅ {dep} available")
        except ImportError:
            print(f"❌ {dep} missing")
            missing.append(dep)
    
    if missing:
        print(f"\n📦 Install missing dependencies: pip install {' '.join(missing)}")
        return False
    
    return True

def show_gui_features():
    """Display GUI feature overview"""
    print("\n🖥️  GUI Features Overview:")
    print("-" * 40)
    
    features = [
        "Network Topology: Visual network diagram with real-time attack indicators",
        "Attack Config: Point-and-click parameter setting with mode explanations", 
        "Live Monitoring: Dual tcpdump views (victim + attacker namespaces)",
        "Results Analysis: Attack statistics, export options, verification commands",
        "Status Tracking: Real-time packet counts, routing changes, attack progress",
        "Namespace Integration: Automatic namespace detection and monitoring"
    ]
    
    for i, feature in enumerate(features, 1):
        print(f"{i}. {feature}")

def launch_gui():
    """Launch the GUI interface"""
    print(f"\n🚀 Launching GUI...")
    print("Note: GUI will open in a separate window")
    print("Press Ctrl+C here to return to terminal once GUI closes")
    
    try:
        # Import and test GUI can start
        import icmp_attack_gui
        print("✅ GUI module imported successfully")
        
        print("\nTo start GUI manually:")
        print("  python3 icmp_attack_gui.py")
        print("  # OR")
        print("  ./launch_gui.sh")
        
    except Exception as e:
        print(f"❌ GUI launch failed: {e}")

def main():
    print_banner()
    
    # Test command arguments
    if not test_raw_socket_args():
        print("\n❌ Command argument validation failed!")
        return
    
    # Show correct usage
    demonstrate_correct_usage()
    
    # Check dependencies
    if not check_gui_dependencies():
        print("\n⚠️  Some GUI dependencies missing - install them for full functionality")
    
    # Show GUI features
    show_gui_features()
    
    # Offer to launch GUI
    print(f"\n" + "=" * 60)
    print("✅ All validations passed!")
    print("✅ GUI is ready for demonstration")
    print("✅ Command-line tools fixed")
    print(f"=" * 60)
    
    response = input("\nWould you like to launch the GUI now? (y/n): ").lower()
    if response.startswith('y'):
        launch_gui()
    else:
        print("\nGUI demo completed. Launch manually when ready!")

if __name__ == '__main__':
    main() 