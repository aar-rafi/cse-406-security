#!/usr/bin/env python3
"""
Lab Configuration
=================

Centralized configuration for the namespace lab environment.
Modify these settings to match your specific lab setup.
"""

# Network Configuration
LAB_CONFIG = {
    # Namespace names
    'victim_namespace': 'victim',
    'attacker_namespace': 'spoofed',
    
    # IP addresses
    'victim_ip': '192.168.100.2',
    'attacker_ip': '192.168.100.3',
    'gateway_ip': '192.168.100.1',
    'network_subnet': '192.168.100.0/24',
    
    # Test targets (local/simulated)
    'local_targets': [
        '10.0.0.1',      # Simulated external server
        '172.16.0.1',    # Another simulated target
        '203.0.113.1',   # TEST-NET-3 (RFC 5737)
        '192.0.2.1',     # TEST-NET-1 (RFC 5737)
    ],
    
    # Interface names (auto-detected if None)
    'victim_interface': 'v-veth',
    'attacker_interface': 's-veth',
    
    # Attack parameters
    'default_duration': 30,
    'flood_packet_delay': 0.01,
    'redirect_interval': 5,
}

def get_config():
    """Get the lab configuration"""
    return LAB_CONFIG

def get_attack_scenarios():
    """Get predefined attack scenarios"""
    config = get_config()
    
    scenarios = {
        'basic_spoof': {
            'name': 'Basic ICMP Spoofing',
            'description': 'Send spoofed ICMP packets from attacker to victim',
            'target': config['victim_ip'],
            'source': config['attacker_ip'],
            'namespace': config['attacker_namespace'],
        },
        
        'cross_spoof': {
            'name': 'Cross-namespace Spoofing',
            'description': 'Spoof packets with external source to victim',
            'target': config['victim_ip'],
            'source': config['local_targets'][0],
            'namespace': config['attacker_namespace'],
        },
        
        'local_redirect': {
            'name': 'Local Target Redirect',
            'description': 'Redirect victim traffic to local target via fake gateway',
            'victim': config['victim_ip'],
            'target': config['local_targets'][0],
            'gateway': config['gateway_ip'],
            'fake_gateway': config['attacker_ip'],
            'attack_namespace': config['attacker_namespace'],
            'monitor_namespace': config['victim_namespace'],
        },
        
        'internal_redirect': {
            'name': 'Internal Network Redirect', 
            'description': 'Redirect victim to attacker IP (simulated server)',
            'victim': config['victim_ip'],
            'target': config['attacker_ip'],
            'gateway': config['gateway_ip'],
            'fake_gateway': config['attacker_ip'],
            'attack_namespace': config['attacker_namespace'],
            'monitor_namespace': config['victim_namespace'],
        },
        
        'test_net_redirect': {
            'name': 'Test Network Redirect',
            'description': 'Redirect to RFC 5737 test network addresses',
            'victim': config['victim_ip'],
            'target': config['local_targets'][2],  # TEST-NET-3
            'gateway': config['gateway_ip'],
            'fake_gateway': config['attacker_ip'],
            'attack_namespace': config['attacker_namespace'],
            'monitor_namespace': config['victim_namespace'],
        }
    }
    
    return scenarios

def print_lab_info():
    """Print current lab configuration"""
    config = get_config()
    
    print("Lab Configuration:")
    print("=" * 50)
    print(f"Victim Namespace: {config['victim_namespace']} ({config['victim_ip']})")
    print(f"Attacker Namespace: {config['attacker_namespace']} ({config['attacker_ip']})")
    print(f"Gateway: {config['gateway_ip']}")
    print(f"Network: {config['network_subnet']}")
    print(f"\nLocal Test Targets:")
    for i, target in enumerate(config['local_targets'], 1):
        print(f"  {i}. {target}")

if __name__ == '__main__':
    print_lab_info()
    print("\nAvailable Scenarios:")
    scenarios = get_attack_scenarios()
    for key, scenario in scenarios.items():
        print(f"  {key}: {scenario['name']}")
        print(f"    {scenario['description']}") 