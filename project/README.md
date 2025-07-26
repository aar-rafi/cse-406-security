# ICMP Attack Tool

## Advanced ICMP Attack Tool with Real-time Monitoring

Comprehensive ICMP spoofing and redirect attack tool featuring real-time visual monitoring, automatic attack verification, and advanced attack modes.

### Available Versions

This project provides two implementation approaches:

#### 1. Scapy-based Tools (High-level)
- `icmp_spoofer.py` - Full-featured with packet sniffing
- `icmp_redirect.py` - Advanced redirect attacks
- `icmp_attack_tool.py` - Interactive tool with all features

#### 2. Raw Socket Tools (Low-level, No Dependencies)
- `icmp_spoofer_raw.py` - Pure Python with socket module
- `icmp_redirect_raw.py` - Lightweight redirect attacks
- Reduced dependencies (no Scapy required)
- Better performance and control
- Educational value for understanding packet structures

### Requirements

#### For Scapy-based tools:
```bash
pip install scapy netifaces colorama faker
```

#### For Raw Socket tools:
```bash
pip install netifaces colorama faker
```

## Key Features

- **Multiple Attack Types**: ICMP spoofing, redirect attacks, flood attacks
- **Real-time Monitoring**: Live packet statistics and routing change detection
- **Visual Dashboard**: ASCII art banners and colored output
- **Attack Verification**: Automatic detection of successful attacks
- **Namespace Support**: Works with Linux network namespaces
- **Continuous Mode**: Sustained attacks for persistent route poisoning
- **Raw Socket Option**: No external dependencies beyond Python standard library

## Quick Start

### Raw Socket Version (Recommended for minimal setup)

```bash
# ICMP Spoofing
sudo python3 icmp_spoofer_raw.py 192.168.1.100 --source 10.0.0.1

# Flood Attack
sudo python3 icmp_spoofer_raw.py 192.168.1.100 --flood --duration 30

# ICMP Redirect
sudo python3 icmp_redirect_raw.py 192.168.100.2 8.8.8.8 192.168.100.1 --fake-gateway 192.168.100.3

# Continuous Redirect
sudo python3 icmp_redirect_raw.py 192.168.100.2 8.8.8.8 192.168.100.1 --fake-gateway 192.168.100.3 --continuous

# With Namespace Monitoring
sudo python3 icmp_redirect_raw.py 192.168.100.2 8.8.8.8 192.168.100.1 --fake-gateway 192.168.100.3 --namespace victim
```

### Scapy Version (Full features)

```bash
# Interactive tool
sudo python3 icmp_attack_tool.py

# Direct attacks
sudo python3 icmp_spoofer.py 192.168.1.100 --flood --duration 60
sudo python3 icmp_redirect.py 192.168.100.2 8.8.8.8 192.168.100.1 --fake-gateway 192.168.100.3
```

## Attack Modules

### ICMP Spoofing
- **Purpose**: Send forged ICMP packets with fake source IPs
- **Methods**: Single packet, flood attack, stealth scan
- **Detection**: Response monitoring, packet statistics

### ICMP Redirect Attack
- **Purpose**: Manipulate victim's routing table
- **Methods**: Single redirect, continuous poisoning
- **Verification**: Real-time route monitoring, manual verification steps

## Visual Monitoring Features

- Live packet counters
- Real-time routing change detection
- Attack success indicators
- Performance statistics
- Network state visualization

## Advanced Usage

### Network Namespace Setup
```bash
# Create isolated test environment
sudo ip netns add victim
sudo ip netns add spoofed
sudo ip link add v-veth type veth peer name veth-host
sudo ip link add s-veth type veth peer name veth-host2
```

### Attack Verification
```bash
# Monitor victim routing table
sudo ip netns exec victim watch -n 1 'ip route get 8.8.8.8'

# Capture attack traffic
sudo ip netns exec spoofed tcpdump -i s-veth icmp -nn
```

## Technical Details

### Raw Socket Implementation
- Manual IP and ICMP header construction using `struct.pack()`
- Internet checksum calculation
- Direct socket transmission with `SOCK_RAW`
- No external packet manipulation libraries

### Packet Structure
```
IP Header (20 bytes) + ICMP Header (8+ bytes) + Payload
```

### ICMP Redirect Structure
```
IP(src=gateway, dst=victim) / ICMP(type=5, code=1, gw=fake_gw) / Original_Packet
```

## Security Note

This tool is for authorized security testing and educational purposes only. Ensure you have proper authorization before use.

## Dependencies Comparison

| Feature | Raw Socket | Scapy |
|---------|------------|-------|
| Dependencies | Python stdlib + colorama | scapy + netifaces |
| Performance | High | Medium |
| Learning Value | High | Medium |
| Features | Core attacks | Full suite |
| Size | Lightweight | Heavy | 