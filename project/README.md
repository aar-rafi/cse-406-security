# ICMP Attack Tool

## ⚡ Advanced ICMP Attack Tool with Real-time Monitoring

Comprehensive ICMP spoofing and redirect attack tool featuring real-time visual monitoring, automatic attack verification, and advanced attack modes.

**⚠️ WARNING:** Use only in authorized environments. Unauthorized use is illegal.

## 🔥 Key Features

### **Real-time Visual Monitoring**
- Live attack statistics dashboard
- ASCII art attack banners  
- Real-time packet capture and analysis
- Routing table change detection
- Color-coded status indicators

### **Advanced Attack Modes**
- **ICMP Spoofing**: High-speed packet spoofing with response monitoring
- **ICMP Redirect**: Routing table manipulation with verification
- **Continuous Attack**: Sustained attack with live monitoring
- **Stealth Mode**: Random timing to avoid detection
- **Flood Attack**: High-speed packet flooding
- **MITM Setup**: Automatic man-in-the-middle configuration

### **Attack Verification**
- Automatic routing change detection
- Live packet capture and analysis
- Success rate calculations
- Attack effectiveness monitoring
- Verification command suggestions

## 🚀 Quick Start

### Installation (Arch Linux)
```bash
# Run the automated setup
./setup_arch.sh

# Manual installation
sudo pacman -S python python-pip tcpdump
pip install -r requirements.txt
```

### Basic Usage

#### Interactive Mode
```bash
sudo python3 icmp_attack_tool.py
```

#### Direct Attacks
```bash
# ICMP Spoofing
sudo python3 icmp_spoofer.py 192.168.1.1 --source 10.0.0.1 --count 10

# ICMP Redirect
sudo python3 icmp_redirect.py 192.168.1.20 8.8.8.8 192.168.1.1 --fake-gateway 192.168.1.100
```

## 🎯 Attack Modules

### 1. ICMP Spoofing (`icmp_spoofer.py`)

**Capabilities:**
- Packet spoofing with customizable source IPs
- Real-time response monitoring
- Multiple attack modes (single, continuous, flood, stealth)
- Live statistics and response rate calculation

**Usage Examples:**
```bash
# Basic spoofing
sudo python3 icmp_spoofer.py 192.168.1.1

# High-speed flood attack
sudo python3 icmp_spoofer.py 192.168.1.1 --flood --duration 60

# Continuous attack mode
sudo python3 icmp_spoofer.py 192.168.1.1 --continuous

# Stealth scan
sudo python3 icmp_spoofer.py 192.168.1.1 --stealth
```

### 2. ICMP Redirect (`icmp_redirect.py`)

**Capabilities:**
- Routing table manipulation
- Real-time routing change detection
- Automatic traffic monitoring
- MITM setup assistance
- Attack verification

**Usage Examples:**
```bash
# Basic redirect attack
sudo python3 icmp_redirect.py 192.168.1.20 8.8.8.8 192.168.1.1

# Continuous redirect mode
sudo python3 icmp_redirect.py 192.168.1.20 8.8.8.8 192.168.1.1 --continuous

# Stealth redirect
sudo python3 icmp_redirect.py 192.168.1.20 8.8.8.8 192.168.1.1 --stealth

# MITM setup
sudo python3 icmp_redirect.py 192.168.1.20 8.8.8.8 192.168.1.1 --mitm
```

### 3. Main Tool (`icmp_attack_tool.py`)

**Interactive Menu Features:**
- Guided attack configuration
- Real-time attack monitoring
- Live network traffic analysis
- Attack statistics dashboard
- Integrated verification tools

## 📊 Visual Monitoring Features

### Live Attack Dashboard
```
============================================================
             LIVE ATTACK MONITOR
============================================================
📊 Attack Statistics:
   📤 Packets Sent: 1,247
   🔀 Redirects Sent: 23
   📥 Responses Received: 891
   🔄 Routing Changes: 3
⏰ Time: 14:23:45
============================================================
```

### Real-time Packet Analysis
- Live packet capture with type identification
- Source/destination analysis
- Response monitoring
- Traffic redirection detection

### Attack Verification
- Automatic success detection
- Routing table monitoring
- Attack effectiveness metrics
- Verification command generation

## 🔬 Advanced Features

### Continuous Attack Mode
- Sustained attacks with live monitoring
- Automatic attack adjustment
- Real-time statistics
- Interrupt-safe operation

### Stealth Capabilities
- Random timing patterns
- Multiple ICMP types
- Detection avoidance
- Low-profile operation

### Traffic Analysis
- Real-time packet inspection
- Protocol identification
- Flow monitoring
- Attack impact assessment

## ⚙️ Technical Details

### Requirements
- Linux operating system (optimized for Arch)
- Python 3.7+
- Root privileges for raw sockets
- Network access to targets

### Dependencies
- `scapy` - Packet manipulation
- `netifaces` - Network interface handling
- `colorama` - Terminal colors
- `faker` - Random IP generation

### Network Setup
```
Attacker: 192.168.1.10/24    (Your machine)
Victim:   192.168.1.20/24    (Target for redirects)
Target:   192.168.1.30/24    (Destination to redirect)
Gateway:  192.168.1.1/24     (Legitimate gateway)
```

## 🔍 Monitoring and Verification

### Real-time Monitoring
```bash
# Live network monitor
sudo python3 icmp_attack_tool.py --monitor

# Packet analysis
sudo tcpdump -i any icmp

# Routing table monitoring
watch -n 1 'ip route show'
```

### Attack Verification
```bash
# Check routing changes
ip route show | grep <target_ip>

# Trace route verification
traceroute <target_ip>

# Connectivity testing
ping -c 3 <target_ip>
```

## 🎭 Attack Scenarios

### Scenario 1: Network Reconnaissance
1. Use stealth ICMP scan to probe targets
2. Monitor responses and network behavior
3. Identify active hosts and routing

### Scenario 2: Traffic Redirection
1. Perform ICMP redirect attack
2. Monitor routing table changes
3. Verify traffic redirection
4. Analyze intercepted traffic

### Scenario 3: Man-in-the-Middle
1. Execute successful redirect attack
2. Set up traffic forwarding
3. Intercept and analyze traffic
4. Maintain persistent access

## 📈 Performance Features

- **High-speed packet generation** (1000+ packets/second)
- **Real-time monitoring** with minimal latency
- **Efficient memory usage** for sustained attacks
- **Multi-threaded operations** for concurrent monitoring
- **Interrupt-safe** attack termination

## 🛡️ Detection and Countermeasures

### Attack Signatures
- ICMP redirect packet analysis
- Routing table anomaly detection
- Traffic pattern analysis
- Source IP verification

### Defensive Measures
- ICMP redirect filtering
- Static routing table entries
- Network segmentation
- Traffic monitoring

## ⚖️ Legal and Ethical Usage

**IMPORTANT:** This tool is designed for:
- Authorized penetration testing
- Network security research
- Controlled lab environments
- Security professional training

**DO NOT USE FOR:**
- Unauthorized network attacks
- Malicious traffic interception
- Network disruption
- Any illegal activities

## 🔧 Troubleshooting

### Common Issues
1. **Permission denied**: Run with `sudo`
2. **No responses**: Check target connectivity
3. **Routing unchanged**: Verify network configuration
4. **High packet loss**: Check network capacity

### Debug Mode
```bash
# Enable verbose output
sudo python3 icmp_attack_tool.py --debug

# Packet analysis
sudo tcpdump -i any -vvv icmp
```

---

**⚠️ DISCLAIMER:** This tool is provided for authorized security testing only. Users are responsible for compliance with all applicable laws and regulations. 