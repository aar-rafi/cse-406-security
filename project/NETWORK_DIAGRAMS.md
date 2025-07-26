# ICMP Attack Tool - Network Diagrams

This document contains visual diagrams using Mermaid to illustrate the network setup, attack flows, and defense mechanisms described in the technical guides.

## Table of Contents
1. [Network Namespace Topology](#network-namespace-topology)
2. [ICMP Redirect Attack Flow](#icmp-redirect-attack-flow)
3. [Packet Structure Diagrams](#packet-structure-diagrams)
4. [Defense Architecture](#defense-architecture)
5. [Attack Verification Process](#attack-verification-process)
6. [Kernel Protection Layers](#kernel-protection-layers)

---

## Network Namespace Topology

### Lab Environment Setup

```mermaid
graph TB
    subgraph "Host System"
        subgraph "Default Namespace"
            Host["Host192.168.100.1"]
            Bridge["Bridge br0192.168.100.1/24"]
        end
        
        subgraph "Victim Namespace"
            VictimHost["Victim192.168.100.2/24"]
            VVeth["v-veth interface"]
        end
        
        subgraph "Attacker Namespace" 
            AttackerHost["Attacker192.168.100.3/24"]
            SVeth["s-veth interface"]
        end
        
        subgraph "Virtual Interfaces"
            VHost["v-host"]
            SHost["s-host"]
        end
    end
    
    subgraph "Simulated External Network"
        TestNet1["Test Server 203.0.113.1"]
        TestNet2["Test Server 192.0.2.1"]
        Local1["Local Server 10.0.0.1"]
    end
    
    %% Connections
    VictimHost ---|"veth pair"| VVeth
    VVeth ---|"virtual link"| VHost
    VHost --- Bridge
    
    AttackerHost ---|"veth pair"| SVeth  
    SVeth ---|"virtual link"| SHost
    SHost --- Bridge
    
    Host --- Bridge
    Bridge ---|"simulated routing"| TestNet1
    Bridge ---|"simulated routing"| TestNet2  
    Bridge ---|"simulated routing"| Local1
    
    %% Styling
    classDef victim fill:#ffcccc,color:#000000
    classDef attacker fill:#ccffcc,color:#000000  
    classDef host fill:#ccccff,color:#000000
    classDef external fill:#ffffcc,color:#000000
    
    class VictimHost,VVeth victim
    class AttackerHost,SVeth attacker
    class Host,Bridge,VHost,SHost host
    class TestNet1,TestNet2,Local1 external
```

### Network Isolation Concept

```mermaid
graph TD
    subgraph "Process Isolation"
        P1["Process 1<br/>Default netns"]
        P2["Process 2<br/>Victim netns"]  
        P3["Process 3<br/>Attacker netns"]
    end
    
    subgraph "Network Stacks"
        NS1["Default Network Stack<br/>• Routing table<br/>• Interface list<br/>• Firewall rules"]
        NS2["Victim Network Stack<br/>• Independent routing<br/>• v-veth interface<br/>• Isolated rules"]
        NS3["Attacker Network Stack<br/>• Independent routing<br/>• s-veth interface<br/>• Attack capability"]
    end
    
    P1 --> NS1
    P2 --> NS2
    P3 --> NS3
    
    classDef process fill:#e1f5fe,color:#000000
    classDef stack fill:#f3e5f5,color:#000000
    
    class P1,P2,P3 process
    class NS1,NS2,NS3 stack
```

---

## ICMP Redirect Attack Flow

### Normal Routing vs Attack

```mermaid
sequenceDiagram
    participant V as Victim<br/>192.168.100.2
    participant G as Gateway<br/>192.168.100.1  
    participant A as Attacker<br/>192.168.100.3
    participant T as Test Server<br/>203.0.113.1
    
    Note over V,T: Normal Traffic Flow
    V->>G: 1. Packet to 203.0.113.1
    G->>T: 2. Forward packet
    T->>G: 3. Response
    G->>V: 4. Forward response
    
    Note over V,T: ICMP Redirect Attack
    A->>V: 5. ICMP Redirect<br/>"Use 192.168.100.3 for 203.0.113.1"
    Note over V: 6. Update routing table
    V->>A: 7. Future packets to 203.0.113.1
    A->>T: 8. Forward (if desired)
    T->>A: 9. Response
    A->>V: 10. Forward response
    
    Note over A: Attacker can now intercept,<br/>modify, or drop traffic
```

### Attack Packet Structure Flow

```mermaid
graph TB
    subgraph "Attacker Crafts Malicious Packet"
        A1["1. Create fake inner IP packet"]
        A2["2. Craft ICMP redirect Type 5 Code 1"]
        A3["3. Create outer IP header"]
        A4["4. Calculate checksums"]
    end
    
    subgraph "Packet Transmission"
        T1["5. Send via raw socket"]
        T2["6. Packet appears from gateway"]
    end
    
    subgraph "Victim Processing"  
        V1["7. Receive ICMP redirect"]
        V2["8. Validate source"]
        V3["9. Update routing table"]
        V4["10. Future traffic redirected"]
    end
    
    A1 --> A2 --> A3 --> A4
    A4 --> T1 --> T2
    T2 --> V1 --> V2 --> V3 --> V4
    
    classDef attacker fill:#ffcccc,color:#000000
    classDef transmission fill:#ccffcc,color:#000000
    classDef victim fill:#ccccff,color:#000000
    
    class A1,A2,A3,A4 attacker
    class T1,T2 transmission
    class V1,V2,V3,V4 victim
```

---

## Packet Structure Diagrams

### IP Header Structure (20 bytes)

```mermaid
graph TD
    subgraph "IP Header Fields"
        subgraph "Byte 0-3"
            Ver["Version4 bits(IPv4 = 4)"]
            IHL["IHL4 bits(5 = 20 bytes)"]
            TOS["Type of Service8 bits(usually 0)"]
            Len["Total Length16 bits(IP + payload)"]
        end
        
        subgraph "Byte 4-7"
            ID["Identification16 bits(packet ID)"]
            Flags["Flags3 bits(fragmentation)"]
            Frag["Fragment Offset13 bits(position)"]
        end
        
        subgraph "Byte 8-11"
            TTL["TTL8 bits(hop limit)"]
            Proto["Protocol8 bits(1 = ICMP)"]
            Chk["Header Checksum16 bits(error detection)"]
        end
        
        subgraph "Byte 12-19"
            Src["Source Address32 bits(source IP)"]
            Dst["Destination Address32 bits(destination IP)"]
        end
    end
    
    subgraph "struct.pack Format"
        Format["'!BBHHHBBH4s4s'• ! = network byte order• B = 1 byte unsigned• H = 2 byte unsigned• 4s = 4 byte string"]
    end
    
    classDef header fill:#e3f2fd,color:#000000
    classDef format fill:#f1f8e9,color:#000000
    
    class Ver,IHL,TOS,Len,ID,Flags,Frag,TTL,Proto,Chk,Src,Dst header
    class Format format
```

### ICMP Packet Types

```mermaid
graph TB
    subgraph "ICMP Echo (Ping) - Type 8/0"
        E1["Type: 8 (Request) / 0 (Reply)Code: 0Checksum: calculatedIdentifier: process IDSequence: packet numberPayload: optional data"]
    end
    
    subgraph "ICMP Redirect - Type 5"  
        R1["Type: 5 (Redirect)Code: 1 (Host redirect)Checksum: calculatedGateway: new gateway IPOriginal packet: IP header + 8 bytes"]
    end
    
    subgraph "struct.pack Formats"
        EF["Echo: '!BBHHH'5 fields, 8 bytes total"]
        RF["Redirect: '!BBH4s'4 fields, 8 bytes + data"]
    end
    
    classDef echo fill:#e8f5e8,color:#000000
    classDef redirect fill:#fff3e0,color:#000000
    classDef format fill:#f3e5f5,color:#000000
    
    class E1 echo
    class R1 redirect
    class EF,RF format
```

---

## Defense Architecture

### Multi-Layer Defense Strategy

```mermaid
graph TD
    subgraph "Layer 1: Network Perimeter"
        FW["Firewall• Block ICMP redirects• Rate limit ICMP• Source validation"]
        IPS["IPS• Detect attack patterns• Block malicious sources• Alert on anomalies"]
    end
    
    subgraph "Layer 2: Network Segmentation"
        VLAN["VLANs• Isolate critical systems• Control inter-VLAN routing• Micro-segmentation"]
        ACL["ACLs• Granular access control• Protocol filtering• Source/destination rules"]
    end
    
    subgraph "Layer 3: Host Hardening"
        Kernel["Kernel Parameters• accept_redirects=0• secure_redirects=1• rp_filter=1"]
        Services["Service Configuration• Minimal ICMP responses• Secure defaults• Regular updates"]
    end
    
    subgraph "Layer 4: Monitoring & Detection"
        SIEM["SIEM• Log correlation• Attack detection• Automated response"]
        Monitor["Network Monitoring• Route change detection• Traffic analysis• Real-time alerting"]
    end
    
    Attack["🔴 ICMP Redirect Attack"] --> FW
    FW --> IPS
    IPS --> VLAN
    VLAN --> ACL
    ACL --> Kernel
    Kernel --> Services
    Services --> SIEM
    SIEM --> Monitor
    
    classDef perimeter fill:#ffebee,color:#000000
    classDef segmentation fill:#e8f5e8,color:#000000  
    classDef hardening fill:#e3f2fd,color:#000000
    classDef monitoring fill:#fff3e0,color:#000000
    
    class FW,IPS perimeter
    class VLAN,ACL segmentation
    class Kernel,Services hardening
    class SIEM,Monitor monitoring
```

### Kernel Protection Mechanisms

```mermaid
flowchart TD
    Packet["📦 ICMP Redirect Packet"]
    
    subgraph "Kernel Processing"
        Check1{{"accept_redirectsenabled?"}}
        Check2{{"secure_redirectsenabled?"}}
        Check3{{"Source ingateway list?"}}
        Check4{{"Valid redirectformat?"}}
        Check5{{"Rate limitexceeded?"}}
    end
    
    subgraph "Actions"
        Accept["✅ Accept RedirectUpdate routing table"]
        Drop1["❌ Drop - Redirects disabled"]
        Drop2["❌ Drop - Untrusted source"]
        Drop3["❌ Drop - Invalid format"]
        Drop4["❌ Drop - Rate limited"]
    end
    
    Packet --> Check1
    Check1 -->|No| Drop1
    Check1 -->|Yes| Check2
    Check2 -->|Yes| Check3
    Check2 -->|No| Check4
    Check3 -->|No| Drop2
    Check3 -->|Yes| Check4
    Check4 -->|No| Drop3
    Check4 -->|Yes| Check5
    Check5 -->|Yes| Drop4
    Check5 -->|No| Accept
    
    classDef packet fill:#e1f5fe,color:#000000
    classDef check fill:#fff3e0,color:#000000
    classDef accept fill:#e8f5e8,color:#000000
    classDef drop fill:#ffebee,color:#000000
    
    class Packet packet
    class Check1,Check2,Check3,Check4,Check5 check
    class Accept accept
    class Drop1,Drop2,Drop3,Drop4 drop
```

---

## Attack Verification Process

### Multi-Terminal Monitoring Setup

```mermaid
graph TB
    subgraph "Terminal 1: Route Monitoring"
        T1["watch -n 1 'ip route get 203.0.113.1'📊 Continuous route tracking🔍 Detect table changes"]
    end
    
    subgraph "Terminal 2: Attacker Traffic"
        T2["tcpdump -i s-veth icmp -v📡 Monitor outgoing attacks✅ Verify packet transmission"]
    end
    
    subgraph "Terminal 3: Attack Execution"
        T3["python3 icmp_redirect_raw.py🚀 Execute redirect attack📈 Live statistics display"]
    end
    
    subgraph "Terminal 4: Victim Testing"
        T4["ping 203.0.113.1traceroute 203.0.113.1🎯 Test connectivity changes"]
    end
    
    subgraph "Terminal 5: Victim Traffic"
        T5["tcpdump -i v-veth👁️ Monitor incoming/outgoing🔄 Verify redirection"]
    end
    
    subgraph "Attack Results"
        Success["✅ Attack Success Indicators• Route table updated• Traffic redirected• Path through attacker"]
        Failure["❌ Attack Failure Indicators• No route changes• Direct path maintained• Security controls active"]
    end
    
    T1 --> Success
    T2 --> Success
    T3 --> Success
    T4 --> Success
    T5 --> Success
    
    T1 --> Failure
    T2 --> Failure
    T3 --> Failure
    T4 --> Failure
    T5 --> Failure
    
    classDef terminal fill:#e3f2fd,color:#000000
    classDef success fill:#e8f5e8,color:#000000
    classDef failure fill:#ffebee,color:#000000
    
    class T1,T2,T3,T4,T5 terminal
    class Success success
    class Failure failure
```

### Attack Success Timeline

```mermaid
gantt
    title ICMP Redirect Attack Timeline
    dateFormat X
    axisFormat %Xs
    
    section Preparation
    Setup namespaces           :done, prep1, 0, 5s
    Configure interfaces       :done, prep2, 5s, 10s
    Start monitoring           :done, prep3, 10s, 15s
    
    section Attack Execution
    Send ICMP redirect         :active, attack1, 15s, 17s
    Kernel processes packet    :attack2, 17s, 18s
    Route table update         :attack3, 18s, 20s
    
    section Verification
    Monitor route changes      :verify1, 20s, 60s
    Test connectivity          :verify2, 25s, 35s
    Capture redirected traffic :verify3, 30s, 60s
    
    section Analysis
    Generate attack report     :analysis1, 60s, 70s
    Document results           :analysis2, 65s, 75s
```

---

## Kernel Protection Layers

### sysctl Parameter Hierarchy

```mermaid
graph TD
    subgraph "Global Settings"
        Global["accept_redirects secure_redirects send_redirects rp_filter"]
    end
    
    subgraph "Default Settings"
        Default["New interface defaults Applied to new interfaces"]
    end
    
    subgraph "Interface-Specific Settings"
        Eth0["eth0 Physical interface"]
        Veth["v-veth Virtual interface"]
        Lo["lo Loopback interface"]
    end
    
    subgraph "Priority Order"
        Priority["1. Interface-specific highest 2. Global settings 3. Default settings lowest"]
    end
    
    Global -.-> Eth0
    Global -.-> Veth  
    Global -.-> Lo
    Default -.-> Eth0
    Default -.-> Veth
    Default -.-> Lo
    
    Priority --> Global
    Priority --> Default
    
    classDef global fill:#ffcccc,color:#000000
    classDef default fill:#ccffcc,color:#000000
    classDef interface fill:#ccccff,color:#000000
    classDef priority fill:#ffffcc,color:#000000
    
    class Global global
    class Default default
    class Eth0,Veth,Lo interface
    class Priority priority
```

### Lab vs Production Configuration

```mermaid
graph LR
    subgraph "Lab Environment (Attack Enabled)"
        Lab["🧪 Lab Settingsaccept_redirects = 1secure_redirects = 0send_redirects = 1rp_filter = 0"]
        LabResult["📊 Lab Results✅ Attacks succeed✅ Route manipulation✅ Traffic redirection✅ Educational value"]
    end
    
    subgraph "Production Environment (Secure)"
        Prod["🔒 Production Settingsaccept_redirects = 0secure_redirects = 1send_redirects = 0rp_filter = 1"]
        ProdResult["🛡️ Production Results❌ Attacks blocked❌ No route changes❌ Traffic protected✅ Security maintained"]
    end
    
    subgraph "Migration Process"
        Migration["📋 Migration Checklist1. Verify kernel parameters2. Configure firewalls3. Enable monitoring4. Test functionality5. Document changes6. Train team"]
    end
    
    Lab --> Migration
    Migration --> Prod
    Lab --> LabResult
    Prod --> ProdResult
    
    classDef lab fill:#fff3e0,color:#000000
    classDef prod fill:#e8f5e8,color:#000000
    classDef migration fill:#e3f2fd,color:#000000
    classDef result fill:#f3e5f5,color:#000000
    
    class Lab lab
    class Prod prod
    class Migration migration
    class LabResult,ProdResult result
```

---

## Summary

These diagrams illustrate the complete ICMP attack ecosystem:

1. **Network Topology**: Shows the namespace isolation and virtual networking setup
2. **Attack Flow**: Demonstrates how ICMP redirect attacks manipulate routing
3. **Packet Structure**: Details the binary packet construction process
4. **Defense Architecture**: Presents multi-layered security approach
5. **Verification Process**: Outlines the monitoring and testing procedures
6. **Kernel Protections**: Explains the built-in Linux security mechanisms

The visual representations make it easier to understand the complex interactions between network namespaces, kernel parameters, packet structures, and defense mechanisms described in the technical documentation. 