#!/bin/bash

# ICMP Attack Tool Setup Script for Arch Linux
# =============================================
# 
# This script installs all necessary dependencies and configures
# the system for the ICMP attack educational tool.
#
# Educational use only - requires proper authorization.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${CYAN}================================="
echo "ICMP Attack Tool Setup for Arch"
echo "Educational Security Tool"
echo -e "=================================${NC}"
echo ""
echo -e "${YELLOW}⚠️  WARNING: Educational use only!${NC}"
echo -e "${YELLOW}⚠️  Use only in authorized environments!${NC}"
echo ""

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}❌ This script should NOT be run as root (except for specific parts)"
   echo -e "Please run as a regular user. Sudo will be used when needed.${NC}"
   exit 1
fi

# Function to print status
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on Arch Linux
check_arch() {
    if ! command -v pacman &> /dev/null; then
        print_error "This script is designed for Arch Linux systems only!"
        exit 1
    fi
    print_success "Arch Linux detected"
}

# Update system
update_system() {
    print_status "Updating system packages..."
    sudo pacman -Syu --noconfirm
    print_success "System updated"
}

# Install Python and pip
install_python() {
    print_status "Installing Python and pip..."
    
    # Check if python is already installed
    if command -v python3 &> /dev/null; then
        print_warning "Python3 already installed"
        python3 --version
    else
        sudo pacman -S python python-pip --noconfirm
        print_success "Python and pip installed"
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Required packages for the tool
    packages=(
        "python-pip"
        "python-virtualenv" 
        "net-tools"        # For netstat, route commands
        "iproute2"         # For ip command  
        "tcpdump"          # For packet capture
        "wireshark-cli"    # For packet analysis
        "nmap"             # For network scanning
        "bind-tools"       # For dig, nslookup
    )
    
    for package in "${packages[@]}"; do
        if pacman -Qi "$package" &> /dev/null; then
            print_warning "$package already installed"
        else
            print_status "Installing $package..."
            sudo pacman -S "$package" --noconfirm
            print_success "$package installed"
        fi
    done
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Check if requirements.txt exists
    if [[ ! -f "requirements.txt" ]]; then
        print_error "requirements.txt not found in current directory!"
        print_status "Creating requirements.txt..."
        cat > requirements.txt << EOF
scapy>=2.4.3
netifaces>=0.11.0
colorama>=0.4.4
faker>=15.0.0
argparse
ipaddress
socket
EOF
        print_success "requirements.txt created"
    fi
    
    # Install Python packages
    pip install --user -r requirements.txt
    print_success "Python dependencies installed"
}

# Set up file permissions
setup_permissions() {
    print_status "Setting up file permissions..."
    
    # Make Python scripts executable
    scripts=("icmp_attack_tool.py" "icmp_spoofer.py" "icmp_redirect.py")
    
    for script in "${scripts[@]}"; do
        if [[ -f "$script" ]]; then
            chmod +x "$script"
            print_success "$script made executable"
        else
            print_warning "$script not found"
        fi
    done
}

# Configure network capabilities (optional)
configure_capabilities() {
    print_status "Configuring network capabilities..."
    
    echo "This will allow the Python scripts to create raw sockets without sudo."
    echo "This is optional but convenient for development."
    read -p "Do you want to configure capabilities? (y/N): " response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        # Find Python executable
        PYTHON_PATH=$(which python3)
        
        if [[ -n "$PYTHON_PATH" ]]; then
            print_status "Setting capabilities on $PYTHON_PATH..."
            sudo setcap cap_net_raw+ep "$PYTHON_PATH"
            print_success "Capabilities set on Python binary"
            print_warning "Note: This affects the system Python binary"
        else
            print_error "Python3 not found in PATH"
        fi
    else
        print_status "Skipping capability configuration"
        print_warning "You'll need to run scripts with sudo"
    fi
}

# Create demo scripts
create_demos() {
    print_status "Creating demonstration scripts..."
    
    # Create ICMP spoofing demo
    cat > demo_spoof.sh << 'EOF'
#!/bin/bash
echo "=== ICMP Spoofing Demo ==="
echo "This script demonstrates ICMP spoofing attacks"
echo ""
echo "Usage examples:"
echo "  sudo python3 icmp_spoofer.py --explain"
echo "  sudo python3 icmp_spoofer.py 192.168.1.1"
echo "  sudo python3 icmp_spoofer.py 192.168.1.1 --source 10.0.0.1 --count 3"
echo ""
echo "Run with --help for full options"
EOF
    chmod +x demo_spoof.sh
    
    # Create ICMP redirect demo
    cat > demo_redirect.sh << 'EOF'
#!/bin/bash
echo "=== ICMP Redirect Demo ==="
echo "This script demonstrates ICMP redirect attacks"
echo ""
echo "Usage examples:"
echo "  sudo python3 icmp_redirect.py --explain"
echo "  sudo python3 icmp_redirect.py 192.168.1.10 8.8.8.8 192.168.1.1"
echo "  sudo python3 icmp_redirect.py 192.168.1.10 8.8.8.8 192.168.1.1 --fake-gateway 192.168.1.100"
echo ""
echo "Run with --help for full options"
EOF
    chmod +x demo_redirect.sh
    
    print_success "Demo scripts created"
}

# Create lab setup guide
create_lab_guide() {
    print_status "Creating lab setup guide..."
    
    cat > LAB_SETUP.md << 'EOF'
# Lab Setup Guide

## Quick Test Environment

### Single Machine Testing
For basic functionality testing on a single machine:

```bash
# Test ICMP spoofing (educational mode)
sudo python3 icmp_spoofer.py --explain

# Test with localhost
sudo python3 icmp_spoofer.py 127.0.0.1 --count 1

# Test ICMP redirect explanation
sudo python3 icmp_redirect.py --explain
```

### VM Lab Environment

#### Setup
1. Create 3 VMs:
   - **Attacker**: Arch Linux (this machine)
   - **Victim**: Any Linux distribution  
   - **Target**: Any reachable machine/service

2. Network Configuration:
   ```
   Attacker: 192.168.1.10/24
   Victim:   192.168.1.20/24
   Gateway:  192.168.1.1/24
   Target:   192.168.1.30/24 (or external like 8.8.8.8)
   ```

#### Testing ICMP Spoofing
```bash
# From attacker machine
sudo python3 icmp_spoofer.py 192.168.1.20 --source 192.168.1.100

# Monitor on victim machine
sudo tcpdump -i any icmp
```

#### Testing ICMP Redirect
```bash
# From attacker machine
sudo python3 icmp_redirect.py 192.168.1.20 8.8.8.8 192.168.1.1 --fake-gateway 192.168.1.10

# Check routing on victim machine
ip route show
```

### Verification Commands

On victim machine:
```bash
# Monitor routing table
watch -n 1 'ip route show'

# Capture ICMP traffic
sudo tcpdump -i any -n icmp

# View network interfaces
ip addr show
```

On attacker machine:
```bash
# Monitor network traffic
sudo tcpdump -i any -n 'icmp or (host 192.168.1.20)'

# Check IP forwarding
cat /proc/sys/net/ipv4/ip_forward
```

## Security Notes

- Always test in isolated environments
- Obtain proper authorization before testing
- Understand legal implications
- Use only for educational purposes
- Monitor and analyze traffic ethically

## Troubleshooting

1. **Permission Errors**: Run with sudo
2. **No Response**: Check firewalls and routing
3. **Module Errors**: Verify Python dependencies
4. **Network Issues**: Confirm connectivity between machines
EOF

    print_success "Lab setup guide created (LAB_SETUP.md)"
}

# Check system compatibility
check_compatibility() {
    print_status "Checking system compatibility..."
    
    # Check kernel version
    kernel_version=$(uname -r)
    print_status "Kernel version: $kernel_version"
    
    # Check if raw sockets are available
    if [[ -e /proc/sys/net/ipv4/ip_forward ]]; then
        print_success "IP forwarding support available"
    else
        print_warning "IP forwarding support not found"
    fi
    
    # Check network interfaces
    interfaces=$(ip link show | grep -E '^[0-9]+:' | wc -l)
    print_status "Network interfaces found: $interfaces"
    
    print_success "System compatibility check complete"
}

# Main installation flow
main() {
    print_status "Starting ICMP Attack Tool setup for Arch Linux..."
    echo ""
    
    # Step 1: System checks
    check_arch
    check_compatibility
    echo ""
    
    # Step 2: Update system
    read -p "Update system packages? (recommended) (y/N): " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        update_system
    fi
    echo ""
    
    # Step 3: Install dependencies
    install_python
    install_system_deps
    install_python_deps
    echo ""
    
    # Step 4: Setup
    setup_permissions
    configure_capabilities
    echo ""
    
    # Step 5: Create helpers
    create_demos
    create_lab_guide
    echo ""
    
    # Final message
    print_success "Setup completed successfully!"
    echo ""
    echo -e "${CYAN}📚 Next Steps:${NC}"
    echo "  1. Read LAB_SETUP.md for testing guidance"
    echo "  2. Run: sudo python3 icmp_attack_tool.py --educational"
    echo "  3. Try: ./demo_spoof.sh and ./demo_redirect.sh"
    echo ""
    echo -e "${YELLOW}⚠️  Remember: Use only for authorized testing and education!${NC}"
    echo ""
    echo -e "${GREEN}🎓 Happy learning!${NC}"
}

# Run main function
main "$@" 