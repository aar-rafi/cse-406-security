#!/bin/bash

# ICMP Attack GUI Launcher
# ========================

echo "ICMP Attack Demonstration GUI"
echo "=============================="
echo

# Check if running as root for network operations
if [[ $EUID -eq 0 ]]; then
    echo "⚠️  Running as root - GUI will have full network access"
else
    echo "ℹ️  Note: GUI will need sudo permissions for attacks"
fi

echo "Starting GUI..."
echo

# Launch the GUI
python3 icmp_attack_gui.py

echo "GUI closed." 