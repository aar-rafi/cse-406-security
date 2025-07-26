#!/bin/bash
# === 1. Host bridge ===
ip link add br0 type bridge
ip addr add 192.168.100.1/24 dev br0
ip link set br0 up
echo 1 > /proc/sys/net/ipv4/ip_forward

# === 2. Victim namespace ===
ip netns add victim
ip link add v-veth type veth peer name v-veth-br
ip link set v-veth   netns victim
ip link set v-veth-br master br0
ip link set v-veth-br up

ip netns exec victim ip link set lo up
ip netns exec victim ip link set v-veth up
ip netns exec victim ip addr add 192.168.100.2/24 dev v-veth
ip netns exec victim ip route add default via 192.168.100.1
ip netns exec victim sysctl -w net.ipv4.conf.all.rp_filter=0

# === 3. Spoofed namespace ===
ip netns add spoofed
ip link add s-veth type veth peer name s-veth-br
ip link set s-veth   netns spoofed
ip link set s-veth-br master br0
ip link set s-veth-br up

ip netns exec spoofed ip link set lo up
ip netns exec spoofed ip link set s-veth up
ip netns exec spoofed ip addr add 192.168.100.3/24 dev s-veth
ip netns exec spoofed ip route add default via 192.168.100.1
ip netns exec spoofed sysctl -w net.ipv4.conf.all.rp_filter=0