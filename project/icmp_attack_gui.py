#!/usr/bin/env python3
"""
ICMP Attack Demonstration GUI
============================

A comprehensive GUI for demonstrating ICMP spoofing and redirect attacks
with real-time monitoring and visual feedback.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import threading
import time
import queue
import json
import os
from datetime import datetime
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Import our lab configuration
try:
    from lab_config import get_config, get_attack_scenarios
except ImportError:
    # Fallback configuration
    def get_config():
        return {
            'victim_namespace': 'victim',
            'attacker_namespace': 'spoofed',
            'victim_ip': '192.168.100.2',
            'attacker_ip': '192.168.100.3',
            'gateway_ip': '192.168.100.1',
            'network_subnet': '192.168.100.0/24',
        }
    
    def get_attack_scenarios():
        return {}

class AttackMonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ICMP Attack Demonstration Lab")
        self.root.geometry("1200x800")
        
        # Configuration
        self.config = get_config()
        self.scenarios = get_attack_scenarios()
        
        # Attack state
        self.attack_running = False
        self.attack_process = None
        self.monitor_threads = []
        self.output_queues = {
            'victim_tcpdump': queue.Queue(),
            'attacker_tcpdump': queue.Queue(),
            'victim_routing': queue.Queue(),
            'attack_output': queue.Queue(),
            'status': queue.Queue()
        }
        
        # Statistics
        self.stats = {
            'packets_sent': 0,
            'redirects_sent': 0,
            'route_changes': 0,
            'attack_start_time': None
        }
        
        self.create_widgets()
        self.start_monitoring()
        
    def create_widgets(self):
        """Create the main GUI widgets"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_topology_tab()
        self.create_config_tab()
        self.create_monitoring_tab()
        self.create_results_tab()
        
        # Status bar
        self.create_status_bar()
        
    def create_topology_tab(self):
        """Create network topology visualization tab"""
        self.topology_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.topology_frame, text="Network Topology")
        
        # Title
        title = ttk.Label(self.topology_frame, text="Network Lab Topology", 
                         font=('Arial', 16, 'bold'))
        title.pack(pady=10)
        
        if MATPLOTLIB_AVAILABLE:
            self.create_network_diagram()
        else:
            self.create_text_topology()
    
    def create_network_diagram(self):
        """Create visual network diagram using matplotlib"""
        fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = fig.add_subplot(111)
        
        # Draw network components
        self.draw_network_topology()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.topology_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=20)
        
    def draw_network_topology(self):
        """Draw the network topology diagram"""
        self.ax.clear()
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 6)
        self.ax.set_aspect('equal')
        
        # Colors
        victim_color = '#FF6B6B'
        attacker_color = '#4ECDC4'
        gateway_color = '#45B7D1'
        bridge_color = '#96CEB4'
        
        # Draw bridge
        bridge = patches.Rectangle((4, 2.5), 2, 1, 
                                 facecolor=bridge_color, edgecolor='black', linewidth=2)
        self.ax.add_patch(bridge)
        self.ax.text(5, 3, 'Bridge\nbr0', ha='center', va='center', fontweight='bold')
        
        # Draw victim namespace
        victim = patches.Circle((2, 4), 0.8, facecolor=victim_color, edgecolor='black', linewidth=2)
        self.ax.add_patch(victim)
        self.ax.text(2, 4, f'Victim\n{self.config["victim_ip"]}', ha='center', va='center', fontweight='bold')
        
        # Draw attacker namespace  
        attacker = patches.Circle((8, 4), 0.8, facecolor=attacker_color, edgecolor='black', linewidth=2)
        self.ax.add_patch(attacker)
        self.ax.text(8, 4, f'Attacker\n{self.config["attacker_ip"]}', ha='center', va='center', fontweight='bold')
        
        # Draw gateway
        gateway = patches.Circle((5, 1), 0.8, facecolor=gateway_color, edgecolor='black', linewidth=2)
        self.ax.add_patch(gateway)
        self.ax.text(5, 1, f'Gateway\n{self.config["gateway_ip"]}', ha='center', va='center', fontweight='bold')
        
        # Draw connections
        # Victim to bridge
        self.ax.plot([2.8, 4], [4, 3.5], 'k-', linewidth=2)
        self.ax.text(3.2, 3.8, 'v-veth', fontsize=8, ha='center')
        
        # Attacker to bridge
        self.ax.plot([7.2, 6], [4, 3.5], 'k-', linewidth=2)
        self.ax.text(6.8, 3.8, 's-veth', fontsize=8, ha='center')
        
        # Bridge to gateway
        self.ax.plot([5, 5], [2.5, 1.8], 'k-', linewidth=2)
        
        # Add attack indicators
        if self.attack_running:
            # Draw attack arrow
            self.ax.annotate('', xy=(7.2, 4), xytext=(2.8, 4),
                           arrowprops=dict(arrowstyle='->', color='red', lw=3))
            self.ax.text(5, 4.5, 'ATTACK IN PROGRESS', ha='center', va='center',
                        fontsize=12, fontweight='bold', color='red')
        
        self.ax.set_title('Namespace Lab Network Topology', fontsize=14, fontweight='bold')
        self.ax.axis('off')
        
    def create_text_topology(self):
        """Create text-based topology view if matplotlib unavailable"""
        topology_text = f"""
        Network Lab Topology
        ==================
        
        Victim Namespace: {self.config['victim_namespace']} ({self.config['victim_ip']})
        ├─ Interface: v-veth
        └─ Connected to: br0 bridge
        
        Attacker Namespace: {self.config['attacker_namespace']} ({self.config['attacker_ip']})
        ├─ Interface: s-veth  
        └─ Connected to: br0 bridge
        
        Gateway: {self.config['gateway_ip']}
        └─ Connected to: br0 bridge
        
        Network: {self.config['network_subnet']}
        
        Attack Flow:
        {self.config['attacker_namespace']} → {self.config['victim_namespace']}
        Spoofed packets with fake source IPs
        ICMP redirects to manipulate routing
        """
        
        text_widget = scrolledtext.ScrolledText(self.topology_frame, wrap=tk.WORD)
        text_widget.pack(fill='both', expand=True, padx=20, pady=20)
        text_widget.insert('1.0', topology_text)
        text_widget.config(state='disabled')
    
    def create_config_tab(self):
        """Create attack configuration tab"""
        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text="Attack Configuration")
        
        # Configuration panel
        config_panel = ttk.LabelFrame(self.config_frame, text="Attack Configuration", padding=10)
        config_panel.pack(fill='x', padx=20, pady=10)
        
        # Scenario selection
        ttk.Label(config_panel, text="Attack Scenario:").grid(row=0, column=0, sticky='w', pady=5)
        self.scenario_var = tk.StringVar()
        scenario_combo = ttk.Combobox(config_panel, textvariable=self.scenario_var, width=30)
        scenario_combo['values'] = [f"{k}: {v['name']}" for k, v in self.scenarios.items()]
        scenario_combo.grid(row=0, column=1, sticky='ew', padx=(10, 0), pady=5)
        scenario_combo.bind('<<ComboboxSelected>>', self.on_scenario_change)
        
        # Attack type
        ttk.Label(config_panel, text="Attack Type:").grid(row=1, column=0, sticky='w', pady=5)
        self.attack_type = tk.StringVar(value="spoof")
        type_frame = ttk.Frame(config_panel)
        type_frame.grid(row=1, column=1, sticky='ew', padx=(10, 0), pady=5)
        ttk.Radiobutton(type_frame, text="ICMP Spoofing", variable=self.attack_type, 
                       value="spoof").pack(side='left', padx=5)
        ttk.Radiobutton(type_frame, text="ICMP Redirect", variable=self.attack_type, 
                       value="redirect").pack(side='left', padx=5)
        
        # Parameters
        params_frame = ttk.LabelFrame(config_panel, text="Parameters", padding=5)
        params_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=10)
        
        # Target IP
        ttk.Label(params_frame, text="Target IP:").grid(row=0, column=0, sticky='w', pady=2)
        self.target_ip = tk.StringVar(value=self.config['victim_ip'])
        ttk.Entry(params_frame, textvariable=self.target_ip, width=15).grid(row=0, column=1, padx=5, pady=2)
        
        # Source IP (for spoofing)
        ttk.Label(params_frame, text="Source IP:").grid(row=0, column=2, sticky='w', pady=2)
        self.source_ip = tk.StringVar(value=self.config['attacker_ip'])
        ttk.Entry(params_frame, textvariable=self.source_ip, width=15).grid(row=0, column=3, padx=5, pady=2)
        
        # Duration
        ttk.Label(params_frame, text="Duration (s):").grid(row=1, column=0, sticky='w', pady=2)
        self.duration = tk.StringVar(value="30")
        ttk.Entry(params_frame, textvariable=self.duration, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        # Attack mode
        ttk.Label(params_frame, text="Mode:").grid(row=1, column=2, sticky='w', pady=2)
        self.attack_mode = tk.StringVar(value="single")
        mode_combo = ttk.Combobox(params_frame, textvariable=self.attack_mode, width=12)
        mode_combo['values'] = ['single', 'flood', 'stealth']
        mode_combo.grid(row=1, column=3, padx=5, pady=2)
        
        # Control buttons
        control_frame = ttk.Frame(self.config_frame)
        control_frame.pack(fill='x', padx=20, pady=10)
        
        self.start_button = ttk.Button(control_frame, text="Start Attack", 
                                     command=self.start_attack, style='success.TButton')
        self.start_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Attack", 
                                    command=self.stop_attack, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Check Namespaces", 
                  command=self.check_namespaces).pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Setup Lab", 
                  command=self.setup_lab).pack(side='left', padx=5)
        
        # Help section
        help_frame = ttk.LabelFrame(self.config_frame, text="Attack Mode Guide", padding=5)
        help_frame.pack(fill='x', padx=20, pady=5)
        
        help_text = """Attack Modes:
• Single: Send one packet and stop
• Flood: Send packets continuously (for spoofing) / continuous redirects (for redirect)
• Stealth: Send random packets with delays (spoofing only)

Attack Types:
• ICMP Spoofing: Send ICMP packets with fake source IP
• ICMP Redirect: Send ICMP redirect messages to change victim's routing table"""
        
        help_label = ttk.Label(help_frame, text=help_text, font=('Arial', 9), justify='left')
        help_label.pack(anchor='w')
        
        # Configuration display
        self.config_display = scrolledtext.ScrolledText(self.config_frame, height=12, wrap=tk.WORD)
        self.config_display.pack(fill='both', expand=True, padx=20, pady=10)
        self.update_config_display()
        
    def create_monitoring_tab(self):
        """Create real-time monitoring tab"""
        self.monitoring_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.monitoring_frame, text="Live Monitoring")
        
        # Create paned window for multiple monitoring areas
        paned = ttk.PanedWindow(self.monitoring_frame, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel - Network monitoring
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        # Victim namespace monitoring
        victim_frame = ttk.LabelFrame(left_frame, text="Victim Namespace", padding=5)
        victim_frame.pack(fill='both', expand=True, pady=5)
        
        # Victim tcpdump
        ttk.Label(victim_frame, text="Network Traffic:").pack(anchor='w')
        self.victim_tcpdump = scrolledtext.ScrolledText(victim_frame, height=8, width=50)
        self.victim_tcpdump.pack(fill='both', expand=True, pady=2)
        
        # Victim routing
        ttk.Label(victim_frame, text="Routing Table:").pack(anchor='w')
        self.victim_routing = scrolledtext.ScrolledText(victim_frame, height=4, width=50)
        self.victim_routing.pack(fill='both', expand=True, pady=2)
        
        # Attacker namespace monitoring
        attacker_net_frame = ttk.LabelFrame(left_frame, text="Attacker Namespace", padding=5)
        attacker_net_frame.pack(fill='both', expand=True, pady=5)
        
        # Attacker tcpdump
        ttk.Label(attacker_net_frame, text="Network Traffic:").pack(anchor='w')
        self.attacker_tcpdump = scrolledtext.ScrolledText(attacker_net_frame, height=6, width=50)
        self.attacker_tcpdump.pack(fill='both', expand=True, pady=2)
        
        # Right panel - Attack monitoring
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        # Attack status
        status_frame = ttk.LabelFrame(right_frame, text="Attack Status", padding=5)
        status_frame.pack(fill='x', pady=5)
        
        self.status_labels = {}
        status_items = [
            ('Status:', 'Idle'),
            ('Packets Sent:', '0'),
            ('Redirects Sent:', '0'),
            ('Route Changes:', '0'),
            ('Duration:', '00:00'),
        ]
        
        for i, (label, value) in enumerate(status_items):
            ttk.Label(status_frame, text=label).grid(row=i, column=0, sticky='w', pady=2)
            self.status_labels[label] = ttk.Label(status_frame, text=value, font=('Arial', 10, 'bold'))
            self.status_labels[label].grid(row=i, column=1, sticky='w', padx=(10, 0), pady=2)
        
        # Attacker namespace monitoring
        attacker_frame = ttk.LabelFrame(right_frame, text="Attacker Namespace", padding=5)
        attacker_frame.pack(fill='both', expand=True, pady=5)
        
        ttk.Label(attacker_frame, text="Attack Output:").pack(anchor='w')
        self.attack_output = scrolledtext.ScrolledText(attacker_frame, height=15, width=50)
        self.attack_output.pack(fill='both', expand=True, pady=2)
        
    def create_results_tab(self):
        """Create results and analysis tab"""
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results & Analysis")
        
        # Results display
        results_text = ttk.LabelFrame(self.results_frame, text="Attack Results", padding=10)
        results_text.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.results_display = scrolledtext.ScrolledText(results_text, wrap=tk.WORD)
        self.results_display.pack(fill='both', expand=True)
        
        # Control buttons
        button_frame = ttk.Frame(self.results_frame)
        button_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Button(button_frame, text="Save Results", 
                  command=self.save_results).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Clear Results", 
                  command=self.clear_results).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Export Report", 
                  command=self.export_report).pack(side='left', padx=5)
        
    def create_status_bar(self):
        """Create status bar at bottom"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side='bottom', fill='x')
        
        self.status_text = ttk.Label(self.status_bar, text="Ready")
        self.status_text.pack(side='left', padx=10, pady=5)
        
        # Connection indicators
        indicator_frame = ttk.Frame(self.status_bar)
        indicator_frame.pack(side='right', padx=10, pady=5)
        
        ttk.Label(indicator_frame, text="Victim:").pack(side='left', padx=2)
        self.victim_indicator = ttk.Label(indicator_frame, text="●", foreground='red')
        self.victim_indicator.pack(side='left', padx=2)
        
        ttk.Label(indicator_frame, text="Attacker:").pack(side='left', padx=2)
        self.attacker_indicator = ttk.Label(indicator_frame, text="●", foreground='red')
        self.attacker_indicator.pack(side='left', padx=2)
        
    def on_scenario_change(self, event=None):
        """Handle scenario selection change"""
        selection = self.scenario_var.get()
        if ':' in selection:
            scenario_key = selection.split(':')[0]
            if scenario_key in self.scenarios:
                scenario = self.scenarios[scenario_key]
                # Update parameters based on scenario
                if 'target' in scenario:
                    self.target_ip.set(scenario['target'])
                if 'source' in scenario:
                    self.source_ip.set(scenario['source'])
                # Set attack type based on scenario
                if 'redirect' in scenario_key.lower():
                    self.attack_type.set('redirect')
                else:
                    self.attack_type.set('spoof')
        
        self.update_config_display()
    
    def update_config_display(self):
        """Update configuration display"""
        config_text = f"""Current Configuration:
============================

Network Setup:
• Victim Namespace: {self.config['victim_namespace']} ({self.config['victim_ip']})
• Attacker Namespace: {self.config['attacker_namespace']} ({self.config['attacker_ip']})
• Gateway: {self.config['gateway_ip']}
• Network: {self.config['network_subnet']}

Attack Parameters:
• Target IP: {self.target_ip.get()}
• Source IP: {self.source_ip.get()}
• Attack Type: {self.attack_type.get().upper()}
• Duration: {self.duration.get()} seconds
• Mode: {self.attack_mode.get()}

Available Scenarios:
"""
        
        for key, scenario in self.scenarios.items():
            config_text += f"• {key}: {scenario['name']}\n"
        
        self.config_display.delete('1.0', tk.END)
        self.config_display.insert('1.0', config_text)
        
    def start_monitoring(self):
        """Start background monitoring threads"""
        # Check namespace connectivity
        self.check_namespace_connectivity()
        
        # Start output processing
        self.process_output_queues()
        
    def check_namespace_connectivity(self):
        """Check if namespaces are accessible"""
        try:
            # Check victim namespace
            result = subprocess.run(['sudo', 'ip', 'netns', 'exec', 
                                   self.config['victim_namespace'], 'ip', 'addr'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.victim_indicator.config(foreground='green')
            else:
                self.victim_indicator.config(foreground='red')
                
            # Check attacker namespace  
            result = subprocess.run(['sudo', 'ip', 'netns', 'exec',
                                   self.config['attacker_namespace'], 'ip', 'addr'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.attacker_indicator.config(foreground='green')
            else:
                self.attacker_indicator.config(foreground='red')
                
        except Exception as e:
            self.status_text.config(text=f"Connection check failed: {e}")
            
        # Schedule next check
        self.root.after(5000, self.check_namespace_connectivity)
        
    def process_output_queues(self):
        """Process output from monitoring threads"""
        try:
            # Process victim tcpdump output
            while not self.output_queues['victim_tcpdump'].empty():
                line = self.output_queues['victim_tcpdump'].get_nowait()
                self.append_to_widget(self.victim_tcpdump, line)
                
            # Process attacker tcpdump output
            while not self.output_queues['attacker_tcpdump'].empty():
                line = self.output_queues['attacker_tcpdump'].get_nowait()
                self.append_to_widget(self.attacker_tcpdump, line)
                
            # Process victim routing output
            while not self.output_queues['victim_routing'].empty():
                output = self.output_queues['victim_routing'].get_nowait()
                self.victim_routing.delete('1.0', tk.END)
                self.victim_routing.insert('1.0', output)
                
            # Process attack output
            while not self.output_queues['attack_output'].empty():
                line = self.output_queues['attack_output'].get_nowait()
                self.append_to_widget(self.attack_output, line)
                
            # Process status updates
            while not self.output_queues['status'].empty():
                status = self.output_queues['status'].get_nowait()
                self.update_status_display(status)
                
        except Exception as e:
            pass  # Ignore queue empty exceptions
            
        # Schedule next processing
        self.root.after(100, self.process_output_queues)
        
    def append_to_widget(self, widget, text):
        """Append text to a scrolled text widget"""
        widget.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} {text}\n")
        widget.see(tk.END)
        
        # Limit widget size (keep last 1000 lines)
        lines = widget.get('1.0', tk.END).split('\n')
        if len(lines) > 1000:
            widget.delete('1.0', f'{len(lines)-1000}.0')
            
    def update_status_display(self, status):
        """Update status labels"""
        if 'packets_sent' in status:
            self.status_labels['Packets Sent:'].config(text=str(status['packets_sent']))
        if 'redirects_sent' in status:
            self.status_labels['Redirects Sent:'].config(text=str(status['redirects_sent']))
        if 'route_changes' in status:
            self.status_labels['Route Changes:'].config(text=str(status['route_changes']))
        if 'duration' in status:
            self.status_labels['Duration:'].config(text=status['duration'])
        if 'status' in status:
            self.status_labels['Status:'].config(text=status['status'])
            
    def start_attack(self):
        """Start the selected attack"""
        if self.attack_running:
            messagebox.showwarning("Warning", "Attack already running!")
            return
            
        try:
            attack_type = self.attack_type.get()
            target = self.target_ip.get()
            source = self.source_ip.get()
            duration = self.duration.get()
            mode = self.attack_mode.get()
            
            self.attack_running = True
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.stats['attack_start_time'] = time.time()
            
            # Start monitoring threads
            self.start_victim_tcpdump()
            self.start_attacker_tcpdump()
            self.start_victim_routing_monitor()
            
            # Build attack command
            if attack_type == 'spoof':
                cmd = [
                    'sudo', '/home/torr20/.local/bin/uv', 'run', 'icmp_spoofer_raw.py',
                    target, '--source', source, '--duration', duration,
                    '--namespace', self.config['attacker_namespace']
                ]
                if mode == 'flood':
                    cmd.append('--flood')
                elif mode == 'stealth':
                    cmd.append('--stealth')
            else:  # redirect
                cmd = [
                    'sudo', '/home/torr20/.local/bin/uv', 'run', 'icmp_redirect_raw.py',
                    target, source, self.config['gateway_ip'],
                    '--fake-gateway', self.config['attacker_ip'],
                    '--monitor-namespace', self.config['victim_namespace'],
                    '--duration', duration
                ]
                if mode == 'flood':  # For redirect, continuous mode is the default, flood means continuous
                    cmd.append('--continuous')
            
            # Start attack process
            self.start_attack_process(cmd)
            
            # Update topology if available
            if hasattr(self, 'ax'):
                self.draw_network_topology()
                
            self.status_text.config(text=f"Attack started: {attack_type.upper()}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start attack: {e}")
            self.stop_attack()
            
    def start_attack_process(self, cmd):
        """Start attack process in background thread"""
        def run_attack():
            try:
                self.attack_process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, universal_newlines=True
                )
                
                for line in iter(self.attack_process.stdout.readline, ''):
                    if not self.attack_running:
                        break
                    self.output_queues['attack_output'].put(line.strip())
                    
            except Exception as e:
                self.output_queues['attack_output'].put(f"Error: {e}")
        
        thread = threading.Thread(target=run_attack)
        thread.daemon = True
        thread.start()
        self.monitor_threads.append(thread)
        
    def start_victim_tcpdump(self):
        """Start tcpdump monitoring for victim namespace"""
        def monitor_tcpdump():
            try:
                cmd = ['sudo', 'ip', 'netns', 'exec', self.config['victim_namespace'],
                       'tcpdump', '-i', 'any', 'icmp', '-nn', '-l']
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                         stderr=subprocess.STDOUT, text=True)
                
                for line in iter(process.stdout.readline, ''):
                    if not self.attack_running:
                        break
                    self.output_queues['victim_tcpdump'].put(line.strip())
                    
            except Exception as e:
                self.output_queues['victim_tcpdump'].put(f"Error: {e}")
        
        thread = threading.Thread(target=monitor_tcpdump)
        thread.daemon = True
        thread.start()
        self.monitor_threads.append(thread)
        
    def start_attacker_tcpdump(self):
        """Start tcpdump monitoring for attacker namespace"""
        def monitor_tcpdump():
            try:
                cmd = ['sudo', 'ip', 'netns', 'exec', self.config['attacker_namespace'],
                       'tcpdump', '-i', 'any', 'icmp', '-nn', '-l']
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                         stderr=subprocess.STDOUT, text=True)
                
                for line in iter(process.stdout.readline, ''):
                    if not self.attack_running:
                        break
                    self.output_queues['attacker_tcpdump'].put(line.strip())
                    
            except Exception as e:
                self.output_queues['attacker_tcpdump'].put(f"Error: {e}")
        
        thread = threading.Thread(target=monitor_tcpdump)
        thread.daemon = True
        thread.start()
        self.monitor_threads.append(thread)
        
    def start_victim_routing_monitor(self):
        """Start routing table monitoring for victim namespace"""
        def monitor_routing():
            while self.attack_running:
                try:
                    cmd = ['sudo', 'ip', 'netns', 'exec', self.config['victim_namespace'],
                           'ip', 'route', 'show']
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        self.output_queues['victim_routing'].put(result.stdout)
                except Exception as e:
                    self.output_queues['victim_routing'].put(f"Error: {e}")
                
                time.sleep(2)  # Update every 2 seconds
        
        thread = threading.Thread(target=monitor_routing)
        thread.daemon = True
        thread.start()
        self.monitor_threads.append(thread)
        
    def stop_attack(self):
        """Stop the current attack"""
        self.attack_running = False
        
        if self.attack_process:
            try:
                self.attack_process.terminate()
                self.attack_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.attack_process.kill()
            self.attack_process = None
            
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        # Update topology
        if hasattr(self, 'ax'):
            self.draw_network_topology()
            
        # Generate results
        self.generate_attack_results()
        
        self.status_text.config(text="Attack stopped")
        
    def generate_attack_results(self):
        """Generate and display attack results"""
        if self.stats['attack_start_time']:
            duration = time.time() - self.stats['attack_start_time']
            
            results = f"""
Attack Results Summary
=====================
Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Attack Type: {self.attack_type.get().upper()}
Duration: {duration:.1f} seconds

Target: {self.target_ip.get()}
Source: {self.source_ip.get()}
Mode: {self.attack_mode.get()}

Statistics:
• Packets Sent: {self.stats['packets_sent']}
• Redirects Sent: {self.stats['redirects_sent']}
• Route Changes Detected: {self.stats['route_changes']}

Network Configuration:
• Victim: {self.config['victim_namespace']} ({self.config['victim_ip']})
• Attacker: {self.config['attacker_namespace']} ({self.config['attacker_ip']})
• Gateway: {self.config['gateway_ip']}

Verification Commands Used:
• sudo ip netns exec {self.config['victim_namespace']} tcpdump -i any icmp -nn
• sudo ip netns exec {self.config['victim_namespace']} ip route show
• sudo ip netns exec {self.config['victim_namespace']} ping -c 3 {self.target_ip.get()}

{'-' * 50}
"""
            
            self.results_display.insert(tk.END, results)
            self.results_display.see(tk.END)
            
            # Switch to results tab
            self.notebook.select(self.results_frame)
    
    def check_namespaces(self):
        """Check namespace status"""
        try:
            result = subprocess.run(['sudo', 'ip', 'netns', 'list'], 
                                  capture_output=True, text=True)
            
            info = f"Namespace Status:\n{result.stdout}\n"
            
            # Check each namespace
            for ns in [self.config['victim_namespace'], self.config['attacker_namespace']]:
                try:
                    result = subprocess.run(['sudo', 'ip', 'netns', 'exec', ns, 'ip', 'addr'],
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        info += f"\n{ns} namespace - ACTIVE\n"
                    else:
                        info += f"\n{ns} namespace - ERROR\n"
                except Exception as e:
                    info += f"\n{ns} namespace - FAILED: {e}\n"
            
            messagebox.showinfo("Namespace Status", info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to check namespaces: {e}")
    
    def setup_lab(self):
        """Setup lab environment"""
        setup_script = """
# ICMP Attack Lab Setup
echo "Setting up namespace lab environment..."
echo "This will run the namespace setup script."
python3 namespace_demo.py --setup-only
"""
        messagebox.showinfo("Lab Setup", 
                           "This would run the namespace setup.\n"
                           "Run 'python3 namespace_demo.py --setup-only' manually.")
    
    def save_results(self):
        """Save results to file"""
        try:
            filename = f"attack_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w') as f:
                f.write(self.results_display.get('1.0', tk.END))
            messagebox.showinfo("Success", f"Results saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {e}")
    
    def clear_results(self):
        """Clear results display"""
        self.results_display.delete('1.0', tk.END)
    
    def export_report(self):
        """Export comprehensive report"""
        try:
            filename = f"attack_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'attack_type': self.attack_type.get(),
                'target_ip': self.target_ip.get(),
                'source_ip': self.source_ip.get(),
                'duration': self.duration.get(),
                'mode': self.attack_mode.get(),
                'stats': self.stats,
                'results': self.results_display.get('1.0', tk.END)
            }
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
                
            messagebox.showinfo("Success", f"Report exported to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export report: {e}")

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    
    # Configure ttk styles
    style = ttk.Style()
    style.theme_use('clam')
    
    app = AttackMonitorGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.stop_attack()
        root.quit()

if __name__ == '__main__':
    main() 