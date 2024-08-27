import pyshark
import queue
import threading
import asyncio
from datetime import datetime

# Global queue to hold captured packets
packet_queue = queue.Queue()

# Function to capture live packets
def live_capture(interface):
    # Create a new event loop and set it as the current event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    capture = pyshark.LiveCapture(interface=interface)
    try:
        for packet in capture.sniff_continuously():
            packet_queue.put(packet)
    except pyshark.capture.capture.TSharkCrashException as e:
        print(f"TShark crashed: {e}")
    finally:
        loop.run_until_complete(capture.close_async())
        loop.close()

# Function to extract features from packets
def extract_features(packet):
    if 'WLAN' in packet:
        wlan_layer = packet.wlan
        packet_type = 1 if 'wlan.fc.type_subtype' in packet and packet.wlan.fc_type_subtype == '12' else 0  # 12 indicates deauth
        timestamp = float(packet.sniff_time.timestamp())
        src = wlan_layer.ta if hasattr(wlan_layer, 'ta') else None
        dst = wlan_layer.da if hasattr(wlan_layer, 'da') else None
        rssi = int(packet.radiotap.dbm_antsignal) if hasattr(packet.radiotap, 'dbm_antsignal') else -100
        eapol = 1 if 'EAPOL' in packet else 0

        return {
            'type': packet_type,
            'src': src,
            'dst': dst,
            'rssi': rssi,
            'timestamp': timestamp,
            'eapol': eapol
        }
    return None

def detect_attacks(packets, window_size, deauth_threshold, overlap):
    attack_windows = []
    num_packets = len(packets)
    
    start = 0
    while start < num_packets:
        end = min(start + window_size, num_packets)
        window = packets.iloc[start:end]
        
        deauth_count = window['type'].sum()
        
        if deauth_count >= deauth_threshold:
            attack_windows.append(window)
        
        start += window_size - overlap
    
    if attack_windows:
        print("Attack detected in the following windows:")
        for window in attack_windows:
            print(window)
        return attack_windows
    else:
        print("No deauth attack detected.")
        return []
