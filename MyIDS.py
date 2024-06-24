import pandas as pd
import pyshark
import queue
import threading
import asyncio
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, Sequential
import os

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

# Function for pattern analysis using machine learning
def analyze_patterns_with_ml(X_train, y_train, X_test):
    model_path = 'deauth_detection_model.h5'

    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        model = Sequential([
            layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
        model.save(model_path)
    
    predicted_labels = model.predict(X_test)
    return predicted_labels

# Function for behavioral analysis
def analyze_behavior(packets):
    pass

# Function for anomaly detection
def detect_anomalies(packets):
    pass

def main():
    # Default interface to "Wi-Fi"
    interface = "Wi-Fi"

    capture_thread = threading.Thread(target=live_capture, args=(interface,))
    capture_thread.daemon = True
    capture_thread.start()
    print("starting capture...")

    data = []
    
    try:
        while True:
            while not packet_queue.empty():
                raw_packet = packet_queue.get()
                features = extract_features(raw_packet)
                if features:
                    data.append(features)
            
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                total_deauth = df['type'].sum()
                print("Total deauth:", total_deauth)
                
                overlap = 5
                deauth_threshold = 5
                window_size = 10

                detect_attacks(df, window_size, deauth_threshold, overlap)
                
                # Additional analysis can be performed here
    except KeyboardInterrupt:
        print("Live capture stopped.")
        capture_thread.join()

if __name__ == "__main__":
    main()
