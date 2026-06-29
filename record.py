import mss
import numpy as np
import cv2
import time
import pickle
import os
from pynput import keyboard, mouse

# Storage
recordings = []
current_keys = set()
key_press_times = {}
mouse_pos = (0, 0)
mouse_clicking = False

# Dash detection
last_a_press = 0
last_d_press = 0
DASH_WINDOW = 0.3
is_dashing_left = False
is_dashing_right = False

# Fly detection
space_held_since = None
HOLD_THRESHOLD = 0.2  # seconds of holding space = flying

def on_press(key):
    global last_a_press, last_d_press
    global is_dashing_left, is_dashing_right
    global space_held_since

    try:
        k = key.char
    except:
        k = str(key)

    now = time.time()
    current_keys.add(k)
    key_press_times[k] = now

    # Track space hold start
    if k == 'Key.space' and space_held_since is None:
        space_held_since = now

    # Detect dash left (double tap A)
    if k == 'a':
        if now - last_a_press < DASH_WINDOW:
            is_dashing_left = True
        last_a_press = now

    # Detect dash right (double tap D)
    if k == 'd':
        if now - last_d_press < DASH_WINDOW:
            is_dashing_right = True
        last_d_press = now

def on_release(key):
    global space_held_since, is_dashing_left, is_dashing_right

    try:
        k = key.char
    except:
        k = str(key)

    current_keys.discard(k)

    if k == 'Key.space':
        space_held_since = None

    # Reset dash flags on key release
    if k == 'a':
        is_dashing_left = False
    if k == 'd':
        is_dashing_right = False

def on_move(x, y):
    global mouse_pos
    mouse_pos = (x, y)

def on_click(x, y, button, pressed):
    global mouse_clicking
    if button == mouse.Button.left:
        mouse_clicking = pressed

# Start listeners
kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
kb_listener.start()
mouse_listener.start()

def get_action_label():
    global space_held_since

    now = time.time()

    # Dash takes priority — it's a short window action
    if is_dashing_left:
        return 6  # dash left
    if is_dashing_right:
        return 7  # dash right

    # Fly — holding space longer than threshold
    if space_held_since is not None:
        held_duration = now - space_held_since
        if held_duration > HOLD_THRESHOLD:
            return 5  # fly

    # Heal
    if 'h' in current_keys:
        return 4

    # Single jump — space pressed but not held long enough yet
    if 'Key.space' in current_keys:
        return 3

    # Movement
    if 'a' in current_keys:
        return 1  # move left
    if 'd' in current_keys:
        return 2  # move right

    # Default — shoot/nothing
    return 0

print("=== TERRARIA RECORDER (with fly + dash) ===")
print("Tips:")
print("  - Fight the boss normally")
print("  - Use wings to fly toward/away from boss")
print("  - Use Shield of Cthulhu dash to dodge attacks")
print("  - Try to win! Good gameplay = better training data")
print("  - Record at least 5 fights")
print("")
print("Starting in 5 seconds — switch to Terraria!")
time.sleep(5)
print("Recording! Press Ctrl+C to stop and save...")

# Count actions for balance check
action_counts = {i: 0 for i in range(8)}

try:
    with mss.MSS() as sct:
        monitor = sct.monitors[1]
        frame_count = 0
        start_time = time.time()



        while True:
            img = sct.grab(monitor)
            frame = np.array(img)
            small = cv2.resize(frame, (480, 270))

            action = get_action_label()
            action_counts[action] += 1

            recordings.append({
                'frame': small,
                'action': action,
                'timestamp': time.time()
            })

            frame_count += 1
            if frame_count % 20 == 0:
                elapsed = time.time() - start_time
                action_names = ['shoot', 'left', 'right', 'jump', 'heal', 'fly', 'dash left', 'dash right']
                print(f"Frame {frame_count} ({elapsed:.0f}s) | Action: {action_names[action]}")

            time.sleep(0.05)

except KeyboardInterrupt:
    print(f"\nStopping... recorded {len(recordings)} frames!")

    # Show action distribution
    action_names = ['shoot', 'left', 'right', 'jump', 'heal', 'fly', 'dash left', 'dash right']
    print("\nAction distribution:")
    for i, name in enumerate(action_names):
        count = action_counts[i]
        print(f"  {name}: {count} ({count/len(recordings):.0%})")

    # Save with unique filename
    i = 1
    while os.path.exists(f'recording_{i}.pkl'):
        i += 1
    filename = f'recording_{i}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(recordings, f)
    print(f"\nSaved as {filename}!")