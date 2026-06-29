import mss
import numpy as np
import cv2
import torch
import torch.nn as nn
import pydirectinput
import time

# ── LOAD MODEL — 8 actions ──
class TerrariaAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 33 * 60, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 8)  # 8 actions
        )

    def forward(self, x):
        return self.fc(self.conv(x))

print("Loading model...")
model = TerrariaAgent()
model.load_state_dict(torch.load('terraria_model.pth'))
model.eval()
print("Model loaded!")

# ── CALIBRATION VALUES ──
BOSS_MAX_RED = 0
PLAYER_MAX_RED = 0

# ── SETTINGS ──
POTION_COOLDOWN = 60
PHASE_TRANSITION_WAIT = 5
PHASE_JUMP_THRESHOLD = 0.3
RAGE_COOLDOWN = 30
DASH_COOLDOWN = 1.5         # cooldown between dashes
DASH_DOUBLE_TAP = 0.08      # delay between double tap presses

# ── STATE ──
is_space_held = False
last_dash_left = 0
last_dash_right = 0

def capture():
    with mss.MSS() as sct:
        monitor = sct.monitors[1]
        img = sct.grab(monitor)
        return np.array(img)

def count_red(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    return cv2.countNonZero(cv2.bitwise_or(red1, red2))

def get_boss_health(frame):
    region = frame[1015:1045, 694:1224]
    red = count_red(region)
    if BOSS_MAX_RED == 0:
        return 0.0
    return min(round(red / BOSS_MAX_RED, 2), 1.0)

def get_player_health(frame):
    region = frame[44:72, 1614:1882]
    red = count_red(region)
    if PLAYER_MAX_RED == 0:
        return 0.0
    return min(round(red / PLAYER_MAX_RED, 2), 1.0)

def find_boss(frame, last_pos=None):
    game_area = frame[200:900, 200:1820]
    hsv = cv2.cvtColor(game_area, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 80, 60), (10, 255, 200))
    red2 = cv2.inRange(hsv, (160, 80, 60), (180, 255, 200))
    mask = cv2.bitwise_or(red1, red2)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return last_pos
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1000:
        return last_pos
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return last_pos
    cx = int(M["m10"] / M["m00"]) + 200
    cy = int(M["m01"] / M["m00"]) + 200
    if last_pos is not None:
        dist = ((cx - last_pos[0])**2 + (cy - last_pos[1])**2) ** 0.5
        if dist > 300:
            return last_pos
    return cx, cy

def calibrate():
    global BOSS_MAX_RED, PLAYER_MAX_RED
    print("=== CALIBRATION ===")
    print("Step 1: Spawn boss at FULL health")
    print("Calibrating in 3 seconds...")
    time.sleep(3)
    frame = capture()
    BOSS_MAX_RED = count_red(frame[1015:1045, 694:1224])
    print(f"Boss calibrated! ({BOSS_MAX_RED} red pixels)")
    print("Make sure YOUR health is full, calibrating in 3 seconds...")
    time.sleep(3)
    frame = capture()
    PLAYER_MAX_RED = count_red(frame[44:72, 1614:1882])
    print(f"Player calibrated! ({PLAYER_MAX_RED} red pixels)")
    print("Starting in 1 second...")
    time.sleep(1)

# ── ACTION NAMES ──
action_names = ['shoot', 'move left', 'move right', 'jump', 'heal', 'fly', 'dash left', 'dash right']

def execute_action(action, boss_pos, last_action, last_potion_time, is_shooting):
    global is_space_held, last_dash_left, last_dash_right
    now = time.time()

    if boss_pos:
        pydirectinput.moveTo(boss_pos[0], boss_pos[1])

    # Release movement keys if action changed
    if last_action != action:
        pydirectinput.keyUp('a')
        pydirectinput.keyUp('d')
        # Release space if we were flying and now doing something else
        if last_action == 5 and action != 5:
            pydirectinput.keyUp('space')
            is_space_held = False

    if action == 0:  # shoot
        if not is_shooting and boss_pos:
            pydirectinput.mouseDown(button='left')
            is_shooting = True

    elif action == 1:  # move left
        if not is_shooting and boss_pos:
            pydirectinput.mouseDown(button='left')
            is_shooting = True
        pydirectinput.keyDown('a')
        pydirectinput.keyUp('d')

    elif action == 2:  # move right
        if not is_shooting and boss_pos:
            pydirectinput.mouseDown(button='left')
            is_shooting = True
        pydirectinput.keyDown('d')
        pydirectinput.keyUp('a')

    elif action == 3:  # jump — single tap space
        if not is_shooting and boss_pos:
            pydirectinput.mouseDown(button='left')
            is_shooting = True
        if not is_space_held:
            pydirectinput.press('space')

    elif action == 4:  # heal
        potion_ready = (now - last_potion_time) > POTION_COOLDOWN
        if potion_ready:
            pydirectinput.press('h')
            last_potion_time = now
            print(">> HEALING")

    elif action == 5:  # fly — hold space
        if not is_shooting and boss_pos:
            pydirectinput.mouseDown(button='left')
            is_shooting = True
        if not is_space_held:
            pydirectinput.keyDown('space')
            is_space_held = True
            print(">> FLYING")

    elif action == 6:  # dash left
        dash_ready = (now - last_dash_left) > DASH_COOLDOWN
        if dash_ready:
            if not is_shooting and boss_pos:
                pydirectinput.mouseDown(button='left')
                is_shooting = True
            # Double tap A to dash
            pydirectinput.keyDown('a')
            pydirectinput.keyUp('a')
            time.sleep(DASH_DOUBLE_TAP)
            pydirectinput.keyDown('a')
            last_dash_left = now
            print(">> DASH LEFT")

    elif action == 7:  # dash right
        dash_ready = (now - last_dash_right) > DASH_COOLDOWN
        if dash_ready:
            if not is_shooting and boss_pos:
                pydirectinput.mouseDown(button='left')
                is_shooting = True
            # Double tap D to dash
            pydirectinput.keyDown('d')
            pydirectinput.keyUp('d')
            time.sleep(DASH_DOUBLE_TAP)
            pydirectinput.keyDown('d')
            last_dash_right = now
            print(">> DASH RIGHT")

    return is_shooting, last_potion_time

# ── MAIN LOOP ──
def run():
    global BOSS_MAX_RED, is_space_held

    last_action = None
    last_potion_time = 0
    last_rage_time = 0
    is_shooting = False
    boss_missing_since = None
    prev_boss_hp = 1.0
    current_phase = 1

    print("AI Agent running! Press Ctrl+C to stop.")

    while True:
        frame = capture()
        boss_hp = get_boss_health(frame)
        player_hp = get_player_health(frame)
        boss_pos = find_boss(frame)

        # ── PHASE TRANSITION ──
        if boss_hp > prev_boss_hp + PHASE_JUMP_THRESHOLD:
            current_phase += 1
            print(f">> PHASE {current_phase} DETECTED!")
            BOSS_MAX_RED = count_red(frame[1015:1045, 694:1224])
            print(f">> Recalibrated for phase {current_phase}")

        # ── BOSS MISSING ──
        if boss_hp < 0.02:
            if boss_missing_since is None:
                boss_missing_since = time.time()
            elapsed = time.time() - boss_missing_since
            if elapsed > PHASE_TRANSITION_WAIT:
                print(">> Boss confirmed defeated!")
                pydirectinput.mouseUp(button='left')
                pydirectinput.keyUp('a')
                pydirectinput.keyUp('d')
                pydirectinput.keyUp('space')
                is_space_held = False
                break
            else:
                print(f">> Waiting to confirm... ({elapsed:.1f}s / {PHASE_TRANSITION_WAIT}s)")
                if last_action == 1 or last_action == 6:
                    pydirectinput.keyDown('a')
                else:
                    pydirectinput.keyDown('d')
        else:
            boss_missing_since = None

        # ── EMERGENCY HEAL ──
        if player_hp < 0.25:
            potion_ready = (time.time() - last_potion_time) > POTION_COOLDOWN
            if potion_ready:
                pydirectinput.press('h')
                last_potion_time = time.time()
                print(">> EMERGENCY HEAL")

        # ── RAGE/ADRENALINE ──
        rage_ready = (time.time() - last_rage_time) > RAGE_COOLDOWN
        if rage_ready and boss_pos and boss_hp > 0.02:
            pydirectinput.press('v')
            last_rage_time = time.time()
            print(">> RAGE/ADRENALINE ACTIVATED")

        # ── BOSS NOT VISIBLE ──
        if not boss_pos:
            if last_action in [1, 6]:
                pydirectinput.keyDown('a')
                pydirectinput.keyUp('d')
            else:
                pydirectinput.keyDown('d')
                pydirectinput.keyUp('a')
            print(">> Boss not visible, maintaining movement...")

        else:
            # ── GET AI DECISION ──
            small = cv2.cvtColor(
                cv2.resize(frame, (480, 270)),
                cv2.COLOR_BGR2GRAY
            ).astype(np.float32) / 255.0

            tensor = torch.tensor(small).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                output = model(tensor)
                action = torch.argmax(output).item()

            print(f"Phase: {current_phase} | Boss HP: {boss_hp:.0%} | Player HP: {player_hp:.0%} | Action: {action_names[action]}")
            is_shooting, last_potion_time = execute_action(
                action, boss_pos, last_action, last_potion_time, is_shooting
            )
            last_action = action

        prev_boss_hp = boss_hp
        time.sleep(0.05)

# ── START ──
calibrate()
run()