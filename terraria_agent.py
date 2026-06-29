import mss
import numpy as np
import cv2
import time
import pydirectinput

# ── CALIBRATION VALUES ──
BOSS_MAX_RED = 0
PLAYER_MAX_RED = 0

# ── SETTINGS ──
HEAL_THRESHOLD = 0.5
FLEE_THRESHOLD = 0.25
POTION_COOLDOWN = 60
SCREEN_CENTER_X = 960
PLAYER_Y = 370

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

def find_boss(frame):
    game_area = frame[200:900, 200:1820]  # trim edges to exclude HUD/buff icons
    hsv = cv2.cvtColor(game_area, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 80, 60), (10, 255, 200))
    red2 = cv2.inRange(hsv, (160, 80, 60), (180, 255, 200))
    mask = cv2.bitwise_or(red1, red2)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1000:  # ignore small icons like potion cooldown
        return None
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"]) + 100
    return cx, cy

def calibrate():
    global BOSS_MAX_RED, PLAYER_MAX_RED
    print("=== CALIBRATION ===")
    print("Step 1: Spawn Eye of Cthulhu and let it sit at FULL health")
    print("Calibrating boss health in 5 seconds...")
    time.sleep(5)
    frame = capture()
    BOSS_MAX_RED = count_red(frame[1015:1045, 694:1224])
    print(f"Boss calibrated! ({BOSS_MAX_RED} red pixels)")

    print("\nStep 2: Make sure YOUR health is full")
    print("Calibrating player health in 5 seconds...")
    time.sleep(5)
    frame = capture()
    PLAYER_MAX_RED = count_red(frame[44:72, 1614:1882])
    print(f"Player calibrated! ({PLAYER_MAX_RED} red pixels)")
    print("\nCalibration done! Starting agent in 3 seconds...")
    time.sleep(3)

def run_agent():
    last_potion_time = 0
    is_shooting = False

    print("Agent running! Press Ctrl+C to stop.")

    while True:
        frame = capture()
        boss_hp = get_boss_health(frame)
        player_hp = get_player_health(frame)
        boss_pos = find_boss(frame)

        print(f"Boss HP: {boss_hp:.0%} | Player HP: {player_hp:.0%} | Boss pos: {boss_pos}")

        # ── HEAL ──
        potion_ready = (time.time() - last_potion_time) > POTION_COOLDOWN
        if player_hp < HEAL_THRESHOLD and potion_ready:
            print(">> HEALING")
            pydirectinput.press('h')
            last_potion_time = time.time()

        # ── BOSS DEFEATED ──
        if boss_hp < 0.02:
            print(">> Boss defeated! Stopping agent.")
            if is_shooting:
                pydirectinput.mouseUp(button='left')
            break

        # ── ENGAGE ──
        if boss_pos:
            bx, by = boss_pos

            # Aim at boss
            pydirectinput.moveTo(bx, by)

            # Start shooting if not already
            if not is_shooting:
                pydirectinput.mouseDown(button='left')
                is_shooting = True

            # Always strafe away from boss
            if bx > SCREEN_CENTER_X:
                pydirectinput.keyDown('a')
                pydirectinput.keyUp('d')
            else:
                pydirectinput.keyDown('d')
                pydirectinput.keyUp('a')

            # Jump every 1.5 seconds
            if int(time.time() * 10) % 15 == 0:
                pydirectinput.press('space')

        # ── BOSS NOT VISIBLE ──
        else:
            if is_shooting:
                pydirectinput.mouseUp(button='left')
                is_shooting = False
            # Keep moving right to find boss
            pydirectinput.keyDown('d')
            print(">> Boss not visible, searching...")

        time.sleep(0.05)  # 20 FPS loop

# ── MAIN ──
calibrate()
run_agent()