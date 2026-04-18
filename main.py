import time
import cv2
import mediapipe as mp
from pynput.mouse import Button, Controller
from pynput import keyboard
import ctypes
import showLandmarks as slm

SMOOTHING = 0.6 # lower = smoother
THUMB_TIP = 4
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12

IMAGE_PAUSED = cv2.imread("PTOUNPAUSE.png")
is_paused = False
was_paused = False

CLICK_THRESHOLD = 0.3
DOUBLE_CLICK_THRESHOLD = 0.4
is_pressed = False
click_start = 0

mouse = Controller()
prev_x, prev_y = 0, 0

# Default camera (index O)
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def getScreenResolution():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

def smoothMouse(x, y):
    global prev_x, prev_y
    prev_x = prev_x + (x - prev_x) * SMOOTHING
    prev_y = prev_y + (y - prev_y) * SMOOTHING
    return int(prev_x), int(prev_y)

def moveMouseToPosition(x, y):
    mouse.position = (x, y)

def convertCoordsToScreenResolution(x, y, margin=0.1):
    # Clamp
    x = max(0, min(1, x))
    y = max(0, min(1, y))

    # Remove borders
    x = (x - margin) / (1 - 2 * margin)
    y = (y - margin) / (1 - 2 * margin)

    # Re-clamp
    x = max(0, min(1, x))
    y = max(0, min(1, y))

    screen_w, screen_h = getScreenResolution()
    return int(x * screen_w), int(y * screen_h)

def mpSpacePosition(hand):
    return hand.landmark[THUMB_TIP].x, hand.landmark[THUMB_TIP].y

# region PYNPUT CLICK
def isClicking(hand):
    dist_x = abs(hand.landmark[INDEX_FINGER_TIP].x - hand.landmark[THUMB_TIP].x)
    dist_y = abs(hand.landmark[INDEX_FINGER_TIP].y - hand.landmark[THUMB_TIP].y)
    return dist_x < 0.05 and dist_y < 0.05

def isRightClick(hand):
    dist_x = abs(hand.landmark[MIDDLE_FINGER_TIP].x - hand.landmark[THUMB_TIP].x)
    dist_y = abs(hand.landmark[MIDDLE_FINGER_TIP].y - hand.landmark[THUMB_TIP].y)
    return dist_x < 0.05 and dist_y < 0.05

def mouseClick(nb_click):
    mouse.click(Button.left, nb_click)

def mousePress():
    global is_pressed, click_start
    if not is_pressed:
        mouse.press(Button.left)
        is_pressed = True
        click_start = time.time()

def mouseRelease():
    global is_pressed, click_start
    if is_pressed:
        mouse.release(Button.left)
        is_pressed = False
        click_start = 0
# end region

# region PYNPUT PAUSE
def on_press(key):
    if hasattr(key, 'char') and key.char == 'p':
        global is_paused
        is_paused = not is_paused
        if is_paused:
            print("paused")
        else:
            print("unpaused")

listener = keyboard.Listener(on_press = on_press)
listener.start()
#end region

# region MAIN
while True:
    if is_paused:
        # Used to display the img one time only
        if not was_paused:
            cv2.imshow("Camera", IMAGE_PAUSED)
            was_paused = True
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        continue

    was_paused = False

    success, image = cap.read()
    if not success:
        break
    image = cv2.flip(image, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    # Process
    h_results = hands.process(rgb)

    rgb.flags.writeable = True
    # ===== HANDS =====
    if not h_results.multi_hand_landmarks:
        if is_pressed:
            is_pressed = False
    else:
        for hand in h_results.multi_hand_landmarks:
            slm.handMarkMap(image, hand)

        x_mp, y_mp = mpSpacePosition(hand)
        x_sr, y_sr = convertCoordsToScreenResolution(x_mp, y_mp)
        x_sm, y_sm = smoothMouse(x_sr, y_sr)
        moveMouseToPosition(x_sm, y_sm)

        if isRightClick(hand):
            mouse.click(Button.right, 1)

        if isClicking(hand):
            mousePress()
        else:
            if is_pressed:
                mouseRelease()
                duration = time.time() - click_start
                if duration <= CLICK_THRESHOLD:
                    mouseClick(1)

    cv2.imshow("Camera", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# end region

listener.stop()
# Release the cam
cap.release()
# Close all windows
cv2.destroyAllWindows()