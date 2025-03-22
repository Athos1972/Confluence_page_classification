import pyautogui
import random
import time
import subprocess


def send_notification(message):
    script = f'display notification "{message}" with title "Mouse Mover Python"'
    subprocess.run(["osascript", "-e", script])


def move_mouse_slightly():
    while True:
        # Get the current mouse position
        current_x, current_y = pyautogui.position()
        # Generate a small random movement
        move_x = random.randint(-5, 5)
        move_y = random.randint(-5, 5)
        # Move the mouse slightly from its current position
        pyautogui.moveRel(move_x, move_y, duration=0.2)
        # Send a notification
        send_notification(f'Maus-Mover. Aktiv. Mausposition verändert um ({move_x}, {move_y}) pixels')
        # Wait for 1 minute
        time.sleep(60)


if __name__ == "__main__":
    move_mouse_slightly()
