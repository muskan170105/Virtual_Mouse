# 🖱️ AI Virtual Mouse

An advanced, gesture-controlled virtual mouse powered by computer vision and artificial intelligence. This project uses your webcam to track your hand movements and translates them into precise mouse and system controls, completely eliminating the need for a physical mouse!

Built using **Python**, **OpenCV**, **MediaPipe**, and **PyAutoGUI**.

## ✨ Features & Supported Gestures

The system tracks specific hand landmarks to perform various OS-level actions with pure EMA smoothing to eliminate cursor jitter.

### 🎯 Core Mouse Controls
* **Move Cursor:** Raise your **Index finger only**.
* **Left Click:** Pinch **Index + Thumb** together.
* **Double Click:** Two quick pinches (**Index + Thumb**).
* **Drag & Drop:** Pinch **Index + Thumb** and hold for 0.70s to lock, then move to drag.
* **Right Click:** Point your **Thumb far sideways** (while keeping index down).
* **Middle Click:** Raise **Three fingers** straight up.
* **Scroll (Up/Down/Left/Right):** Raise **Index + Middle fingers** with a small gap and move your hand.

### 💻 System & Media Controls
* **Volume Up:** Pinch **Thumb + Middle finger** (while keeping index/ring/pinky up).
* **Volume Down:** Bring **Thumb + Middle finger** closer together.
* **Zoom In:** Spread **Index + Middle fingers** apart (Triggers `Ctrl + =`).
* **Zoom Out:** Bring **Index + Middle fingers** closer together (Triggers `Ctrl + -`).
* **Switch Window (Alt+Tab):** Fast horizontal sweep with an **Open Palm**.
* **Take Screenshot:** Touch the tips of your **Thumb + Pinky** together (saves directly to Desktop).
* **Pause Tracking:** Hold all **5 fingers open and perfectly still**.

### 🎨 Draw Mode
* Press **`D`**: Toggle Draw mode on/off.
* Press **`C`**: Clear the screen of drawings.
* **Erase:** Tilt your hand **90° sideways**.
* Press **`Q`** or **`ESC`**: Quit the application.

### 🎮 Sensitivity Profiles
Switch between tuning profiles on the fly by pressing the number keys:
* **`1`**: Default Profile (Balanced)
* **`2`**: Presentation Profile (Slower, smoother movements)
* **`3`**: Gaming Profile (High sensitivity, faster clicks)

## 🚀 Installation & Setup

### Prerequisites
You need to have **Python 3.8+** installed on your system. A working webcam is also required.

### 1. Clone the repository
```bash
git clone https://github.com/muskan170105/Virtual-Mouse.git
cd Virtual-Mouse
```
### 2. Install required dependencies
Open your terminal or command prompt and install the necessary Python libraries:

```bash
pip install opencv-python mediapipe pyautogui numpy
```

### 3. Run the application
```bash
python virtual_mouse.py
```

## 🛠️ How it Works
1. **MediaPipe** captures your hand via your webcam and maps 21 3D landmarks in real-time.
2. The script calculates the relative normalized distances and angles between these specific landmarks to detect complex gestures.
3. **PyAutoGUI** takes these recognized gestures and fires the corresponding system-level command (like moving the cursor, clicking, or triggering keyboard shortcuts).
4. Smooth EMA (Exponential Moving Average) filtering and dead-zones are applied to eliminate cursor jitter and ignore hand micro-tremors.

---
*Created by Muskan*

