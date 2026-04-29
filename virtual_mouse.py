"""
Virtual Mouse — Optimized Showcase Version
==========================================
BUGS FIXED:
  1. Right click no longer fires during pinch — pinch guard added
  2. Scroll anchor correctly initialises on gesture entry
  3. Click uses normalised distance (hand-distance invariant)
  4. Cursor jitter eliminated — pure EMA, no velocity prediction
  5. Middle click uses cooldown, not unreliable lock flag
  6. Window sweep threshold raised — no accidental triggers
  7. Cursor snap on reappearance fixed — init resets after 5 no-hand frames
  8. Tuning constants tightened across all gestures

GESTURES:
  Index only                   → Move cursor  (smooth)
  Index + thumb pinch          → Left click   (hold 0.14s)
  Pinch held 0.70s             → Drag
  Two quick pinches            → Double click
  Thumb far sideways (index ↓) → Right click  (pinch-guarded)
  Index + middle UP + gap      → Scroll (up/down/left/right)
  Three fingers up             → Middle click
  All 5 fingers open + still   → Pause
  Open palm fast sweep         → Alt+Tab window switch
  Thumb+pinky tips touch       → Screenshot → Desktop
  Thumb+middle pinch           → Volume Up (index/ring/pinky UP)
  Thumb+middle pinch closer    → Volume Down
  Spread index+middle apart    → Zoom In (Ctrl +=)
  Close index+middle together  → Zoom Out (Ctrl +-)
  Hand tilted 90° sideways     → Erase (draw mode)
  D key → Draw mode  |  C key → Clear  |  Q/ESC → Quit
  1 key → Default profile  |  2 key → Presentation  |  3 key → Gaming
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import os
import ctypes
from collections import deque

# ═══════════════════════════════════════════════════════════════════
#  SETUP
# ═══════════════════════════════════════════════════════════════════
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.80,
    min_tracking_confidence=0.70,
    model_complexity=0
)

# ═══════════════════════════════════════════════════════════════════
#  TUNING — change these to adjust feel
# ═══════════════════════════════════════════════════════════════════

# Cursor
FR            = 0.20    # frame border crop
SMOOTH        = 14      # EMA factor — higher = smoother, less jitter
DEAD_ZONE     = 12      # pixels: micro-tremors below this are ignored

# Click  (normalised distances — consistent regardless of hand distance from cam)
CLICK_DIST_N  = 0.055   # normalised pinch distance to register click
CLICK_CONFIRM = 0.14    # seconds pinch must be held to confirm click
CLICK_COOL    = 0.60    # seconds between any two clicks (prevents multi-fire)
DRAG_HOLD     = 0.70    # seconds hold to start drag
DBL_TIME      = 0.35    # seconds: two clicks within this = double click

# Right click  — thumb must be far sideways AND NOT close to index tip
RC_THUMB_H      = 0.18  # normalised: thumb tip vs index MCP x-offset
RC_PINCH_GUARD  = 0.09  # right click blocked if thumb-index dist < this (pinch guard)
RC_COOL         = 1.0   # right click cooldown

# Scroll
SC_GAP        = 0.055   # min gap between index & middle tips
SC_DEAD       = 0.030   # normalised dead zone
SC_SENS       = 12      # scroll speed (slightly calmer)
SC_DRIFT      = 0.04    # anchor drift rate

# Window switch
WS_DELTA      = 0.10    # hand travel threshold (lowered for easier wave)
WS_WINDOW     = 0.38    # seconds accumulation window
WS_COOL       = 0.8     # cooldown between switches (faster for waving through windows)

# Screenshot
SS_DIST       = 0.070   # normalised thumb-pinky tip distance
SS_COOL       = 1.5     # cooldown between screenshots

# Erase
ER_TILT       = 0.10    # pinky tip must be this far left of wrist

# Volume control  (thumb+middle pinch, index+ring+pinky up)
VOL_DIST_N    = 0.060   # normalised thumb-middle distance threshold
VOL_COOL      = 0.50    # seconds between volume steps

# Zoom gesture  (index+middle spread/close, ring+pinky+thumb down)
ZOOM_OPEN_N   = 0.14    # spread threshold → zoom in
ZOOM_CLOSE_N  = 0.055   # close threshold  → zoom out
ZOOM_COOL     = 0.65    # seconds between zoom steps (slower = feels smooth)

# Tab switch
TS_COOL          = 0.60    # cooldown between browser tab switches

# Fist scroll (joystick style)
FIST_DEAD        = 0.025   # normalised dead zone from anchor before scrolling starts
FIST_SCROLL_SENS = 60      # scroll speed multiplier
FIST_RATE        = 0.04    # seconds between each auto-scroll step

# Draw
DRAW_COLOR    = (0, 140, 255)
DRAW_THICK    = 4

# ── Profiles (press 1/2/3 to switch) ─────────────────────────────────────────
PROFILES = {
    '1': dict(name="Default",      SMOOTH=14, CLICK_DIST_N=0.055, CLICK_CONFIRM=0.14, DRAG_HOLD=0.70, SC_SENS=12),
    '2': dict(name="Presentation", SMOOTH=18, CLICK_DIST_N=0.048, CLICK_CONFIRM=0.18, DRAG_HOLD=0.80, SC_SENS=8),
    '3': dict(name="Gaming",       SMOOTH=8,  CLICK_DIST_N=0.065, CLICK_CONFIRM=0.09, DRAG_HOLD=0.55, SC_SENS=18),
}
active_profile = '1'

# ═══════════════════════════════════════════════════════════════════
#  STATE
# ═══════════════════════════════════════════════════════════════════
canvas          = None
draw_mode       = False
prev_draw_x     = 0
prev_draw_y     = 0

# Cursor smoothing (vel_x/vel_y removed — pure EMA, no jitter)
smooth_x        = 0.0
smooth_y        = 0.0
cursor_init     = False
prev_screen_x   = 0
prev_screen_y   = 0

# Click state machine
# States: OPEN → PRESSING → DRAGGING
click_state     = "OPEN"
click_down_t    = 0.0
last_any_click  = 0.0
last_click_t    = 0.0   # for double click timing
drag_active     = False

# Individual gesture locks/cooldowns
right_cd        = 0.0
middle_cd       = 0.0   # replaces middle_lock — cooldown prevents rapid-fire
ss_lock         = False
ss_last_t       = 0.0
no_hand_frames  = 0     # counts consecutive frames with no hand detected

# Fist / tab drag
fist_start_t    = 0.0
fist_held       = False
tab_drag        = False

# Scroll / Auto-scroll
scroll_on          = False
prev_scroll_on     = False
scroll_anc_nx      = 0.0
scroll_anc_ny      = 0.0
scroll_acc_y       = 0.0
scroll_acc_x       = 0.0
auto_scroll_on     = False
auto_scroll_t      = 0.0
fist_anchor_y      = -1.0  # anchor Y captured when fist gesture starts (-1 = not set)
fist_scroll_t      = 0.0   # timer for scroll rate limiting

# Window switch
ws_last_t       = 0.0
ws_hist         = deque()

# Volume / Zoom / Tab cooldowns
vol_cd          = 0.0
zoom_cd         = 0.0
zoom_ref_dist   = -1.0
tab_sw_last_t   = 0.0

# Sticky Switcher States
alt_down        = False  # Is Alt key currently held?
ctrl_down       = False  # Is Ctrl key currently held (for tabs)?
last_sw_t       = 0.0    # Time of last sweep (for auto-release)
SW_RELEASE_T    = 0.8    # Seconds of stillness before releasing Alt/Ctrl

# HUD / Smoothing
fps_hist        = deque(maxlen=30)
gest_hist       = deque(maxlen=3) # gesture voting (3 frames = ~0.1s delay for rock-solid stability)
prev_t          = time.time()
flash_txt       = ""
flash_end       = 0.0
sd_txt          = ""   # scroll direction
sd_end          = 0.0
wa_txt          = ""   # window arrow
wa_end          = 0.0

# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def lm_px(hl, w, h):
    return [(int(l.x*w), int(l.y*h)) for l in hl.landmark]

def fingers_up(lm, nlm):
    """
    [thumb, index, middle, ring, pinky] — 1=extended, 0=curled.
    Uses distance-from-wrist for robustness against hand tilt/rotation.
    """
    # Thumb: distance from tip to pinky MCP base (nlm[17]) is larger when open
    th = 1 if ndist(nlm[4], nlm[17]) > ndist(nlm[3], nlm[17]) else 0
    
    # Other fingers: tip must be significantly further from wrist (nlm[0]) than its base MCP
    res = [th]
    for tip, base in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        d_tip  = ndist(nlm[tip], nlm[0])
        d_base = ndist(nlm[base], nlm[0])
        res.append(1 if d_tip > d_base * 1.15 else 0) # 15% further = extended
    return res

def pdist(a, b):
    return float(np.hypot(a[0]-b[0], a[1]-b[1]))

def ndist(a, b):
    return float(np.hypot(a.x-b.x, a.y-b.y))

def to_screen(nx, ny):
    sx = np.interp(nx, (FR, 1.0-FR), (0, screen_w))
    sy = np.interp(ny, (FR, 1.0-FR), (0, screen_h))
    return float(np.clip(sx, 0, screen_w)), float(np.clip(sy, 0, screen_h))

def move_cursor(tx, ty):
    """Pure EMA smooth move with dead zone — no velocity prediction (eliminates jitter)."""
    global smooth_x, smooth_y, cursor_init
    global prev_screen_x, prev_screen_y
    if not cursor_init:
        smooth_x, smooth_y = tx, ty
        cursor_init = True
        prev_screen_x, prev_screen_y = int(tx), int(ty)
        return
    a = 1.0 / SMOOTH
    smooth_x = a * tx + (1-a) * smooth_x
    smooth_y = a * ty + (1-a) * smooth_y
    # Dead zone — only move if cursor actually shifted enough
    if pdist((smooth_x, smooth_y), (prev_screen_x, prev_screen_y)) > DEAD_ZONE:
        pyautogui.moveTo(int(smooth_x), int(smooth_y))
        prev_screen_x, prev_screen_y = int(smooth_x), int(smooth_y)

def drag_cursor(tx, ty):
    """Faster EMA for drag (less lag)."""
    global smooth_x, smooth_y
    a = 1.0 / max(3, SMOOTH-3)
    smooth_x = a * tx + (1-a) * smooth_x
    smooth_y = a * ty + (1-a) * smooth_y
    pyautogui.dragTo(int(smooth_x), int(smooth_y), button='left')

def flash(txt, dur=1.0):
    global flash_txt, flash_end
    flash_txt = txt
    flash_end = time.time() + dur

def scr_dir(d):
    global sd_txt, sd_end
    sd_txt = d
    sd_end = time.time() + 0.5

def win_arrow(d):
    global wa_txt, wa_end
    wa_txt = d
    wa_end = time.time() + 1.0

def detect_win_sweep(hand_nx, now):
    """Returns 'right'|'left'|None based on hand movement."""
    global ws_last_t, ws_hist
    if now - ws_last_t < WS_COOL:
        return None
    ws_hist.append((now, hand_nx))
    cutoff = now - WS_WINDOW
    while ws_hist and ws_hist[0][0] < cutoff:
        ws_hist.popleft()
    if len(ws_hist) < 4:
        return None
    delta = ws_hist[-1][1] - ws_hist[0][1]
    if abs(delta) >= WS_DELTA:
        d = 'right' if delta > 0 else 'left'
        ws_last_t = now
        ws_hist.clear()
        return d
    return None

def detect_tab_sweep(hand_nx, now):
    """Specific sweep detection for tabs with its own cooldown."""
    global tab_sw_last_t, ws_hist
    if now - tab_sw_last_t < TS_COOL:
        return None
    # We share ws_hist with window switch since they don't happen together
    ws_hist.append((now, hand_nx))
    cutoff = now - WS_WINDOW
    while ws_hist and ws_hist[0][0] < cutoff:
        ws_hist.popleft()
    if len(ws_hist) < 4:
        return None
    delta = ws_hist[-1][1] - ws_hist[0][1]
    if abs(delta) >= WS_DELTA:
        d = 'right' if delta > 0 else 'left'
        tab_sw_last_t = now
        ws_hist.clear()
        return d
    return None

def take_screenshot():
    """Capture full screen and save to Desktop."""
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.exists(desktop):
        desktop = os.path.expanduser("~")
    fname = os.path.join(desktop, f"vmouse_shot_{int(time.time())}.png")
    shot = pyautogui.screenshot()
    shot.save(fname)
    print(f"\n[SCREENSHOT] Saved to: {fname}\n")
    return fname

def send_to_background(keys=None, scroll=0):
    """Focus the browser and send either hotkeys or a scroll event."""
    try:
        vm_hwnd = ctypes.windll.user32.FindWindowW(None, WIN_NAME)
        target = vm_hwnd
        for _ in range(5):
            target = ctypes.windll.user32.GetWindow(target, 2)
            if not target: break
            if ctypes.windll.user32.IsWindowVisible(target):
                break
        
        if target and target != vm_hwnd:
            ctypes.windll.user32.AllowSetForegroundWindow(-1)
            ctypes.windll.user32.SetForegroundWindow(target)
            time.sleep(0.02) # slightly longer for focus to settle
            
            if keys:
                pyautogui.hotkey(*keys)
            if scroll != 0:
                pyautogui.keyDown('ctrl')
                pyautogui.scroll(scroll)
                pyautogui.keyUp('ctrl')
            return
    except Exception:
        pass
    # Fallback to local
    if keys: pyautogui.hotkey(*keys)
    if scroll != 0: 
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(scroll)
        pyautogui.keyUp('ctrl')

def composite(frame, cnv):
    """Blend ink canvas onto camera frame."""
    gray     = cv2.cvtColor(cnv, cv2.COLOR_BGR2GRAY)
    _, mask  = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    return cv2.add(cv2.bitwise_and(frame, frame, mask=mask_inv),
                   cv2.bitwise_and(cnv,   cnv,   mask=mask))

def hud(img, mode, fps, draw_on, now):
    h, w = img.shape[:2]

    # ─ Status bar background
    cv2.rectangle(img, (0, h-46), (w, h), (10, 10, 10), -1)

    # Colour coding per gesture
    col = (80, 210, 80)          # default green — cursor moving
    if "No hand"    in mode: col = (60,  60,  60)
    if "PAUSE"      in mode: col = (70,  70, 230)
    if "FIST"       in mode: col = (90,  90,  90)
    if "DRAW"       in mode: col = (40, 140, 255)
    if "ERASE"      in mode: col = (50, 200, 200)
    if "PINCHING"   in mode: col = (180, 180, 50)
    if "CLICK"      in mode: col = (50, 180, 255)
    if "DOUBLE"     in mode: col = (50, 230, 255)
    if "DRAG"       in mode: col = (200, 70, 160)
    if "SCROLL"     in mode: col = (200, 200, 50)
    if "WINDOW"     in mode: col = (230, 140, 40)
    if "SCREENSHOT" in mode: col = (50, 220, 50)
    if "RIGHT"      in mode: col = (60,  60, 220)
    if "MIDDLE"     in mode: col = (50, 180, 130)
    if "VOL"        in mode: col = (130, 80, 220)
    if "ZOOM"       in mode: col = (80, 200, 180)
    if "TAB"        in mode: col = (200, 100, 60)

    # Gesture dot + label
    cv2.circle(img, (14, h-23), 7, col, -1)
    cv2.putText(img, mode, (28, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)

    # FPS (colour: green=good, yellow=ok, red=slow)
    fps_col = (50, 200, 50) if fps >= 25 else (50, 200, 200) if fps >= 15 else (50, 50, 200)
    cv2.putText(img, f"FPS {fps}", (w-82, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, fps_col, 1)

    # Draw mode badge
    if draw_on:
        cv2.rectangle(img, (w//2-32, h-42), (w//2+32, h-6), (30, 100, 200), -1)
        cv2.putText(img, "DRAW", (w//2-24, h-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    # Profile badge (top-left corner)
    pname = PROFILES[active_profile]['name']
    cv2.rectangle(img, (0, 0), (130, 24), (20, 20, 20), -1)
    cv2.putText(img, f"[{pname}] 1/2/3", (4, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 160, 160), 1, cv2.LINE_AA)

    # Active zone border
    cv2.rectangle(img,
                  (int(w*FR), int(h*FR)),
                  (int(w*(1-FR)), h-50),
                  (45, 45, 45), 1)

    # Flash label (top-right, fades out)
    if now < flash_end:
        alpha = min(1.0, (flash_end - now) / 0.35)
        c = (int(230*alpha), int(230*alpha), int(230*alpha))
        cv2.putText(img, flash_txt, (w-270, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, c, 2, cv2.LINE_AA)

    # Scroll direction arrow
    if now < sd_end and sd_txt:
        cx2, cy2 = w//2, 82
        s, ac = 26, (200, 200, 50)
        if sd_txt == "UP":
            pts = np.array([[cx2,cy2-s],[cx2-s,cy2+s//2],[cx2+s,cy2+s//2]], np.int32)
        elif sd_txt == "DOWN":
            pts = np.array([[cx2,cy2+s],[cx2-s,cy2-s//2],[cx2+s,cy2-s//2]], np.int32)
        elif sd_txt == "LEFT":
            pts = np.array([[cx2-s,cy2],[cx2+s//2,cy2-s],[cx2+s//2,cy2+s]], np.int32)
        else:
            pts = np.array([[cx2+s,cy2],[cx2-s//2,cy2-s],[cx2-s//2,cy2+s]], np.int32)
        cv2.fillPoly(img, [pts], ac)

    # Window switch arrow
    if now < wa_end and wa_txt:
        wy = 56
        wc = (230, 140, 40)
        if wa_txt == "RIGHT":
            cv2.arrowedLine(img, (w//2-100, wy), (w//2+100, wy), wc, 3, tipLength=0.28)
            cv2.putText(img, "Next Window (Alt+Tab)",
                        (w//2-100, wy+26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, wc, 2, cv2.LINE_AA)
        else:
            cv2.arrowedLine(img, (w//2+100, wy), (w//2-100, wy), wc, 3, tipLength=0.28)
            cv2.putText(img, "Prev Window (Alt+Shift+Tab)",
                        (w//2-110, wy+26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, wc, 2, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════════╗
║       Virtual Mouse — Optimized Showcase Version                 ║
╠══════════════════════════════════════════════════════════════════╣
║  Index only              → Move cursor (smooth)                  ║
║  Index+thumb pinch       → Left click  (hold 0.14s)              ║
║  Pinch hold 0.70s        → Drag                                  ║
║  Two quick pinches       → Double click                          ║
║  Thumb sideways (far)    → Right click (pinch-guarded)           ║
║  Index+middle + gap      → Scroll up/down/left/right             ║
║  3 fingers up            → Middle click                          ║
║  All 5 open              → Pause cursor                          ║
║  Hand Wave (Hi! gesture) → Alt+Tab window switch                 ║
║  Thumb+pinky tips touch  → Screenshot → saved to Desktop         ║
║    (index+middle+ring UP)                                        ║
║  Thumb+middle pinch      → Volume Down  (index/ring/pinky UP)    ║
║  Thumb+middle spread     → Volume Up    (index/ring/pinky UP)    ║
║  Index+middle close      → Zoom Out (Ctrl +-)                    ║
║  Index+middle spread     → Zoom In  (Ctrl +=)                    ║
║  Ring+pinky up + sweep   → Browser Tab switch (Ctrl+Tab)         ║
║  Hand 90° tilted         → Erase (draw mode)                     ║
║  D → Draw  | C → Clear | Q/ESC → Quit                           ║
║  1 → Default | 2 → Presentation | 3 → Gaming profile             ║
╚══════════════════════════════════════════════════════════════════╝
""")

# ═══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════
WIN_NAME        = "Virtual Mouse"
_win_setup_done = False   # flag so always-on-top setup runs only once
while True:
    ok, img = cap.read()
    if not ok:
        continue

    img  = cv2.flip(img, 1)
    h, w = img.shape[:2]

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    now = time.time()
    dt  = now - prev_t
    prev_t = now
    fps_hist.append(1.0 / max(dt, 0.001))
    fps = int(sum(fps_hist) / len(fps_hist))

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = hands.process(rgb)
    rgb.flags.writeable = True

    mode = "No hand"

    if res.multi_hand_landmarks:
        hl  = res.multi_hand_landmarks[0]
        nlm = hl.landmark
        lm  = lm_px(hl, w, h)

        mp_draw.draw_landmarks(img, hl, mp_hands.HAND_CONNECTIONS)

        f    = fingers_up(lm, nlm)
        n_up = sum(f)

        ix, iy = lm[8]
        tx, ty = lm[4]
        nx_i   = nlm[8].x
        ny_i   = nlm[8].y
        no_hand_frames = 0  # reset counter — hand is present

        # ── Gesture flags ────────────────────────────────────────────────
        pinch_n = ndist(nlm[4], nlm[8])

        # Screenshot: thumb-pinky pinch + index/middle/ring UP
        f_shot = (f[1]==1 and f[2]==1 and f[3]==1 and ndist(nlm[4], nlm[20]) < SS_DIST)

        # Erase/Palm: All fingers open (4 or 5)
        f_palm = (n_up >= 4)

        # --- ZOOM vs SCROLL Toggles ---
        # Instead of f[0], we use the distance from thumb tip to index base (MCP)
        # Tucked thumb (dist < 0.08) = Zoom | Outstretched thumb = Scroll
        th_tucked = ndist(nlm[4], nlm[5]) < 0.08

        # Finger straightness: tip must be above DIP joint (in image y)
        idx_straight = lm[8][1] < lm[7][1]   # index tip above DIP = straight
        mid_straight = lm[12][1] < lm[11][1]  # middle tip above DIP = straight

        # Zoom: Index+Middle STRAIGHT UP, Ring+Pinky DOWN, Thumb TUCKED
        f_zoom = (f[1]==1 and f[2]==1 and f[3]==0 and f[4]==0 and th_tucked
                  and idx_straight and mid_straight)

        # ── FIST SCROLL ────────────────────────────────────────────────
        # All 4 fingers curled (tips NOT much further from wrist than MCP)
        # Thumb NOT pointing sideways (to avoid conflict with vol_dn)
        all_curled = (ndist(nlm[8],  nlm[0]) < ndist(nlm[5],  nlm[0]) * 1.4 and
                      ndist(nlm[12], nlm[0]) < ndist(nlm[9],  nlm[0]) * 1.4 and
                      ndist(nlm[16], nlm[0]) < ndist(nlm[13], nlm[0]) * 1.4 and
                      ndist(nlm[20], nlm[0]) < ndist(nlm[17], nlm[0]) * 1.4)
        f_fist_scroll = (all_curled and abs(nlm[4].x - nlm[0].x) <= 0.14)

        f_scroll = False  # old scroll disabled

        # ── VOLUME DOWN fix ────────────────────────────────────────────
        # Thumb sideways (large x-displacement from wrist) + all 4 fingers curled
        f_vol_dn = (abs(nlm[4].x - nlm[0].x) > 0.14 and all_curled)

        # Tab switch: Ring+Pinky UP, others DOWN
        f_tab = (f[3]==1 and f[4]==1 and f[0]==0 and f[1]==0 and f[2]==0)

        # Vol UP: Index+Pinky UP, others DOWN
        f_vol_up = (f[1]==1 and f[4]==1 and f[0]==0 and f[2]==0 and f[3]==0)


        # Right Click: Thumb far sideways, others DOWN
        f_right = (abs(nlm[4].x - nlm[5].x) > RC_THUMB_H and pinch_n > RC_PINCH_GUARD and f[1]==0 and f[2]==0)

        # Middle Click: Index+Middle+Ring UP, others DOWN
        f_mid = (f[1]==1 and f[2]==1 and f[3]==1 and f[4]==0 and f[0]==0)

        # Cursor/Click/Drag: Index UP only (Thumb can be up or down for clicking)
        f_cursor_base = (f[1]==1 and f[2]==0 and f[3]==0 and f[4]==0)

        # ── DETERMINING RAW MODE ─────────────────────────────────────────
        raw_m = "NONE"
        if f_shot:         raw_m = "SHOT"
        elif f_palm:       raw_m = "PALM"
        elif f_zoom:       raw_m = "ZOOM"
        elif f_fist_scroll: raw_m = "SCROLL"  # fist up/down = scroll
        elif f_tab:        raw_m = "TAB"
        elif f_vol_up:     raw_m = "VOL_UP"
        elif f_vol_dn:     raw_m = "VOL_DN"
        elif f_right:      raw_m = "RIGHT"
        elif f_mid:        raw_m = "MIDDLE"
        elif f_cursor_base: raw_m = "CURSOR"
        
        # Temporal smoothing: voting on the last 3 frames
        gest_hist.append(raw_m)
        voted_m = max(set(gest_hist), key=list(gest_hist).count)

        # ── EXECUTION ────────────────────────────────────────────────────
        
        # 1. SCREENSHOT
        if voted_m == "SHOT":
            mode = "SCREENSHOT"
            scroll_on = False
            if not ss_lock and now - ss_last_t > SS_COOL:
                ss_lock  = True
                ss_last_t = now
                take_screenshot()
                flash("Screenshot → Desktop!")

        # 2. PALM (Erase / Window Switch)
        elif voted_m == "PALM":
            scroll_on = False
            if draw_mode:
                mode = "ERASE"
                px = int((nlm[0].x + nlm[5].x + nlm[9].x + nlm[13].x + nlm[17].x) / 5 * w)
                py = int((nlm[0].y + nlm[5].y + nlm[9].y + nlm[13].y + nlm[17].y) / 5 * h)
                cv2.circle(canvas, (px, py), 40, (0, 0, 0), -1)
                cv2.circle(img,    (px, py), 40, (255, 255, 255), 3)
            else:
                sw = detect_win_sweep(nlm[9].x, now)
                if sw:
                    last_sw_t = now
                    if not alt_down:
                        pyautogui.keyDown('alt')
                        alt_down = True
                    
                    if sw == 'right':
                        pyautogui.press('tab')
                        flash("Cycle Next >>")
                        win_arrow("RIGHT")
                    else:
                        pyautogui.hotkey('shift', 'tab')
                        flash("<< Cycle Prev")
                        win_arrow("LEFT")
                else:
                    mode = "PAUSED / WAVE"
                    # Auto-release Alt if we stop waving for a bit
                    if alt_down and now - last_sw_t > SW_RELEASE_T:
                        pyautogui.keyUp('alt')
                        alt_down = False
                        flash("Window Selected")

        # 3. ZOOM
        elif voted_m == "ZOOM":
            scroll_on = False
            im_dist = ndist(nlm[8], nlm[12])
            if zoom_ref_dist < 0:
                zoom_ref_dist = im_dist
                mode = "ZOOM READY"
            else:
                delta = im_dist - zoom_ref_dist
                if now > zoom_cd:
                    if delta > 0.020:              # spread → Zoom In
                        send_to_background(scroll=150)
                        flash("Zoom In +")
                        zoom_cd = now + 0.35       # Faster zoom with scroll
                        zoom_ref_dist = im_dist
                    elif delta < -0.015:           # pinch → Zoom Out
                        send_to_background(scroll=-150)
                        flash("Zoom Out -")
                        zoom_cd = now + 0.35
                        zoom_ref_dist = im_dist
                    mode = "ZOOMING"

        # 4. SCROLL — Fist joystick style
        elif voted_m == "SCROLL":
            cur_y = nlm[0].y
            if fist_anchor_y < 0:
                # First frame in fist mode — capture anchor position
                fist_anchor_y = cur_y
                mode = "SCROLL READY"
            else:
                offset = fist_anchor_y - cur_y  # positive = fist moved UP
                if abs(offset) > FIST_DEAD:
                    # Continuous auto-scroll at timed intervals
                    if now - fist_scroll_t >= FIST_RATE:
                        clks = int(offset * FIST_SCROLL_SENS)
                        clks = max(-12, min(12, clks))  # cap per step
                        if clks:
                            pyautogui.scroll(clks)
                            scr_dir("UP" if clks > 0 else "DOWN")
                        fist_scroll_t = now
                    mode = "SCROLL ↑" if offset > 0 else "SCROLL ↓"
                else:
                    mode = "SCROLL READY •"   # in dead zone, not scrolling

        # 5. TAB SWITCH
        elif voted_m == "TAB":
            sw = detect_tab_sweep(nlm[9].x, now)
            if sw:
                last_sw_t = now
                if not ctrl_down:
                    # Focus browser first, then hold Ctrl
                    send_to_background(keys=None) 
                    pyautogui.keyDown('ctrl')
                    ctrl_down = True
                
                if sw == 'right':
                    pyautogui.press('tab')
                    flash("Next Tab →")
                else:
                    pyautogui.hotkey('shift', 'tab')
                    flash("← Prev Tab")
            else:
                mode = "TAB READY"
                if ctrl_down and now - last_sw_t > SW_RELEASE_T:
                    pyautogui.keyUp('ctrl')
                    ctrl_down = False
                    flash("Tab Selected")

        # 6. VOLUME
        elif voted_m == "VOL_UP":
            if now > vol_cd:
                pyautogui.press('volumeup')
                flash("Volume Up")
                vol_cd = now + VOL_COOL
            mode = "VOL UP"
        elif voted_m == "VOL_DN":
            if now > vol_cd:
                pyautogui.press('volumedown')
                flash("Volume Down")
                vol_cd = now + VOL_COOL
            mode = "VOL DOWN"

        # 7. MIDDLE CLICK
        elif voted_m == "MIDDLE":
            if now > middle_cd:
                pyautogui.middleClick()
                middle_cd = now + 0.7
                flash("Middle Click")
            mode = "MIDDLE CLICK"

        # 7.5 RIGHT CLICK
        elif voted_m == "RIGHT":
            if now > right_cd:
                pyautogui.rightClick()
                right_cd = now + RC_COOL
                flash("Right Click")
            mode = "RIGHT CLICK"

        # 8. CURSOR / CLICK / DRAG
        elif voted_m == "CURSOR":
            if draw_mode:
                mode = "DRAW"
                ix, iy = lm[8]
                if prev_draw_x == 0 and prev_draw_y == 0:
                    prev_draw_x, prev_draw_y = ix, iy
                cv2.line(canvas, (prev_draw_x, prev_draw_y), (ix, iy), DRAW_COLOR, DRAW_THICK, cv2.LINE_AA)
                prev_draw_x, prev_draw_y = ix, iy
            else:
                tgt_x, tgt_y = to_screen(nlm[8].x, nlm[8].y)
                if click_state == "OPEN":
                    if pinch_n < CLICK_DIST_N:
                        click_state, click_down_t = "PRESSING", now
                    else:
                        move_cursor(tgt_x, tgt_y)
                        mode = "CURSOR"
                elif click_state == "PRESSING":
                    held = now - click_down_t
                    if pinch_n >= CLICK_DIST_N:
                        click_state = "OPEN"
                        if held >= CLICK_CONFIRM and now - last_any_click > CLICK_COOL:
                            if now - last_click_t < DBL_TIME:
                                pyautogui.doubleClick()
                                flash("Double Click")
                                last_click_t = 0
                            else:
                                pyautogui.click()
                                flash("Click")
                                last_click_t = now
                            last_any_click = now
                    elif held >= DRAG_HOLD:
                        click_state, drag_active = "DRAGGING", True
                        pyautogui.mouseDown(button='left')
                        flash("Dragging")
                    else:
                        move_cursor(tgt_x, tgt_y)
                        mode = "PINCHING"
                elif click_state == "DRAGGING":
                    if pinch_n >= CLICK_DIST_N:
                        click_state, drag_active = "OPEN", False
                        pyautogui.mouseUp(button='left')
                        flash("Dropped")
                    else:
                        drag_cursor(tgt_x, tgt_y)
                        mode = "DRAG"

        else:
            scroll_on = False
            prev_draw_x, prev_draw_y = 0, 0

        # ── Reset locks ──────────────────────────────────────────────────
        if not f_shot:
            ss_lock = False
        
        # Global auto-release for safety when hand leaves gesture mode
        if not f_palm and alt_down:
            pyautogui.keyUp('alt')
            alt_down = False
        if not f_tab and ctrl_down:
            pyautogui.keyUp('ctrl')
            ctrl_down = False

        # Release fist state when hand opens
        if not (n_up == 0 or (n_up==1 and f[0]==1 and not f_right)):
            if tab_drag:
                pyautogui.mouseUp(button='left')
                tab_drag = False
            fist_held    = False
            fist_start_t = 0.0
        # Track previous scroll state for anchor init fix
        prev_scroll_on = f_scroll
        # Reset scroll accumulator when not scrolling
        if not f_scroll:
            scroll_on    = False
            scroll_acc_y = 0.0
            scroll_acc_x = 0.0
        # Reset zoom reference when not in zoom pose
        if not f_zoom:
            zoom_ref_dist = -1.0
        
        # Reset fist scroll anchor when not in fist scroll mode
        if voted_m != "SCROLL":
            fist_anchor_y = -1.0

        # Reset tab sweep hist when not in tab pose
        if not f_tab:
            pass # ws_hist is shared, no need to clear here as it might break window switch

    else:
        # No hand
        mode = "No hand"
        no_hand_frames += 1
        # Reset cursor smoothing after 5 frames of no hand — prevents snap on reappearance
        if no_hand_frames >= 5:
            cursor_init = False
        if drag_active:
            pyautogui.mouseUp(button='left')
            drag_active = False
        if tab_drag:
            pyautogui.mouseUp(button='left')
            tab_drag = False
        if click_state == "DRAGGING":
            pyautogui.mouseUp(button='left')
        click_state      = "OPEN"
        scroll_on        = False
        prev_scroll_on   = False
        scroll_acc_y     = 0.0
        scroll_acc_x     = 0.0
        fist_held        = False
        fist_start_t     = 0.0
        ss_lock          = False
        prev_draw_x      = 0
        prev_draw_y      = 0
        ws_hist.clear()

    # ── Composite + HUD ───────────────────────────────────────────────────
    img = composite(img, canvas)
    hud(img, mode, fps, draw_mode, now)
    cv2.imshow(WIN_NAME, img)

    # One-time window config after the first real frame is rendered:
    # always-on-top + non-focus-stealing (hotkeys go to your browser, not here)
    if not _win_setup_done:
        try:
            cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        try:
            _hwnd2 = ctypes.windll.user32.FindWindowW(None, WIN_NAME)
            if _hwnd2:
                _es2 = ctypes.windll.user32.GetWindowLongW(_hwnd2, -20)
                ctypes.windll.user32.SetWindowLongW(
                    _hwnd2, -20, _es2 | 0x00000008)   # TOPMOST only — keeps keyboard focus
        except Exception:
            pass
        _win_setup_done = True

    # ── Keys ──────────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    elif key == ord('d'):
        draw_mode = not draw_mode
        flash(f"Draw {'ON' if draw_mode else 'OFF'}")
        print(f"Draw: {'ON' if draw_mode else 'OFF'}")
    elif key == ord('c'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        flash("Canvas Cleared")
        print("Canvas cleared")
    elif key in (ord('1'), ord('2'), ord('3')):
        # Profile switch — live update of tuning constants
        k = chr(key)
        p = PROFILES[k]
        active_profile = k
        SMOOTH        = p['SMOOTH']
        CLICK_DIST_N  = p['CLICK_DIST_N']
        CLICK_CONFIRM = p['CLICK_CONFIRM']
        DRAG_HOLD     = p['DRAG_HOLD']
        SC_SENS       = p['SC_SENS']
        flash(f"Profile: {p['name']}")
        print(f"[PROFILE] Switched to: {p['name']}")

# ── Cleanup ────────────────────────────────────────────────────────────────────
if drag_active or tab_drag:
    pyautogui.mouseUp(button='left')
cap.release()
cv2.destroyAllWindows()
print("Exited.")
