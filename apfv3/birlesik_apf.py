#!/usr/bin/env python3
"""
birlesik_apf.py — ZED 2i + YOLO + APF  |  Şerit + Engel Kaçınma
═══════════════════════════════════════════════════════════════════════════════
Akış:
  1. ZED 2i'den RGB görüntü + XYZ point cloud al  (tek grab — gereksiz ölçüm yok)
  2. YOLO (best.pt) ile şerit tespiti → PC'den gerçek dünya konumu (x_m, z_m)
  3. Point cloud ROI filtresi → engel noktaları (zemin/tavan kırpılır)
  4. APF:  ileri çekim  +  şeritlerden itme  +  engellerden itme
  5. Bileşke kuvvet → diferansiyel PWM (paletli tank)
  6. Arduino'ya  "L<sol> R<sag>\n"  gönder

Koordinat sistemi (LEFT_HANDED_Y_UP):
    X → sağ (+)   |   Y → yukarı (+)   |   Z → ileri (+)

Çalıştırma:
  python birlesik_apf.py                  # ZED 2i + Arduino
  python birlesik_apf.py --no-serial      # Arduino olmadan test
  python birlesik_apf.py --no-gui         # Headless / SSH
  python birlesik_apf.py --stream         # MJPEG stream (tarayıcıdan izle)
"""

import math
import os
import time
import argparse
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ─── Seri port ─────────────────────────────────────────────────────────────
SERIAL_PORT = "/dev/ttyCH341USB0"
SERIAL_BAUD = 115200

# ─── Sürüş parametreleri ───────────────────────────────────────────────────
BASE_SPEED  = 120     # düz gidişte her iki palet PWM (0-255)
MAX_PWM     = 180    # maks PWM
STEER_LIMIT = 55     # dönüş farkı sınırı (BASE ± STEER_LIMIT)

# ─── APF — Şerit kaçınma ───────────────────────────────────────────────────
K_ATT       = 1.0    # ileri çekim katsayısı
K_REP_SERIT = 9.0    # şerit itme katsayısı (grid ortalaması için kalibre)
D0_SERIT    = 4.5    # şerit itme etki yarıçapı (m)

# ─── APF — Engel kaçınma (Point Cloud) ────────────────────────────────────
K_REP_ENGEL = 7.0    # engel itme katsayısı
D_INFLUENCE = 3.5    # engel itme etki yarıçapı (m)
REP_CAP     = 5.0    # tek nokta max kuvveti
MAX_PTS     = 2000   # alt-örnekleme eşiği (performans)

# ─── Point Cloud ROI ───────────────────────────────────────────────────────
Z_MIN  = 0.30    # gürültü sınırı (m)
Z_MAX  = D_INFLUENCE
X_HALF = 1.60    # yatay etki genişliği ±m
Y_LOW  = -0.30   # zemin kırpma (kamera yüksekliğine göre ayarla)
Y_HIGH =  1.20   # tavan kırpma (m)

LANE_THRESH  = 170   # beyaz şerit piksel eşiği (gri tonlama, 0-255)
MAX_LANE_PTS = 30    # bbox başına max örneklenen şerit noktası (performans)


# ══════════════════════════════════════════════════════════════════════════
# Frame tamponları  (ZED grab thread  ↔  ana thread)
# ══════════════════════════════════════════════════════════════════════════
_zed_frame: "np.ndarray | None" = None
_zed_pc:    "np.ndarray | None" = None   # (H, W, 3) float32 — XYZ
_zed_lock   = threading.Lock()

# MJPEG stream tamponu
_stream_frame: bytes = b""
_stream_lock   = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════
# ZED — açma ve grab thread
# ══════════════════════════════════════════════════════════════════════════

def open_zed():
    """ZED 2i'yi açar; sadece LEFT görüntü + XYZ ölçümü kullanılır."""
    import pyzed.sl as sl

    zed = sl.Camera()
    p = sl.InitParameters()
    p.camera_resolution         = sl.RESOLUTION.HD720
    p.camera_fps                = 30
    p.depth_mode                = sl.DEPTH_MODE.PERFORMANCE
    p.coordinate_units          = sl.UNIT.METER
    p.coordinate_system         = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP  # Z ileri
    p.depth_minimum_distance    = 0.2
    p.enable_right_side_measure = False   # sağ kamera ölçümü yok → kaynak tasarrufu
    p.sdk_verbose               = False   # gereksiz log yok

    err = zed.open(p)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.exit(f"[HATA] ZED açılamadı: {err}")

    zed.disable_positional_tracking()     # navigasyon takibi kapalı → hafiflik

    print("[INFO] ZED 2i açıldı  (PERFORMANCE, HD720, LEFT_HANDED_Y_UP)")

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold         = 50
    runtime.texture_confidence_threshold = 100

    image_mat = sl.Mat()   # RGB
    pc_mat    = sl.Mat()   # XYZ point cloud

    return zed, runtime, image_mat, pc_mat


def _grab_thread(zed, runtime, image_mat, pc_mat, running: list) -> None:
    """ZED'i ayrı thread'de sürekli okur.
    Alınan ölçüm: VIEW.LEFT (RGB) + MEASURE.XYZ  — başka hiçbir şey yok."""
    import pyzed.sl as sl
    while running[0]:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        zed.retrieve_image(image_mat, sl.VIEW.LEFT)       # RGB
        zed.retrieve_measure(pc_mat,  sl.MEASURE.XYZ)     # tek ölçüm

        frame = image_mat.get_data()[:, :, :3].copy()
        pc    = pc_mat.get_data()[:, :, :3].astype(np.float32)  # (H,W,3) XYZ

        with _zed_lock:
            global _zed_frame, _zed_pc
            _zed_frame = frame
            _zed_pc    = pc


def get_zed_data():
    with _zed_lock:
        return _zed_frame, _zed_pc


# ══════════════════════════════════════════════════════════════════════════
# MJPEG stream sunucusu
# ══════════════════════════════════════════════════════════════════════════

def _update_stream(frame: np.ndarray) -> None:
    global _stream_frame
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        with _stream_lock:
            _stream_frame = buf.tobytes()


class _MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with _stream_lock:
                    data = _stream_frame
                if data:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + data + b"\r\n"
                    )
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass


def start_stream(port: int = 8080) -> None:
    srv = HTTPServer(("0.0.0.0", port), _MJPEGHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[STREAM] http://<ip>:{port}  adresinden izleyebilirsin")


# ══════════════════════════════════════════════════════════════════════════
# YOLO — Şerit tespiti + PC mesafe
# ══════════════════════════════════════════════════════════════════════════

def detect_serits(model: YOLO, frame: np.ndarray, pc: np.ndarray,
                  conf: float = 0.25):
    """
    YOLO ile şerit tespiti + bbox içinde beyaz piksel maskeleme.
    Sadece gerçek şerit çizgisi üzerindeki piksellerin dünya koordinatları
    itici kaynak olarak döner. Asfalt/arka plan noktaları filtrelenir.

    Döner:
        serit_pts : [(x_m, z_m), ...]   şerit piksellerinden örneklenen noktalar
        boxes     : [(x1,y1,x2,y2, score, z_m), ...]
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = model(frame, verbose=False, conf=conf)[0]
    serit_pts = []
    boxes     = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        score = float(box.conf[0].cpu())

        # bbox içindeki beyaz pikselleri maskele
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        mask = roi > LANE_THRESH
        ys_local, xs_local = np.where(mask)
        if len(ys_local) == 0:
            continue

        # frame koordinatlarına çevir
        ys_frame = ys_local + y1
        xs_frame = xs_local + x1

        # çok nokta varsa alt-örnekle (performans)
        if len(ys_frame) > MAX_LANE_PTS:
            idx = np.random.choice(len(ys_frame), MAX_LANE_PTS, replace=False)
            ys_frame = ys_frame[idx]
            xs_frame = xs_frame[idx]

        # point cloud'dan dünya koordinatlarını oku
        box_pts = []
        for py, px in zip(ys_frame, xs_frame):
            xw = float(pc[py, px, 0])
            zw = float(pc[py, px, 2])
            if math.isfinite(xw) and math.isfinite(zw) and 0.1 < zw < 10.0:
                box_pts.append((xw, zw))

        if not box_pts:
            continue

        z_med = float(np.median([p[1] for p in box_pts]))
        boxes.append((x1, y1, x2, y2, score, z_med))
        serit_pts.extend(box_pts)

    return serit_pts, boxes


# ══════════════════════════════════════════════════════════════════════════
# Point Cloud — Engel tespiti (ROI filtresi)
# ══════════════════════════════════════════════════════════════════════════

def detect_obstacles(pc: np.ndarray) -> np.ndarray:
    """
    PC ROI filtresi: önde, zemin/tavan dışındaki engel noktaları.
    Döner: (N, 3) float32 — XYZ
    """
    pts = pc.reshape(-1, 3)

    valid = np.isfinite(pts).all(axis=1)
    pts   = pts[valid]
    if len(pts) == 0:
        return pts

    roi = (
        (pts[:, 2] >= Z_MIN)   & (pts[:, 2] <= Z_MAX)  &
        (pts[:, 0] >= -X_HALF) & (pts[:, 0] <= X_HALF) &
        (pts[:, 1] >= Y_LOW)   & (pts[:, 1] <= Y_HIGH)
    )
    pts = pts[roi]

    n = len(pts)
    if n > MAX_PTS:
        idx = np.random.choice(n, MAX_PTS, replace=False)
        pts = pts[idx]

    return pts


# ══════════════════════════════════════════════════════════════════════════
# Yapay Potansiyel Alanlar — birleşik kuvvet
# ══════════════════════════════════════════════════════════════════════════

def apf_force(serit_pts: list, obs_pts: np.ndarray):
    """
    Birleşik APF kuvveti (LEFT_HANDED_Y_UP: X sağ, Z ileri).

    Fx > 0 → sağa dön   |   Fx < 0 → sola dön
    Fz daima > 0 (ileri) — geri gidiş engellenir.
    """
    Fx, Fz = 0.0, K_ATT

    # ── Şeritlerden itme (çoklu nokta, vektörize) ────────────────────
    if serit_pts:
        sp = np.asarray(serit_pts, dtype=np.float64)
        xm, zm = sp[:, 0], sp[:, 1]
        mask = (zm > 0) & (zm <= D0_SERIT)
        xm, zm = xm[mask], zm[mask]
        if len(xm) > 0:
            d = np.hypot(xm, zm)
            good = d > 1e-4
            xm, zm, d = xm[good], zm[good], d[good]
            if len(d) > 0:
                d_safe = np.maximum(d, 0.05)
                mag = K_REP_SERIT * (1.0 / d_safe - 1.0 / D0_SERIT) / (d ** 2)
                n = float(len(d))
                Fx -= np.sum(mag * (xm / d)) / n
                Fz -= np.sum(mag * (zm / d)) / n

    # ── Engellerden itme (vektörize) ─────────────────────────────────
    if len(obs_pts) > 0:
        ox = obs_pts[:, 0].astype(np.float64)
        oz = obs_pts[:, 2].astype(np.float64)
        d = np.hypot(ox, oz)
        mask = (d > 1e-4) & (d <= D_INFLUENCE)
        ox, oz, d = ox[mask], oz[mask], d[mask]
        if len(d) > 0:
            mag = np.minimum(
                K_REP_ENGEL * (1.0 / d - 1.0 / D_INFLUENCE) / (d * d),
                REP_CAP,
            )
            n = float(len(d))
            Fx -= np.sum(mag * (ox / d)) / n
            Fz -= np.sum(mag * (oz / d)) / n

    Fz = max(Fz, 0.05)
    return Fx, Fz


# ══════════════════════════════════════════════════════════════════════════
# Kuvvet → PWM
# ══════════════════════════════════════════════════════════════════════════

def force_to_pwm(Fx: float, Fz: float):
    """
    atan2(Fx, Fz) → direksiyon açısı → diferansiyel PWM.
    θ > 0 (sağa kuvvet) → sol hızlı, sağ yavaş → sağa dönüş.
    """
    theta = math.atan2(Fx, max(Fz, 0.05))
    steer = int(STEER_LIMIT * math.sin(theta))
    steer = max(-STEER_LIMIT, min(STEER_LIMIT, steer))
    left  = max(-MAX_PWM, min(MAX_PWM, BASE_SPEED + steer))
    right = max(-MAX_PWM, min(MAX_PWM, BASE_SPEED - steer))
    return left, right


# ══════════════════════════════════════════════════════════════════════════
# Seri port
# ══════════════════════════════════════════════════════════════════════════

def send_cmd(ser, left: int, right: int) -> None:
    ser.write(f"L{right} R{left}\n".encode())


# ══════════════════════════════════════════════════════════════════════════
# Debug görselleştirme
# ══════════════════════════════════════════════════════════════════════════

def draw_debug(frame, boxes, n_serit_pts, n_obs, Fx, Fz, left_pwm, right_pwm) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Şerit kutuları — renk: kırmızı=yakın, yeşil=uzak
    for (x1, y1, x2, y2, score, z_m) in boxes:
        ratio = min(z_m / D0_SERIT, 1.0) if z_m > 0 else 1.0
        color = (0, int(255 * ratio), int(255 * (1 - ratio)))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{score:.2f} {z_m:.2f}m" if z_m > 0 else f"{score:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # APF kuvvet oku (ekran alt merkezinden)
    rx, ry = w // 2, h - 20
    cv2.circle(vis, (rx, ry), 8, (255, 100, 0), -1)
    norm = math.hypot(Fx, Fz) + 1e-9
    ax = rx + int(Fx / norm * 90)
    ay = ry - int(Fz / norm * 90)   # Fz ileri = ekranda yukarı
    cv2.arrowedLine(vis, (rx, ry), (ax, ay), (0, 255, 255), 3, tipLength=0.25)

    # PWM çubukları (sol/sağ)
    bh = 60
    lv = int(abs(left_pwm)  / MAX_PWM * bh)
    rv = int(abs(right_pwm) / MAX_PWM * bh)
    cv2.rectangle(vis, (10,      bh - lv), (30,      bh), (0, 200, 0), -1)
    cv2.rectangle(vis, (w - 30,  bh - rv), (w - 10,  bh), (0, 200, 0), -1)

    # Metin
    cv2.putText(vis, f"L:{left_pwm:+4d} R:{right_pwm:+4d}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"Serit:{len(boxes)}({n_serit_pts}pt)  Engel:{n_obs}  Fx:{Fx:+.2f} Fz:{Fz:+.2f}",
                (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Yönelim oku üstte
    cx_bar = w // 2
    tr = (left_pwm - right_pwm) / (2 * MAX_PWM + 1e-9)
    cv2.arrowedLine(vis, (cx_bar, 20), (int(cx_bar + tr * 120), 20),
                    (255, 200, 0), 3, tipLength=0.3)

    return vis


# ══════════════════════════════════════════════════════════════════════════
# Ana döngü
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="APF Şerit + Engel Kaçınma — ZED 2i + Paletli Tank")
    parser.add_argument("--weights",     default="best.pt")
    parser.add_argument("--port",        default=SERIAL_PORT)
    parser.add_argument("--baud",        type=int,   default=SERIAL_BAUD)
    parser.add_argument("--no-serial",   action="store_true")
    parser.add_argument("--no-gui",      action="store_true")
    parser.add_argument("--stream",      action="store_true")
    parser.add_argument("--stream-port", type=int,   default=8080)
    parser.add_argument("--conf",        type=float, default=0.25,
                        help="YOLO güven eşiği")
    args = parser.parse_args()

    running = [True]
    signal.signal(signal.SIGINT,  lambda *_: running.__setitem__(0, False))
    signal.signal(signal.SIGTERM, lambda *_: running.__setitem__(0, False))

    # ── Ekran kontrolü — display yoksa GUI otomatik kapatılır ─────────────
    if not args.no_gui and not os.environ.get("DISPLAY"):
        print("[INFO] DISPLAY bulunamadı — otomatik --no-gui moduna geçildi")
        args.no_gui = True

    # headless modda OpenCV'nin Qt/xcb plugin'ini yüklememesi için
    if args.no_gui:
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "0")
        cv2.setNumThreads(1)

    # ── YOLO ──────────────────────────────────────────────────────────────
    print(f"[INFO] YOLO yükleniyor: {args.weights}")
    model = YOLO(args.weights)
    model.fuse()
    if torch.cuda.is_available():
        model.to('cuda')
        model.overrides['half'] = True
        torch.backends.cudnn.benchmark = True
        print(f"[INFO] YOLO — GPU ({torch.cuda.get_device_name(0)}) + FP16")
    else:
        print("[WARN] CUDA bulunamadı — YOLO CPU'da çalışacak (yavaş)")

    # ── ZED ───────────────────────────────────────────────────────────────
    print("[INFO] ZED 2i başlatılıyor...")
    zed, runtime, image_mat, pc_mat = open_zed()
    t_grab = threading.Thread(
        target=_grab_thread,
        args=(zed, runtime, image_mat, pc_mat, running),
        daemon=True,
    )
    t_grab.start()
    print("[INFO] ZED grab thread başladı")

    # ── Seri port ─────────────────────────────────────────────────────────
    ser = None
    if not args.no_serial:
        import serial
        print(f"[INFO] Seri port: {args.port} @ {args.baud}")
        ser = serial.Serial(args.port, args.baud, timeout=1)
        time.sleep(2)
        resp = ser.readline().decode(errors="ignore").strip()
        print(f"[ARDUINO] {resp or 'hazır'}")
    else:
        print("[INFO] Seri port devre dışı (--no-serial)")

    if args.stream:
        start_stream(args.stream_port)

    print("[INFO] Çalışıyor. Çıkmak için Ctrl+C.\n")

    t_prev  = time.time()
    fps_avg = 0.0

    try:
        while running[0]:
            # ── Kare al ───────────────────────────────────────────────────
            frame, pc = get_zed_data()
            if frame is None:
                time.sleep(0.01)
                continue

            # ── Şerit tespiti (YOLO + PC mesafe) ──────────────────────────
            serit_pts, boxes = detect_serits(model, frame, pc, args.conf)

            # ── Engel tespiti (PC ROI) ────────────────────────────────────
            obs_pts = detect_obstacles(pc)

            # ── APF → PWM ─────────────────────────────────────────────────
            Fx, Fz = apf_force(serit_pts, obs_pts)
            left_pwm, right_pwm = force_to_pwm(Fx, Fz)

            # ── Arduino ───────────────────────────────────────────────────
            if ser:
                send_cmd(ser, left_pwm, right_pwm)

            # ── FPS ───────────────────────────────────────────────────────
            t_now    = time.time()
            fps_avg  = 0.9 * fps_avg + 0.1 * (1.0 / max(t_now - t_prev, 1e-9))
            t_prev   = t_now

            # ── Görsel ────────────────────────────────────────────────────
            if not args.no_gui or args.stream:
                vis = draw_debug(frame, boxes, len(serit_pts), len(obs_pts),
                                 Fx, Fz, left_pwm, right_pwm)
                cv2.putText(vis, f"FPS:{fps_avg:4.1f}",
                            (vis.shape[1] - 110, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                if args.stream:
                    _update_stream(vis)
                if not args.no_gui:
                    cv2.imshow("APF Serit+Engel", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            if args.no_gui:
                nearest_s = min((b[5] for b in boxes if b[5] > 0), default=-1.0)
                if len(obs_pts) > 0:
                    nearest_o = float(
                        np.min(np.sqrt(obs_pts[:, 0] ** 2 + obs_pts[:, 2] ** 2))
                    )
                else:
                    nearest_o = -1.0
                print(
                    f"\rFPS:{fps_avg:4.1f}  L:{left_pwm:+4d} R:{right_pwm:+4d}"
                    f"  Serit:{len(boxes)}({nearest_s:.2f}m)"
                    f"  Engel:{len(obs_pts)}({nearest_o:.2f}m)",
                    end="", flush=True,
                )

    finally:
        print("\n[INFO] Durduruluyor...")
        if ser:
            send_cmd(ser, 0, 0)
            time.sleep(0.1)
            ser.close()
        running[0] = False
        zed.close()
        cv2.destroyAllWindows()
        print("[INFO] Kapatıldı.")


if __name__ == "__main__":
    main()
