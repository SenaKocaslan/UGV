#!/usr/bin/env python3
"""
VISUAL PROCESSOR - UGVC-10 Şartnamesine Göre
Görev 1: Beyaz şerit takibi (asfalt zemin, beyaz çizgi)
Görev 2: Beyaz daire kaçınma (60cm pothole simülasyonu)
Kamera: ZED Mini (sol göz RGB + 32FC1 derinlik haritası)

DEĞİŞİKLİKLER (ZED Mini):
  - process_frame(frame, depth_frame=None) → depth_frame eklendi
  - _detect_trap → engel merkezinde gerçek derinlik okunuyor
  - obstacles tuple: (x_px, side_norm, z_metre)  ← z artık gerçek
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Hafıza sistemi
# ---------------------------------------------------------------------------
class LaneMemory:
    def __init__(self):
        self.left_fit      = None
        self.right_fit     = None
        self.detected      = False
        self.missed_frames = 0
        self.use_polyfit   = False
        self.last_error    = 0.0
        self.last_width    = 200.0  # piksel cinsinden tahmini şerit genişliği


# ---------------------------------------------------------------------------
# Ana sınıf
# ---------------------------------------------------------------------------
class VisualProcessor:
    def __init__(self):

        # --- GÖRÜNTÜ BOYUTU ---
        self.img_width  = 640
        self.img_height = 480

        # --- ŞERİT AYARLARI ---
        self.roi_top_ratio         = 0.40
        self.lane_min_pixels       = 40
        self.sliding_margin        = 70
        self.polyfit_margin        = 55
        self.min_pixels_per_window = 35

        # Beyaz şerit — HLS
        self.lane_lower = np.array([0,   200,   0]) # hue(renk) değeri beyaza ayarlanmamış  yüksek parlaklığa sahip herhangi bir bölge beyaz olarak kabul ediliyor.
        self.lane_upper = np.array([180, 255, 255])

        # --- TUZAK (BEYAZ DAİRE/ELİPS) AYARLARI ---
        self.trap_min_area   = 300
        self.trap_max_area   = 200000
        self.min_circularity = 0.35
        self.max_aspect      = 4.0
        self.trap_roi_top    = 0

        # --- ZED MİNİ DERİNLİK AYARLARI ---
        # ZED Mini'nin geçerli ölçüm aralığı: 0.1m – 15m
        # Bu aralık dışındaki değerler NaN/Inf olabilir → filtrele
        self.depth_min_valid = 0.10   # metre — çok yakın → geçersiz
        self.depth_max_valid = 10.0   # metre — çok uzak → ilgisiz
        # Engel merkezinde tek piksel yerine küçük alan ortalaması al
        self.depth_sample_radius = 5  # piksel

        # --- HAFIZA ---
        self.memory = LaneMemory()

        # --- PERSPEKTİF ---
        self.M    = None
        self.Minv = None

        print("[VisualProcessor UGVC-10 + ZED Mini] Hazir -- {}x{}".format(
            self.img_width, self.img_height))

    # -----------------------------------------------------------------------
    # ANA GİRİŞ NOKTASI
    # -----------------------------------------------------------------------
    def process_frame(self, frame, depth_frame=None):
        """
        Parametreler:
          frame       : BGR görüntü (sol göz)
          depth_frame : float32 derinlik haritası, metre cinsinden (32FC1)
                        None ise eski geometrik yöntem kullanılır (fallback)

        Dönüş:
          obstacles : list of (x_px, side_norm, z_metre)
              z_metre: ZED'den okunan gerçek mesafe (veya fallback: 0.5)
          lane_info : dict
              detected, left_fit, right_fit, center_error, debug_frame
        """
        frame = cv2.resize(frame, (self.img_width, self.img_height))

        # Derinlik haritasını da aynı boyuta getir
        if depth_frame is not None:
            depth_frame = cv2.resize(
                depth_frame,
                (self.img_width, self.img_height),
                interpolation=cv2.INTER_NEAREST   # derinlikte interpolasyon istenmiyor
            )
        cv2.imshow("perpektif değişmeden önce", frame)
        self._init_perspective()  #perspektif değiştirilir
       
        # 1. TUZAK TESPİTİ
        trap_detected, trap_x, trap_y, trap_cnt = self._detect_trap(frame) 

        obstacles = []
        if trap_detected:
            side_norm = (trap_x - self.img_width / 2.0) / (self.img_width / 2.0)  # aracın sağa mı sola mı kaydığını -1.0(sol) - +1.0(sağ) arasındaki değerler ile ifade eder.

            # ── ZED Mini'den gerçek mesafe ──────────────────────────────────
            z_real = self._read_depth(depth_frame, trap_x, trap_y)
            # ────────────────────────────────────────────────────────────────

            obstacles.append((float(trap_x), float(side_norm), float(z_real)))   

        # 2. ŞERİT TESPİTİ
        left_fit, right_fit = self._detect_lanes(frame)

        # 3. SONUÇ
        lane_info = self._build_lane_info(
            frame, left_fit, right_fit, obstacles, trap_cnt)

        return obstacles, lane_info

    # -----------------------------------------------------------------------
    # ZED MİNİ DERİNLİK OKUMA — ROBUST YÖNTEM  ? e hani zedn derinlik okuduğu bir yer mi var? epth_frame değişkeni nerede tanımlandı?
    # -----------------------------------------------------------------------
    def _read_depth(self, depth_frame, cx, cy):
        """
        Engel merkezinin etrafındaki küçük alanda medyan derinliği döndürür.
        Geçersiz (NaN / Inf / aralık dışı) değerler filtrelenir.
        depth_frame None ise 0.5 m fallback döner.
        """
        if depth_frame is None:
            return 0.5   # ZED bağlı değil → eski sabit değer

        r = self.depth_sample_radius
        h, w = depth_frame.shape[:2]

        # Sınır kontrolü
        x1 = max(cx - r, 0)
        x2 = min(cx + r, w - 1)
        y1 = max(cy - r, 0)
        y2 = min(cy + r, h - 1)

        patch = depth_frame[y1:y2, x1:x2].flatten()

        # Geçerli değerleri filtrele
        valid = patch[
            np.isfinite(patch) &
            (patch >= self.depth_min_valid) &
            (patch <= self.depth_max_valid)
        ]

        if len(valid) == 0:
            return 0.5   # geçerli piksel bulunamadı → fallback

        return float(np.median(valid))

    # -----------------------------------------------------------------------
    # TUZAK TESPİTİ — cy de döndürülüyor (derinlik için gerekli)
    def _detect_trap(self, frame):
        """
        YCrCb renk uzayı ile beyaz daire tuzak tespiti.
        Dönüş: (detected: bool, trap_x: int, trap_y: int, contour)
        """
        # --- 1. YCrCb maskesi ---
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)

        mask_Y  = cv2.inRange(Y,  180, 255)   # yüksek parlaklık → beyaz
        mask_Cr = cv2.inRange(Cr, 120, 140)   # kırmızı kroması nötr
        mask_Cb = cv2.inRange(Cb, 120, 140)   # mavi kroması nötr
        mask = cv2.bitwise_and(mask_Y, cv2.bitwise_and(mask_Cr, mask_Cb))

        # --- 2. Morfolojik temizleme (gürültü + delik kapatma) ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        cv2.imshow("ycrcb renk uzayı", mask)

        # --- 3. Kontur tabanlı daire/elips tespiti ---
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt  = None
        best_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.trap_min_area or area > self.trap_max_area:
                continue

            # Yuvarlıklık filtresi: 4πA / P²
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity < self.min_circularity:
                continue

            # En-boy oranı filtresi
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > self.max_aspect:
                continue

            if area > best_area:
                best_area = area
                best_cnt  = cnt

        if best_cnt is not None:
            M_cnt = cv2.moments(best_cnt)
            if M_cnt["m00"] != 0:
                cx = int(M_cnt["m10"] / M_cnt["m00"])
                cy = int(M_cnt["m01"] / M_cnt["m00"])
            else:
                x, y, w, h = cv2.boundingRect(best_cnt)
                cx, cy = x + w // 2, y + h // 2

            return True, cx, cy, best_cnt

        return False, self.img_width // 2, self.img_height // 2, None
       
    # -----------------------------------------------------------------------
    # ŞERİT TESPİTİ  (değişmedi)
    # -----------------------------------------------------------------------
    def _detect_lanes(self, frame):
        mem = self.memory

        hls  = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS) #resmi rgb den hls ye çevirdi
        mask = cv2.inRange(hls, self.lane_lower, self.lane_upper) #eşik değer oluşturuluyor

        roi_top  = int(self.img_height * self.roi_top_ratio)
        roi_mask = np.zeros_like(mask)  
        roi_mask[roi_top:, :] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

        binary_warped = cv2.warpPerspective(
            mask, self.M, (self.img_width, self.img_height))
        

        #debug pencereleri


        cv2.imshow("hls maskesi", mask)
        cv2.imshow("Kuş bakışı görüntü", binary_warped)
        cv2.waitKey(1)




        if mem.use_polyfit and mem.left_fit is not None:
            left_fit, right_fit = self._polyfit_search(binary_warped)
        else:
            left_fit, right_fit = self._sliding_window_search(binary_warped)

        if left_fit is not None and right_fit is not None:
            mem.detected      = True
            mem.missed_frames = 0
            mem.use_polyfit   = True
            if mem.left_fit is not None:
                mem.left_fit  = 0.5 * mem.left_fit  + 0.5 * left_fit
                mem.right_fit = 0.5 * mem.right_fit + 0.5 * right_fit
            else:
                mem.left_fit  = left_fit
                mem.right_fit = right_fit
        else:
            mem.missed_frames += 1
            if mem.missed_frames > 10:
                mem.detected    = False
                mem.use_polyfit = False
                mem.left_fit    = None
                mem.right_fit   = None

        return mem.left_fit, mem.right_fit

    # -----------------------------------------------------------------------
    # SLİDİNG WINDOW  (değişmedi)
    # -----------------------------------------------------------------------
    def _sliding_window_search(self, binary_warped):
        height    = binary_warped.shape[0]
        histogram = np.sum(binary_warped[height // 2:, :], axis=0)

        if np.max(histogram) < self.lane_min_pixels:
            return None, None

        midpoint    = len(histogram) // 2
        leftx_base  = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        if abs(rightx_base - leftx_base) < 60:
            return None, None

        nwindows      = 10
        window_height = height // nwindows
        nonzero       = binary_warped.nonzero()
        nonzeroy      = np.array(nonzero[0])
        nonzerox      = np.array(nonzero[1])
        leftx_current  = leftx_base
        rightx_current = rightx_base
        left_lane_inds  = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low  = height - (window + 1) * window_height
            win_y_high = height - window       * window_height
            wll = leftx_current  - self.sliding_margin
            wlh = leftx_current  + self.sliding_margin
            wrl = rightx_current - self.sliding_margin
            wrh = rightx_current + self.sliding_margin

            good_left  = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= wll) & (nonzerox < wlh)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= wrl) & (nonzerox < wrh)).nonzero()[0]

            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

            if len(good_left)  > self.min_pixels_per_window:
                leftx_current  = int(np.mean(nonzerox[good_left]))
            if len(good_right) > self.min_pixels_per_window:
                rightx_current = int(np.mean(nonzerox[good_right]))

        left_lane_inds  = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        if len(left_lane_inds) < 60 or len(right_lane_inds) < 60:
            return None, None

        left_fit  = np.polyfit(nonzeroy[left_lane_inds],
                               nonzerox[left_lane_inds],  2)
        right_fit = np.polyfit(nonzeroy[right_lane_inds],
                               nonzerox[right_lane_inds], 2)
        return left_fit, right_fit

    # -----------------------------------------------------------------------
    # POLYFIT SEARCH  (değişmedi)
    # -----------------------------------------------------------------------
    def _polyfit_search(self, binary_warped):
        mem      = self.memory
        nonzero  = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        lf = mem.left_fit
        rf = mem.right_fit
        m  = self.polyfit_margin

        left_lane_inds = (
            (nonzerox > (lf[0]*nonzeroy**2 + lf[1]*nonzeroy + lf[2] - m)) &
            (nonzerox < (lf[0]*nonzeroy**2 + lf[1]*nonzeroy + lf[2] + m)))
        right_lane_inds = (
            (nonzerox > (rf[0]*nonzeroy**2 + rf[1]*nonzeroy + rf[2] - m)) &
            (nonzerox < (rf[0]*nonzeroy**2 + rf[1]*nonzeroy + rf[2] + m)))

        leftx  = nonzerox[left_lane_inds]
        lefty  = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if len(leftx) < 60 or len(rightx) < 60:
            return None, None

        return np.polyfit(lefty, leftx, 2), np.polyfit(righty, rightx, 2)

    # -----------------------------------------------------------------------
    # LANE INFO DICT + DEBUG GÖRSEL
    # -----------------------------------------------------------------------
    def _build_lane_info(self, frame, left_fit, right_fit,
                         obstacles, trap_cnt):
        debug_frame = frame.copy()
        mem = self.memory

        if trap_cnt is not None:
            cv2.drawContours(debug_frame, [trap_cnt], -1, (0, 0, 255), 3)
            try:
                if len(trap_cnt) >= 5:
                    ellipse = cv2.fitEllipse(trap_cnt)
                    cv2.ellipse(debug_frame, ellipse, (0, 180, 255), 2)
            except Exception:
                pass
            M_cnt = cv2.moments(trap_cnt)
            if M_cnt["m00"] != 0:
                cx = int(M_cnt["m10"] / M_cnt["m00"])
                cy = int(M_cnt["m01"] / M_cnt["m00"])
                cv2.circle(debug_frame, (cx, cy), 8, (0, 0, 255), -1)

                # ── Mesafeyi de debug ekranda göster ─────────────────────
                if obstacles:
                    z_m = obstacles[0][2]
                    label = f"TUZAK  {z_m:.2f}m"
                else:
                    label = "TUZAK"
                cv2.putText(debug_frame, label,
                            (cx - 30, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if left_fit is None or right_fit is None:
            self._draw_status(debug_frame, "SERIT KAYIP", (0, 0, 255))
            return {
                'detected'    : False,
                'left_fit'    : None,
                'right_fit'   : None,
                'center_error': mem.last_error,
                'debug_frame' : debug_frame,
            }

        ploty      = np.linspace(0, self.img_height - 1, self.img_height)
        left_fitx  = left_fit[0]  * ploty**2 + left_fit[1]  * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        lane_center  = (left_fitx[-1] + right_fitx[-1]) / 2.0
        car_center   = self.img_width / 2.0
        center_error = lane_center - car_center

        lane_width = float(right_fitx[-1] - left_fitx[-1])
        mem.last_width  = max(lane_width, 50.0)
        mem.last_error  = center_error

        debug_frame = self._draw_lane(debug_frame, left_fitx, right_fitx, ploty)

        pts_left = np.array(
            [np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
        cv2.polylines(debug_frame, pts_left, False, (0, 255, 255), 2)

        pts_right = np.array(
            [np.transpose(np.vstack([right_fitx, ploty]))], dtype=np.int32)
        cv2.polylines(debug_frame, pts_right, False, (255, 0, 0), 2)

        cy_line = self.img_height - 30
        cv2.line(debug_frame,
                 (int(lane_center), cy_line - 12),
                 (int(lane_center), cy_line + 12),
                 (0, 255, 255), 3)
        cv2.line(debug_frame,
                 (int(car_center), cy_line - 12),
                 (int(car_center), cy_line + 12),
                 (255, 100, 0), 3)

        cv2.putText(debug_frame,
                    "Hata: {:.1f}px".format(center_error),
                    (8, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if obstacles:
            self._draw_status(debug_frame, "TUZAK TESPIT!", (0, 80, 255))
        else:
            curvature = abs(left_fit[0]) + abs(right_fit[0])
            if curvature > 0.0003:
                mode = "VIRAJ-SERT"
            elif curvature > 0.0001:
                mode = "VIRAJ"
            else:
                mode = "DUZLU"
            self._draw_status(debug_frame,
                              "Serit OK | " + mode, (0, 220, 0))

        return {
            'detected'    : True,
            'left_fit'    : left_fit,
            'right_fit'   : right_fit,
            'center_error': center_error,
            'debug_frame' : debug_frame,
        }

    # -----------------------------------------------------------------------
    def _init_perspective(self):
        if self.M is not None:
            return
        w, h = self.img_width, self.img_height
        src = np.float32([
            [w * 0.10, h], [w * 0.10, 0],
            [w * 0.90, 0], [w * 0.90, h],
        ])
        dst = np.float32([
            [0, h], [0, 0],
            [w, 0], [w, h],
        ])
        self.M    = cv2.getPerspectiveTransform(src, dst)  #ileri dönüşüm yamuk görütntüden -> düzeltilmiş görüntü
        self.Minv = cv2.getPerspectiveTransform(dst, src)  #geri dönüşüm

    def _draw_lane(self, frame, left_fitx, right_fitx, ploty):
        lane_mask = np.zeros(
            (self.img_height, self.img_width, 3), dtype=np.uint8)
        pts_left  = np.array(
            [np.transpose(np.vstack([left_fitx,  ploty]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(lane_mask, np.int32([pts]), (0, 180, 0))
        lane_unwarp = cv2.warpPerspective(
            lane_mask, self.Minv, (self.img_width, self.img_height))
        return cv2.addWeighted(frame, 1, lane_unwarp, 0.25, 0)

    def _draw_status(self, frame, text, color):
        cv2.putText(frame, text, (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4)
        cv2.putText(frame, text, (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)