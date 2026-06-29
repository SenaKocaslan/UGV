#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YARIŞ SİSTEMİ v10.3 — KÜMÜLATIF OYLAMA
========================================
v10.2 → v10.3 değişiklikleri:

  [DÜZELTME #1] Kümülatif oylama sistemi — tanıma hızı ~2-3x arttı
    NEDEN: v10.2'de her başarılı onaydan sonra emb_pencere.temizle()
           çağrılıyordu. Bu yüzden onay sayacı dolmadan araya bir
           başarısız frame girince sistem sıfırdan başlıyordu.
           8m'de bu döngü 7 saniye sürebiliyordu.
    ÇÖZÜM: Kümülatif oylama:
           - Aynı isim gelince onay_sayac += 1 (temizleme yok)
           - Farklı isim gelince aday değişir, sayac = 1
           - Başarısız tanımada sayac azalır ama sıfırlanmaz
           - emb_pencere.temizle() kaldırıldı (onay döngüsünden)

  [DÜZELTME #2] _aday_isim durum değişkeni eklendi
    Kümülatif oylamada hangi adayın sayıldığını takip eder.

=================================================
v10.0 + v10.1 → v10.2 birleştirme değişiklikleri (korundu):
=================================================
v10.0 + v10.1 → v10.2 birleştirme değişiklikleri:

  [v10.0'DAN ALINAN]
    ✓ Çift kamera profili: --kamera-profil brio/c925
    ✓ Baskılı yüz (manken) augmentasyonu (+5 baskı simülasyonu, toplam ~15 emb)
    ✓ --manken-modu: hareketsiz hedef için optimize parametreler
    ✓ arcface_hard_reject: 0.14  (baskılı yüz için düşürüldü)
    ✓ arcface_esik_ultra: 0.20   (baskılı yüz + uzak mesafe)
    ✓ ROI odak bölgesi (yan manken koruması) — roi_aktif, roi_margin_x/y
    ✓ yarisma_esikleri: hard_reject=0.12, uzak=0.34, ultra=0.16

  [v10.1'DEN ALINAN]
    ✓ PartialLock — tarama koruması (kadrajdan çıkan yüzü 3s hafızada tut)
    ✓ TaramaYonBellegi — dominant yön takibi ile PartialLock doğrulama
    ✓ Multi-person güvenlik katmanı — 1/2/3+ yüz için dinamik fark eşiği
    ✓ Jetson Orin Nano desteği — DLA kaldırıldı, TRT workspace=2GB
    ✓ YOLO adaptive polling — duruma göre YOLO çağrı sıklığı
    ✓ ThreadPoolExecutor tabanlı AsenkronEmbedding (bellek sızıntısı yok)
    ✓ KayanEmbeddingPenceresi.ekle_liste() — PartialLock entegrasyonu
    ✓ px<15 → min_onay=1 (çok küçük yüzler için)
    ✓ _beklened_px typo düzeltmesi (zaten v10.0'da da düzeltilmişti)

  [v10.2 YENİ]
    ✓ Lazer ateşleme gecikmesi: DOĞRULANDI sonrası 15 frame beklenir
      (yanlış kilitleme varsa reverif yakalamak için süre tanınır)
    ✓ PartialLock expire süresini manken modunda 150 frame'e çıkar
      (hareketsiz hedef → aynı konumdan geri dönme olasılığı yüksek)
    ✓ Manken modunda multi-person fark eşiği +0.02 artırılır
      (manken modunda yakın mankenler için ekstra güvenlik)
    ✓ _yuz_bul_hizli: ROI aktifse önce ortadan tara (v10.0),
      ardından görünen yüz sayısını döndür (v10.1 multi-person için)
"""

import cv2
import numpy as np
import os
import pickle
import time
import threading
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Deque
import argparse

# ── YOLO ──────────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics yok! pip install ultralytics")
    exit(1)

# ── ArcFace ───────────────────────────────────────────────────────────────────
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_MEVCUT = True
except ImportError:
    print("❌ insightface yok! pip install insightface onnxruntime-gpu")
    exit(1)

# ── ONNX (Super-Resolution) ───────────────────────────────────────────────────
try:
    import onnxruntime as ort
    ONNX_MEVCUT = True
except ImportError:
    ONNX_MEVCUT = False
    print("⚠️  onnxruntime yok — SR Lanczos moduna düşecek")

# ── Arduino ───────────────────────────────────────────────────────────────────
try:
    import serial
    SERIAL_MEVCUT = True
except ImportError:
    SERIAL_MEVCUT = False
    print("⚠️  pyserial yok. Arduino devre dışı.")


# ─────────────────────────────────────────────────────────────────────────────
# KAMERA PROFİLLERİ  [v10.0'dan]
# ─────────────────────────────────────────────────────────────────────────────
KAMERA_PROFILLERI = {
    "brio": {
        "aciklama":    "Logitech Brio 4K (1920x1080 @60fps)",
        "genislik":    1920,
        "yukseklik":   1080,
        "fps":         60,
        "mjpeg":       True,
        "imgsz_yakin": 640,
        "imgsz_orta":  960,
        "imgsz_uzak":  1280,
        "imgsz_ultra": 1920,
        "focus":       10,
        "autofocus":   0,
    },
    "c925": {
        "aciklama":    "Logitech C925 (1280x720 @30fps)",
        "genislik":    1280,
        "yukseklik":   720,
        "fps":         30,
        "mjpeg":       True,
        "imgsz_yakin": 640,
        "imgsz_orta":  960,
        "imgsz_uzak":  1280,
        "imgsz_ultra": 1536,
        "focus":       10,
        "autofocus":   0,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# PARTIAL LOCK — Kısmi Tanıma Hafızası  [v10.1'den]
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PartialLockKayit:
    """
    Yüz kadrajdan çıkmadan önce 'umut verici' skor aldıysa saklanır.
    Araç tarama yaparken aynı bölgeden tekrar yüz gelince buradan devam edilir.
    """
    embeddingler:     list   # Toplanan embedding listesi
    en_iyi_skor:      float  # Şimdiye kadarki en yüksek benzerlik skoru
    aday_isim:        str    # En yüksek skoru alan isim
    son_konum_x:      int    # Kadrajdan çıktığındaki merkez X
    son_konum_y:      int    # Kadrajdan çıktığındaki merkez Y
    cikis_yonu:       str    # "SOL" / "SAG" / "YUKARI" / "BELIRSIZ"
    olusturma_frame:  int
    gecerlilik_frame: int    # Bu frame'e kadar geçerli (expire)
    px_genislik:      int


# ─────────────────────────────────────────────────────────────────────────────
# TARAMA YÖN BELLEĞİ  [v10.1'den]
# ─────────────────────────────────────────────────────────────────────────────
class TaramaYonBellegi:
    """Son N servo komutunun yönünü saklar."""
    def __init__(self, pencere: int = 15):
        self._gecmis: Deque[str] = deque(maxlen=pencere)

    def komut_ekle(self, yon: str):
        self._gecmis.append(yon)

    def dominant_yon(self) -> str:
        if not self._gecmis:
            return "BELIRSIZ"
        sol = self._gecmis.count("SOL")
        sag = self._gecmis.count("SAG")
        if sol == 0 and sag == 0:
            return "MERKEZ"
        return "SOL" if sol > sag else "SAG"

    def temizle(self):
        self._gecmis.clear()


# ─────────────────────────────────────────────────────────────────────────────
# YAPILANDIRMA  [v10.0 + v10.1 birleşimi]
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class YarisConfig:
    # ── Kamera ────────────────────────────────────────────────────────────
    kamera_indeks:    int  = 2
    kamera_genislik:  int  = 1280
    kamera_yukseklik: int  = 720
    kamera_fps:       int  = 30
    mjpeg_aktif:      bool = True
    kamera_profil:    str  = "c925"   # [v10.0] "brio" veya "c925"

    # ── YOLO ──────────────────────────────────────────────────────────────
    yolo_model:        str   = "yolov8s-face.pt"
    yolo_conf_arama:   float = 0.06
    yolo_conf_tespit:  float = 0.10
    imgsz_yakin:       int   = 640
    imgsz_orta:        int   = 960
    imgsz_uzak:        int   = 1280
    imgsz_ultra:       int   = 1536
    min_yuz_piksel:    int   = 8

    # ── YOLO Adaptive Polling  [v10.1] ────────────────────────────────────
    yolo_arama_interval:      int = 1
    yolo_kilitli_interval:    int = 2
    yolo_dogrulandi_interval: int = 3

    # ── Multi-scale ───────────────────────────────────────────────────────
    multiscale_aktif:  bool  = True
    multiscale_overlap: float = 0.3

    # ── ROI (yan manken koruması)  [v10.0] ────────────────────────────────
    roi_aktif:     bool  = True
    roi_margin_x:  float = 0.20   # sol/sağdan %20 kenar kes
    roi_margin_y:  float = 0.10   # üst/alttan %10 kenar kes

    # ── ByteTrack ─────────────────────────────────────────────────────────
    bytetrack_conf:           float = 0.08
    bytetrack_re_assign_esik: int   = 300

    # ── ArcFace eşikleri  [v10.0 baskılı yüz değerleri] ──────────────────
    arcface_hard_reject:      float = 0.14   # v10.0: 0.22→0.14 baskılı yüz
    arcface_esik_yakin:       float = 0.54
    arcface_esik_orta:        float = 0.50
    arcface_esik_uzak:        float = 0.44
    arcface_esik_ultra:       float = 0.20   # v10.0: 0.26→0.20 baskılı yüz
    arcface_esik_varsayilan:  float = 0.40
    arcface_reverif_esik:     float = 0.50
    uzak_esik_indirimi:       float = 0.04

    # ── Doğrulama ─────────────────────────────────────────────────────────
    kilitlenme_min_onay:  int = 4
    dogrulama_deneme_max: int = 4

    # ── Multi-person güvenlik  [v10.1] ────────────────────────────────────
    multiperson_fark_1yuz:  float = 0.08
    multiperson_fark_2yuz:  float = 0.10
    multiperson_fark_3plus: float = 0.14   # 5-6 kişilik ortam için sıkı

    # ── PartialLock  [v10.1] ──────────────────────────────────────────────
    partial_lock_aktif:      bool  = True
    partial_lock_skor_esigi: float = 0.65  # eşiğin %65'ini geçen skor → sakla
    partial_lock_max_frame:  int   = 90    # 3 saniye @30fps
    partial_lock_bolge_esik: int   = 200   # aynı bölge = merkez X farkı <200px
    partial_lock_min_emb:    int   = 1

    # ── Yüz kalite filtresi ───────────────────────────────────────────────
    kalite_aktif:      bool  = True
    kalite_blur_esik:  float = 22.0
    kalite_yaw_esik:   float = 40.0
    kalite_pitch_esik: float = 35.0

    # ── Super-Resolution ──────────────────────────────────────────────────
    sr_aktif:       bool = True
    sr_esik_piksel: int  = 200
    sr_model_yolu:  str  = "realesrgan_x4.onnx"
    sr_scale:       int  = 4

    # ── Kayan embedding penceresi ─────────────────────────────────────────
    emb_pencere_max: int = 20
    emb_en_iyi_n:   int = 10
    emb_al_periyot: int = 2

    # ── Kayıp eşikleri ────────────────────────────────────────────────────
    kayip_esik_kilit:      int = 45
    kayip_esik_dogrulandi: int = 120

    # ── Periyodik re-verifikasyon ─────────────────────────────────────────
    reverifikasyon_periyot:           int = 50
    reverifikasyon_basarisizlik_esik: int = 3
    reverifikasyon_timeout_frame:     int = 120

    # ── Servo ─────────────────────────────────────────────────────────────
    servo_merkez_band: float = 0.07
    servo_k_yakin:     int   = 28
    servo_k_uzak:      int   = 38

    # ── Lazer ateşleme gecikmesi ──────────────────────────────────────────
    lazer_gecikme_frame: int = 15

    dijital_zoom: float = 1.0

    # ── CLAHE ─────────────────────────────────────────────────────────────
    clahe_clip: float = 6.0
    clahe_tile: tuple = field(default_factory=lambda: (4, 4))

    # ── FPS Optimizasyon  [v10.4] ─────────────────────────────────────────
    # _on_isle (filter2D+CLAHE) her kaç frame'de bir çalışsın?
    # 1 = her frame (eski davranış), 2 = ~%50 CPU kazancı, 3 = ~%65 kazanç
    on_isle_interval:    int = 2
    # ARAMA modunda multiscale YOLO her kaç frame'de bir çalışsın?
    # 1 = her frame (eski), 2 önerilir — yüz yokken gereksiz tekrar engellenir
    arama_yolo_interval: int = 2
    # ByteTrack KİLİTLİ/DOĞRULANMIŞ modda kullanılacak imgsz
    # Yüz zaten takipte, düşük çözünürlük takibi bozmaz ama hızı artırır
    bytetrack_imgsz_kilitli: int = 320

    # ── Jetson Orin Nano ──────────────────────────────────────────────────
    tensorrt_aktif:   bool = False
    jetson_orin:      bool = False
    trt_workspace_gb: int  = 2

    # ── Modlar ────────────────────────────────────────────────────────────
    yarisma_modu: bool = False
    manken_modu:  bool = False   # [v10.0] Hareketsiz hedef (baskılı yüz) optimizasyonu

    # ── Debug ─────────────────────────────────────────────────────────────
    debug_mesafe:  bool = False
    debug_skor:    bool = False
    debug_partial: bool = False

    # ─────────────────────────────────────────────────────────────────────
    def kamera_profil_uygula(self):
        """Seçilen kamera profilini config'e uygular."""
        profil = self.kamera_profil.lower()
        if profil not in KAMERA_PROFILLERI:
            print(f"⚠️  Bilinmeyen kamera profili '{profil}', c925 kullanılıyor")
            profil = "c925"
            self.kamera_profil = "c925"
        p = KAMERA_PROFILLERI[profil]
        self.kamera_genislik  = p["genislik"]
        self.kamera_yukseklik = p["yukseklik"]
        self.kamera_fps       = p["fps"]
        self.mjpeg_aktif      = p["mjpeg"]
        self.imgsz_yakin      = p["imgsz_yakin"]
        self.imgsz_orta       = p["imgsz_orta"]
        self.imgsz_uzak       = p["imgsz_uzak"]
        self.imgsz_ultra      = p["imgsz_ultra"]
        print(f"📷 Kamera profili: {p['aciklama']}")

    def yarisma_esikleri_uygula(self):
        """Yarışma modu: eşikleri düşür."""
        self.arcface_hard_reject    = 0.12
        self.arcface_esik_uzak      = 0.34
        self.arcface_esik_ultra     = 0.16
        self.multiperson_fark_3plus = 0.16
        print("⚡ Yarışma modu: eşikler düşürüldü "
              "(hard_reject=0.12, uzak=0.34, ultra=0.16, multi_fark_3+=0.16)")


# ─────────────────────────────────────────────────────────────────────────────
# JETSON ORIN NANO YARDIMCI  [v10.1'den]
# ─────────────────────────────────────────────────────────────────────────────
def jetson_guc_modu_kontrol():
    try:
        r = subprocess.run(["nvpmodel", "-q"],
                           capture_output=True, text=True, timeout=3)
        cikti = r.stdout + r.stderr
        if "MAXN" in cikti or "MODE_ID: 0" in cikti:
            print("✓ Jetson güç modu: MAXN (maksimum performans)")
        else:
            print("⚠️  Jetson Orin Nano güç modu MAXN değil!")
            print("   sudo nvpmodel -m 0 && sudo jetson_clocks")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def jetson_orin_nano_optimizasyonu():
    """Orin Nano'ya özgü ortam değişkenleri. DLA yok, CUDA+TRT kullan."""
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    os.environ.setdefault("TRT_BUILDER_CACHE_ENABLE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")


# ─────────────────────────────────────────────────────────────────────────────
# SUPER-RESOLUTION  [v10.1 tabanlı, TensorRT provider eklendi]
# ─────────────────────────────────────────────────────────────────────────────
class SuperResolution:
    def __init__(self, cfg: YarisConfig):
        self.cfg = cfg
        self.session = None
        self.mod = "LANCZOS"

        if not cfg.sr_aktif:
            print("  SR: devre dışı")
            return

        if ONNX_MEVCUT and os.path.exists(cfg.sr_model_yolu):
            try:
                avail = ort.get_available_providers()
                if "TensorrtExecutionProvider" in avail:
                    providers = ["TensorrtExecutionProvider",
                                 "CUDAExecutionProvider",
                                 "CPUExecutionProvider"]
                elif "CUDAExecutionProvider" in avail:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers = ["CPUExecutionProvider"]
                self.session = ort.InferenceSession(
                    cfg.sr_model_yolu, providers=providers)
                self.mod = "ONNX_SR"
                print(f"✓ Super-Resolution: ONNX ({cfg.sr_model_yolu})")
            except Exception as e:
                print(f"⚠️  SR yüklenemedi ({e}) → Lanczos")
        else:
            if cfg.sr_aktif:
                print(f"  SR: model bulunamadı ({cfg.sr_model_yolu}) → Lanczos")

    def buyut(self, gorsel_bgr: np.ndarray) -> np.ndarray:
        if gorsel_bgr is None or gorsel_bgr.size == 0:
            return gorsel_bgr
        h, w = gorsel_bgr.shape[:2]
        hedef_w, hedef_h = w * self.cfg.sr_scale, h * self.cfg.sr_scale
        if self.session is not None:
            try:
                return self._onnx_sr(gorsel_bgr, hedef_w, hedef_h)
            except Exception:
                pass
        return cv2.resize(gorsel_bgr, (hedef_w, hedef_h),
                          interpolation=cv2.INTER_LANCZOS4)

    def _onnx_sr(self, gorsel_bgr, hedef_w, hedef_h):
        rgb = (cv2.cvtColor(gorsel_bgr, cv2.COLOR_BGR2RGB)
               .astype(np.float32) / 255.0)
        inp = np.transpose(rgb, (2, 0, 1))[np.newaxis]
        cikti = self.session.run(
            None, {self.session.get_inputs()[0].name: inp})[0]
        sonuc = np.clip(
            cikti[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        sonuc = cv2.cvtColor(sonuc, cv2.COLOR_RGB2BGR)
        if sonuc.shape[1] != hedef_w or sonuc.shape[0] != hedef_h:
            sonuc = cv2.resize(sonuc, (hedef_w, hedef_h))
        return sonuc

    def gerekli_mi(self, piksel_genislik: int) -> bool:
        return self.cfg.sr_aktif and piksel_genislik < self.cfg.sr_esik_piksel


# ─────────────────────────────────────────────────────────────────────────────
# YÜZ KALİTE FİLTRESİ  [v10.1 tabanlı, v10.2: küçük px daha esnek]
# ─────────────────────────────────────────────────────────────────────────────
class YuzKaliteFiltrer:
    def __init__(self, cfg: YarisConfig):
        self.cfg = cfg

    def degerlendir(self, yuz_bgr: np.ndarray,
                    insightface_yuzu=None,
                    yuz_px_genislik: int = None) -> Tuple[bool, float, str]:
        if not self.cfg.kalite_aktif:
            return True, 100.0, "OK"
        if yuz_bgr is None or yuz_bgr.size == 0:
            return False, 0.0, "BOŞ"
        h, w = yuz_bgr.shape[:2]
        if w < 30 or h < 30:
            return False, 0.0, f"KÜÇÜK({w}x{h})"

        px_ref = yuz_px_genislik if yuz_px_genislik is not None else w

        # 8m'de px≈12-18: çok gevşek blur eşiği
        if px_ref < 20:
            efektif = self.cfg.kalite_blur_esik * 0.06
        elif px_ref < 40:
            efektif = self.cfg.kalite_blur_esik * 0.12
        elif px_ref < 60:
            efektif = self.cfg.kalite_blur_esik * 0.22
        elif px_ref < 100:
            efektif = self.cfg.kalite_blur_esik * 0.42
        else:
            efektif = self.cfg.kalite_blur_esik

        gri = cv2.cvtColor(yuz_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gri, cv2.CV_64F).var()
        if blur < efektif:
            return False, blur / (efektif + 1e-8) * 50, "BULANIK"

        if insightface_yuzu is not None:
            try:
                pose = insightface_yuzu.pose
                if pose is not None:
                    yaw   = abs(float(pose[0]))
                    pitch = abs(float(pose[1]))
                    # Küçük yüzlerde açı toleransını artır
                    yaw_esik   = (self.cfg.kalite_yaw_esik * 1.5
                                  if px_ref < 30 else self.cfg.kalite_yaw_esik)
                    pitch_esik = (self.cfg.kalite_pitch_esik * 1.5
                                  if px_ref < 30 else self.cfg.kalite_pitch_esik)
                    if yaw > yaw_esik:
                        return False, 30.0, f"YAW({yaw:.0f}°)"
                    if pitch > pitch_esik:
                        return False, 30.0, f"PITCH({pitch:.0f}°)"
            except Exception:
                pass

        skor = min(100.0, (blur / 500.0) * 75 + min(w / 112.0, 2.0) * 25)
        return True, skor, "OK"


# ─────────────────────────────────────────────────────────────────────────────
# KAYAN EMBEDDİNG PENCERESİ  [v10.1 tabanlı + ekle_liste]
# ─────────────────────────────────────────────────────────────────────────────
class KayanEmbeddingPenceresi:
    def __init__(self, cfg: YarisConfig):
        self.cfg = cfg
        self._pencere: deque = deque(maxlen=cfg.emb_pencere_max)

    def ekle(self, emb: np.ndarray, kalite: float, frame_id: int):
        self._pencere.append((emb, kalite, frame_id))

    def ekle_liste(self, liste):
        """[v10.1] PartialLock'tan gelen embedding listesini toplu ekle."""
        for emb, kalite, fid in liste:
            self._pencere.append((emb, kalite, fid))

    def hazir_mi(self, yuz_px_genislik: int = None) -> bool:
        if yuz_px_genislik is not None and yuz_px_genislik < 15:
            gereken = 1
        elif yuz_px_genislik is not None and yuz_px_genislik < 20:
            gereken = 2
        elif yuz_px_genislik is not None and yuz_px_genislik < 40:
            gereken = 2
        elif yuz_px_genislik is not None and yuz_px_genislik < 60:
            gereken = max(2, self.cfg.emb_en_iyi_n // 4)
        elif yuz_px_genislik is not None and yuz_px_genislik < 100:
            gereken = max(3, self.cfg.emb_en_iyi_n // 3)
        else:
            gereken = self.cfg.emb_en_iyi_n

        yeterli = sum(1 for _, k, _ in self._pencere
                      if k > self.cfg.kalite_blur_esik * 0.5)
        if yeterli < gereken:
            yeterli = sum(1 for _, k, _ in self._pencere if k > 0)
        return yeterli >= gereken

    def en_iyi_embedding(self) -> Optional[np.ndarray]:
        if not self._pencere:
            return None
        sirali    = sorted(self._pencere, key=lambda x: x[1], reverse=True)
        secilen   = sirali[:self.cfg.emb_en_iyi_n]
        embler    = [e for e, _, _ in secilen]
        kaliteler = np.array([k for _, k, _ in secilen])
        agirliklar = kaliteler / (kaliteler.sum() + 1e-8)
        ort  = np.sum([a * e for a, e in zip(agirliklar, embler)], axis=0)
        norm = np.linalg.norm(ort)
        return ort / (norm + 1e-8) if norm > 0 else None

    def temizle(self):
        self._pencere.clear()

    def dolu_mu(self) -> bool:
        return len(self._pencere) >= self.cfg.emb_pencere_max

    def boyut(self) -> int:
        return len(self._pencere)


# ─────────────────────────────────────────────────────────────────────────────
# 6D KALMAN
# ─────────────────────────────────────────────────────────────────────────────
class KalmanTakip6D:
    def __init__(self):
        self.baslangic = False
        self.kf = None
        self._olustur()

    def _olustur(self):
        self.kf = cv2.KalmanFilter(6, 4)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ], np.float32)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], np.float32)
        self.kf.processNoiseCov = np.diag(
            [0.01, 0.01, 0.01, 0.01, 0.1, 0.1]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.5
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.baslangic = False

    def guncelle(self, cx, cy, w, h):
        olcum = np.array([[cx], [cy], [w], [h]], np.float32)
        if not self.baslangic:
            self.kf.statePre  = np.array(
                [[cx], [cy], [w], [h], [0], [0]], np.float32)
            self.kf.statePost = np.array(
                [[cx], [cy], [w], [h], [0], [0]], np.float32)
            self.baslangic = True
        self.kf.correct(olcum)
        return self._to_konum(self.kf.predict())

    def tahmin_et(self):
        if not self.baslangic:
            return None
        return self._to_konum(self.kf.predict())

    def _to_konum(self, s):
        cx = int(s[0].item())
        cy = int(s[1].item())
        w2 = max(10, int(s[2].item()) // 2)
        h2 = max(10, int(s[3].item()) // 2)
        return (cy - h2, cx + w2, cy + h2, cx - w2)

    def sifirla(self):
        self._olustur()


# ─────────────────────────────────────────────────────────────────────────────
# BYTETRACK
# ─────────────────────────────────────────────────────────────────────────────
class ByteTrackSarmalayici:
    def __init__(self, yolo_model, cfg: YarisConfig):
        self.model = yolo_model
        self.cfg   = cfg
        self.hedef_track_id   = None
        self.aktif            = False
        self._son_merkez      = None
        self._onceki_track_id = None
        print(f"✓ ByteTrack (re-assign: {cfg.bytetrack_re_assign_esik}px)")

    def aktiflesir(self, konum_hint=None):
        self.aktif = True
        self.hedef_track_id   = None
        self._onceki_track_id = None
        if konum_hint:
            ust, sag, alt, sol = konum_hint
            self._son_merkez = ((sol + sag) // 2, (ust + alt) // 2)
        else:
            self._son_merkez = None

    def guncelle(self, kare_bgr, imgsz: int = 640
                 ) -> Tuple[Optional[tuple], bool]:
        re_assign = False
        if not self.aktif:
            return None, False
        try:
            sonuclar = self.model.track(
                kare_bgr, persist=True, verbose=False,
                conf=self.cfg.bytetrack_conf,
                tracker="bytetrack.yaml", imgsz=imgsz)
        except Exception:
            return None, False

        if not sonuclar or sonuclar[0].boxes is None:
            return None, False
        boxes = sonuclar[0].boxes
        if boxes.id is None:
            return None, False

        liste = []
        for i, box in enumerate(boxes.xyxy):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            if (x2 - x1) < self.cfg.min_yuz_piksel:
                continue
            tid = int(boxes.id[i].item())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            liste.append({'tid': tid, 'x1': x1, 'y1': y1,
                          'x2': x2, 'y2': y2, 'cx': cx, 'cy': cy,
                          'alan': (x2 - x1) * (y2 - y1)})

        if not liste:
            return None, False

        konum = None
        if self.hedef_track_id is None:
            h = self._en_iyi(liste)
            if h:
                self._onceki_track_id = None
                self.hedef_track_id   = h['tid']
                self._son_merkez      = (h['cx'], h['cy'])
                konum = (h['y1'], h['x2'], h['y2'], h['x1'])
        else:
            bulundu = False
            for t in liste:
                if t['tid'] == self.hedef_track_id:
                    self._son_merkez = (t['cx'], t['cy'])
                    konum = (t['y1'], t['x2'], t['y2'], t['x1'])
                    bulundu = True
                    break
            if not bulundu and self._son_merkez:
                en_yakin = self._en_yakin(liste, self._son_merkez)
                if (en_yakin and
                        en_yakin['dist'] < self.cfg.bytetrack_re_assign_esik):
                    self._onceki_track_id = self.hedef_track_id
                    self.hedef_track_id   = en_yakin['tid']
                    self._son_merkez      = (en_yakin['cx'], en_yakin['cy'])
                    konum = (en_yakin['y1'], en_yakin['x2'],
                             en_yakin['y2'], en_yakin['x1'])
                    re_assign = (self._onceki_track_id is not None and
                                 self._onceki_track_id != self.hedef_track_id)
        return konum, re_assign

    def _en_iyi(self, liste):
        if self._son_merkez:
            en = self._en_yakin(liste, self._son_merkez)
            if en and en['dist'] < self.cfg.bytetrack_re_assign_esik:
                return en
        return max(liste, key=lambda t: t['alan'])

    def _en_yakin(self, liste, merkez):
        mx, my = merkez
        en, kisa = None, float('inf')
        for t in liste:
            d = abs(t['cx'] - mx) + abs(t['cy'] - my)
            if d < kisa:
                kisa, en = d, t
        if en:
            en = dict(en)
            en['dist'] = kisa
        return en

    def sifirla(self):
        self.hedef_track_id   = None
        self._onceki_track_id = None
        self.aktif            = False
        self._son_merkez      = None


# ─────────────────────────────────────────────────────────────────────────────
# ARCFACE VERİTABANI  [v10.0 augmentasyon + v10.1 multi-person]
# ─────────────────────────────────────────────────────────────────────────────
class ArcFaceVeritabani:
    def __init__(self, cfg: YarisConfig, dosya="hedefler_v10.pkl", gpu=True):
        self.cfg   = cfg
        self.dosya = dosya
        self.hedefler = {}
        self.app   = self._yukle(gpu)
        self.yukle()
        self._cache     = {}
        self._cache_fid = -1

    def _yukle(self, gpu):
        # InsightFace 1.0.1 uyumlu — providers argümanı kaldırıldı
        try:
            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=0 if gpu else -1, det_size=(640, 640))
            print(f"✓ ArcFace: {'GPU' if gpu else 'CPU'}")
            return app
        except Exception as e:
            print(f"⚠️  GPU başarısız ({e}), CPU'ya düşülüyor...")
            try:
                app = FaceAnalysis(name="buffalo_l")
                app.prepare(ctx_id=-1, det_size=(640, 640))
                print("✓ ArcFace: CPU")
                return app
            except Exception as e2:
                print(f"❌ ArcFace yüklenemedi: {e2}")
                exit(1)

    def kare_yuzleri_al(self, kare_bgr, frame_id=None):
        if frame_id is not None and frame_id == self._cache_fid:
            return self._cache.get('y', [])
        yuzler = self.app.get(kare_bgr)
        self._cache     = {'y': yuzler}
        self._cache_fid = frame_id if frame_id is not None else -1
        return yuzler

    def embedding_cikar_gorsel(self, gorsel_bgr):
        yuzler = self.app.get(gorsel_bgr)
        if not yuzler:
            return None
        return max(
            yuzler,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        ).normed_embedding

    # ── Augmentasyon: mesafe simülasyonları ──────────────────────────────
    def _uzak_mesafe_simulasyonu_uret(self, gorsel: np.ndarray) -> List[Tuple]:
        h, w = gorsel.shape[:2]
        simulasyonlar = []
        mesafe_params = [
            (0.55, 3,   5,  90, "~3-4m"),
            (0.38, 3,   8,  80, "~5-6m"),
            (0.25, 5,  10,  72, "~7m"),
            (0.18, 5,  12,  62, "~8m"),
            (0.18, 5, -20,  60, "~8m_karanlik"),
            (0.18, 9,  15,  58, "~8m_parlak"),
            (0.18, 9,   8,  55, "~8m_daha_blur"),
            (0.12, 7,  15,  65, "~10m"),
            (0.08, 9,  18,  60, "~12m"),
            (0.06, 11, 20,  55, "~14m"),
        ]
        for scale, blur_k, brightness, jpeg_q, etiket in mesafe_params:
            kw = max(80, int(w * scale))
            kh = max(80, int(h * scale))
            kucuk = cv2.resize(gorsel, (kw, kh), interpolation=cv2.INTER_AREA)
            kucuk = cv2.GaussianBlur(kucuk, (blur_k, blur_k), 0)
            _, enc = cv2.imencode('.jpg', kucuk,
                                  [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
            kucuk = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            kucuk = np.clip(kucuk.astype(np.int16) + brightness,
                            0, 255).astype(np.uint8)
            if scale <= 0.25:
                ikw, ikh = kucuk.shape[1], kucuk.shape[0]
                cx_d, cy_d = ikw / 2, ikh / 2
                fx = fy = max(ikw, ikh) * 1.1
                K = np.array([[fx, 0, cx_d], [0, fy, cy_d], [0, 0, 1]],
                              np.float32)
                dist = np.array([0.05, -0.02, 0, 0, 0], np.float32)
                kucuk = cv2.undistort(kucuk, K, dist)
            geri = cv2.resize(kucuk, (w, h), interpolation=cv2.INTER_LANCZOS4)
            simulasyonlar.append((geri, etiket))
        return simulasyonlar

    def hedef_ekle(self, gorsel_yolu: str, isim: str,
                   augmentasyon: bool = True) -> bool:
        gorsel = cv2.imread(gorsel_yolu)
        if gorsel is None:
            print(f"❌ Görsel açılamadı: {gorsel_yolu}")
            return False
        emb_orig = self.embedding_cikar_gorsel(gorsel)
        if emb_orig is None:
            print(f"❌ Yüz bulunamadı: {gorsel_yolu}")
            return False

        if not augmentasyon:
            self.hedefler[isim] = emb_orig
            self.kaydet()
            print(f"✓ '{isim}' eklendi (1 embedding)")
            return True

        print(f"⏳ '{isim}' için mesafe simülasyonları üretiliyor...")
        tum = [emb_orig]
        print(f"   [1] orijinal → ✓")

        for i, (sim, etiket) in enumerate(
                self._uzak_mesafe_simulasyonu_uret(gorsel)):
            emb = self.embedding_cikar_gorsel(sim)
            if emb is None:
                fb = f"aug_hata_{isim}_{i}.jpg"
                cv2.imwrite(fb, sim)
                print(f"   [{i+2}] {etiket} → ❌ (kaydedildi: {fb})")
                continue
            if any(float(np.dot(emb, v)) > 0.95 for v in tum):
                print(f"   [{i+2}] {etiket} → ⚠️  benzer, atlandı")
            else:
                tum.append(emb)
                print(f"   [{i+2}] {etiket} → ✓")

        self.hedefler[isim] = tum
        self.kaydet()
        print(f"\n✅ '{isim}' eklendi — {len(tum)} embedding")
        if len(tum) < 5:
            print("  ⚠️  Az embedding! Daha net/büyük fotoğraf önerilir.")
        return True

    def hedef_sil(self, isim):
        if isim in self.hedefler:
            del self.hedefler[isim]
            self.kaydet()
            print(f"✓ Silindi: {isim}")

    def kaydet(self):
        with open(self.dosya, 'wb') as f:
            pickle.dump(self.hedefler, f)

    def yukle(self):
        if os.path.exists(self.dosya):
            with open(self.dosya, 'rb') as f:
                self.hedefler = pickle.load(f)
            print(f"✓ DB yüklendi: {list(self.hedefler.keys())}")

    def tanimla(self, embedding, mesafe_metre=None,
                yuz_piksel_genislik=None,
                debug_skor=False,
                override_hard_reject=None,
                gorunen_yuz_sayisi=1) -> Tuple[Optional[str], float]:
        """
        gorunen_yuz_sayisi: [v10.1] Tanıma anında kaç yüz görünüyor?
        5-6 kişilik ortamda fark eşiği buna göre sıkılaştırılır.
        """
        if not self.hedefler or embedding is None:
            return None, 0.0

        c = self.cfg

        # Mesafe bazlı eşik
        if mesafe_metre is not None:
            if mesafe_metre < 2.5:   esik = c.arcface_esik_yakin
            elif mesafe_metre < 5.0: esik = c.arcface_esik_orta
            elif mesafe_metre < 8.0: esik = c.arcface_esik_uzak
            else:                    esik = c.arcface_esik_ultra
        else:
            esik = c.arcface_esik_varsayilan

        # Piksel bazlı indirim
        if yuz_piksel_genislik is not None and yuz_piksel_genislik < 80:
            indirim = c.uzak_esik_indirimi * (
                1.0 - yuz_piksel_genislik / 80.0)
            esik = max(c.arcface_hard_reject + 0.02, esik - indirim)
        if yuz_piksel_genislik is not None and yuz_piksel_genislik < 20:
            esik = max(c.arcface_hard_reject + 0.01, esik - 0.12)
        elif yuz_piksel_genislik is not None and yuz_piksel_genislik < 35:
            esik = max(c.arcface_hard_reject + 0.01, esik - 0.05)
        elif yuz_piksel_genislik is not None and yuz_piksel_genislik < 45:
            esik = max(c.arcface_hard_reject + 0.01, esik - 0.03)

        # Skor hesapla
        skorlar = {}
        for isim, emb in self.hedefler.items():
            if isinstance(emb, list):
                skorlar[isim] = max(float(np.dot(embedding, e)) for e in emb)
            else:
                skorlar[isim] = float(np.dot(embedding, emb))

        sirali   = sorted(skorlar.items(), key=lambda x: x[1], reverse=True)
        en_isim, en_skor = sirali[0]

        # Dinamik hard_reject
        base_hard = (override_hard_reject
                     if override_hard_reject is not None
                     else c.arcface_hard_reject)
        dhr = base_hard
        if yuz_piksel_genislik is not None:
            if yuz_piksel_genislik < 20:   dhr = max(0.10, base_hard - 0.10)
            elif yuz_piksel_genislik < 30: dhr = max(0.11, base_hard - 0.09)
            elif yuz_piksel_genislik < 40: dhr = max(0.12, base_hard - 0.08)
            elif yuz_piksel_genislik < 60: dhr = max(0.13, base_hard - 0.06)
            elif yuz_piksel_genislik < 80: dhr = max(0.14, base_hard - 0.04)

        if debug_skor:
            print(f"  [SKOR] {en_isim}={en_skor:.3f} esik={esik:.3f} "
                  f"dhr={dhr:.3f} px={yuz_piksel_genislik} "
                  f"yüz_sayisi={gorunen_yuz_sayisi}")

        if en_skor < dhr:
            if debug_skor:
                print(f"  [RED] hard_reject: {en_skor:.3f} < {dhr:.3f}")
            return None, 0.0
        if en_skor < esik:
            if debug_skor:
                print(f"  [RED] esik: {en_skor:.3f} < {esik:.3f}")
            return None, 0.0

        # [v10.1] Multi-person dinamik fark eşiği
        if len(sirali) >= 2:
            if gorunen_yuz_sayisi >= 3:
                fark_esik = c.multiperson_fark_3plus
            elif gorunen_yuz_sayisi == 2:
                fark_esik = c.multiperson_fark_2yuz
            else:
                fark_esik = c.multiperson_fark_1yuz
            if c.yarisma_modu:
                fark_esik += 0.02

            ikinci = sirali[1][1]
            if (en_skor - ikinci) < fark_esik:
                if debug_skor:
                    print(f"  [RED] fark: {en_skor-ikinci:.3f} < "
                          f"{fark_esik:.3f} ({gorunen_yuz_sayisi} yüz)")
                return None, 0.0

        return en_isim, en_skor * 100

    @property
    def bos(self):
        return not self.hedefler


# ─────────────────────────────────────────────────────────────────────────────
# ASENKRON EMBEDDİNG — ThreadPoolExecutor  [v10.1'den]
# ─────────────────────────────────────────────────────────────────────────────
class AsenkronEmbedding:
    def __init__(self, db: ArcFaceVeritabani, sr: SuperResolution,
                 filtre: YuzKaliteFiltrer, cfg: YarisConfig,
                 max_workers: int = 2):
        self.db      = db
        self.sr      = sr
        self.filtre  = filtre
        self.cfg     = cfg
        self._executor        = ThreadPoolExecutor(max_workers=max_workers)
        self._lock            = threading.Lock()
        self._bekleyen_future = None
        self._sonuc_emb       = None
        self._sonuc_kalite    = 0.0
        self._dur             = False

    def _isle(self, kare_bgr, konum):
        ust, sag, alt, sol = konum
        pad = 12
        h, w = kare_bgr.shape[:2]
        y1, y2 = max(0, ust - pad), min(h, alt + pad)
        x1, x2 = max(0, sol - pad), min(w, sag + pad)
        yuz_crop = kare_bgr[y1:y2, x1:x2]
        if yuz_crop.size == 0:
            return None, 0.0

        px_orig = sag - sol
        if self.sr.gerekli_mi(yuz_crop.shape[1]):
            yuz_crop = self.sr.buyut(yuz_crop)

        try:
            yuzler  = self.db.app.get(yuz_crop)
            ins_yuz = yuzler[0] if yuzler else None
        except Exception:
            ins_yuz = None

        gecti, kalite, _ = self.filtre.degerlendir(
            yuz_crop, ins_yuz, yuz_px_genislik=px_orig)
        if not gecti:
            return None, 0.0

        if ins_yuz is not None:
            return ins_yuz.normed_embedding, kalite

        cx, cy = (sol + sag) // 2, (ust + alt) // 2
        for f in self.db.app.get(kare_bgr):
            fx1, fy1, fx2, fy2 = [int(v) for v in f.bbox]
            if (abs((fx1 + fx2) // 2 - cx) < 70 and
                    abs((fy1 + fy2) // 2 - cy) < 70):
                return f.normed_embedding, kalite * 0.7
        return None, 0.0

    def istek_gonder(self, kare, konum, frame_id):
        if self._dur:
            return
        kare_k = kare.copy()
        with self._lock:
            if (self._bekleyen_future and
                    not self._bekleyen_future.done()):
                self._bekleyen_future.cancel()
            future = self._executor.submit(self._isle, kare_k, konum)
            self._bekleyen_future = future

        def _cb(f):
            if f.cancelled() or f.exception():
                return
            emb, kalite = f.result()
            if emb is not None:
                with self._lock:
                    self._sonuc_emb    = emb
                    self._sonuc_kalite = kalite
        future.add_done_callback(_cb)

    def sonuc_al(self) -> Tuple[Optional[np.ndarray], float]:
        with self._lock:
            emb, kalite = self._sonuc_emb, self._sonuc_kalite
            self._sonuc_emb    = None
            self._sonuc_kalite = 0.0
            return emb, kalite

    def temizle(self):
        with self._lock:
            if self._bekleyen_future:
                self._bekleyen_future.cancel()
            self._bekleyen_future = None
            self._sonuc_emb       = None
            self._sonuc_kalite    = 0.0

    def durdur(self):
        self._dur = True
        self._executor.shutdown(wait=False)


# ─────────────────────────────────────────────────────────────────────────────
# RE-VERİFİKASYON İŞÇİSİ  [v10.1 tabanlı, _beklenen_px typo düzeltildi]
# ─────────────────────────────────────────────────────────────────────────────
class ReVerifIsci:
    def __init__(self, db: ArcFaceVeritabani, sr: SuperResolution,
                 filtre: YuzKaliteFiltrer, cfg: YarisConfig,
                 emb_pencere: KayanEmbeddingPenceresi = None):
        self._async       = AsenkronEmbedding(db, sr, filtre, cfg, max_workers=1)
        self.db           = db
        self.cfg          = cfg
        self.emb_pencere  = emb_pencere
        self._bekleniyor  = False
        self._beklenen_isim   = None
        self._beklenen_mesafe = None
        self._beklenen_px     = None   # typo düzeltildi: _beklened → _beklenen
        self._baslangic_frame = 0

    def baslat(self, kare, konum, aktif_isim, mesafe, px, frame_id):
        if self._bekleniyor:
            return
        self._bekleniyor      = True
        self._beklenen_isim   = aktif_isim
        self._beklenen_mesafe = mesafe
        self._beklenen_px     = px
        self._baslangic_frame = frame_id
        self._async.istek_gonder(kare, konum, frame_id)

    def kontrol_et(self, current_frame_id: int = 0
                   ) -> Tuple[Optional[str], float, bool]:
        if not self._bekleniyor:
            return None, 0.0, False
        if (current_frame_id - self._baslangic_frame >
                self.cfg.reverifikasyon_timeout_frame):
            print("  ⏱️  ReVerif timeout")
            self._bekleniyor = False
            self._async.temizle()
            return None, 0.0, False

        emb, _ = self._async.sonuc_al()
        if emb is None and self.emb_pencere is not None:
            emb = self.emb_pencere.en_iyi_embedding()
        if emb is None:
            return None, 0.0, False

        self._bekleniyor = False
        isim, guven = self.db.tanimla(
            emb, self._beklenen_mesafe, self._beklenen_px)

        # Tolerans katmanı
        if isim is None:
            isim_t, guven_t = self.db.tanimla(
                emb, self._beklenen_mesafe, self._beklenen_px,
                override_hard_reject=0.12)
            if isim_t == self._beklenen_isim:
                isim  = isim_t
                guven = guven_t * 0.9
                print(f"  [ReVerif tolerans] {isim} ({guven:.1f}%)")

        return isim, guven, True

    def temizle(self):
        self._bekleniyor      = False
        self._baslangic_frame = 0
        self._async.temizle()

    def durdur(self):
        self._async.durdur()


# ─────────────────────────────────────────────────────────────────────────────
# ANA SİSTEM v10.2
# ─────────────────────────────────────────────────────────────────────────────
class YarisV10:
    def __init__(self, arduino_port=None, tensorrt=False,
                 gpu=True, asenkron=True,
                 cfg: YarisConfig = None):

        self.cfg = cfg or YarisConfig()
        c = self.cfg

        c.kamera_profil_uygula()
        if c.jetson_orin:
            jetson_orin_nano_optimizasyonu()
            jetson_guc_modu_kontrol()
        if c.yarisma_modu:
            c.yarisma_esikleri_uygula()
        if c.manken_modu:
            c.partial_lock_max_frame  = 150
            c.multiperson_fark_3plus += 0.02
            print('🎭 Manken modu: partial_lock=150f, multi_fark_3+=+0.02')

        self.clahe = cv2.createCLAHE(
            clipLimit=c.clahe_clip, tileGridSize=c.clahe_tile)

        self.yolo       = self._yolo_yukle(c.yolo_model, tensorrt, c)
        self.db         = ArcFaceVeritabani(c, gpu=gpu)
        self.sr         = SuperResolution(c)
        self.filtre     = YuzKaliteFiltrer(c)
        self.emb_pencere = KayanEmbeddingPenceresi(c)

        self.asenkron    = asenkron
        self.async_emb   = (
            AsenkronEmbedding(self.db, self.sr, self.filtre, c, max_workers=2)
            if asenkron else None)
        self._sync_worker = (
            None if asenkron
            else AsenkronEmbedding(self.db, self.sr, self.filtre, c,
                                   max_workers=1))

        self.reverif_isci = ReVerifIsci(
            self.db, self.sr, self.filtre, c, self.emb_pencere)

        self.bytetrack   = ByteTrackSarmalayici(self.yolo, c)
        self.kalman      = KalmanTakip6D()
        self.kalman_aktif = False

        # ── Durum değişkenleri ────────────────────────────────────────────
        self.kilit_durumu  = "ARAMA"
        self.aktif_isim    = None
        self.aktif_konum   = None
        self.hedef_guven   = 0.0
        self.hedef_bulundu = False
        self.hedef_kayip_sayac          = 0
        self.dogrulama_sayac            = 0
        self._onay_sayac                = 0
        self._aday_isim                 = None   # [v10.3] kümülatif oylama
        self.reverifikasyon_sayac       = 0
        self.reverifikasyon_basarisizlik = 0
        self._reverif_aktif             = False
        self.konum_gecmisi: Deque = deque(maxlen=10)
        self.son_mesafe_metre = None
        self.lazer_aktif      = False
        self._lazer_gecikme_sayac = 0  # [v10.2] lazer ateşleme gecikmesi
        self.mod              = "ARANIYOR"
        self._frame_id        = 0
        self._guncel_imgsz    = c.imgsz_ultra
        self._guncel_yuz_px   = 999
        self._yolo_son_frame  = 0
        self._gorunen_yuz_sayisi = 1

        # ── [v10.1] PartialLock & Tarama Yön Belleği ─────────────────────
        self._partial_lock: Optional[PartialLockKayit] = None
        self._tarama_yon   = TaramaYonBellegi(pencere=15)
        self._yuz_x_gecmis: Deque[int] = deque(maxlen=5)

        self.arduino       = None
        self.arduino_bagli = False
        if arduino_port and SERIAL_MEVCUT:
            self._arduino_baglan(arduino_port)

        self.fps_gercek    = 0
        self.isleme_suresi = 0
        self._islenmis_kare_cache = None   # [v10.4] on_isle_interval önbelleği

    # ─────────────────────────────────────────────────────────────────────
    # YOLO YÜKLEYİCİ  [v10.1: Orin Nano için DLA yok]
    # ─────────────────────────────────────────────────────────────────────
    def _yolo_yukle(self, model_yolu, tensorrt, cfg):
        if not tensorrt:
            print(f"⏳ YOLO yükleniyor: {model_yolu}")
            m = YOLO(model_yolu)
            print("✓ YOLO hazır")
            return m
        base   = os.path.splitext(model_yolu)[0]
        engine = f"{base}.engine"
        if os.path.exists(engine):
            print(f"✓ TensorRT engine: {engine}")
            return YOLO(engine)
        print("⏳ TensorRT engine oluşturuluyor "
              "(Orin Nano'da ilk seferde ~8-12 dk)...")
        # Nano: DLA yok, workspace=2GB, dynamic=False
        YOLO(model_yolu).export(
            format="engine",
            half=True,
            device=0,
            dynamic=False,
            workspace=cfg.trt_workspace_gb,
            simplify=True,
        )
        print("✓ TensorRT engine hazır")
        return YOLO(engine)

    # ─────────────────────────────────────────────────────────────────────
    # [v10.1] PARTIAL LOCK YÖNETİMİ
    # ─────────────────────────────────────────────────────────────────────
    def _partial_lock_kontrol_et(self, yuz_merkez_x: int,
                                  yuz_merkez_y: int) -> bool:
        c  = self.cfg
        pl = self._partial_lock
        if pl is None or not c.partial_lock_aktif:
            return False

        if self._frame_id > pl.gecerlilik_frame:
            if c.debug_partial:
                print(f"  [PL] EXPIRE frame {self._frame_id} > "
                      f"{pl.gecerlilik_frame}")
            self._partial_lock = None
            return False

        if abs(yuz_merkez_x - pl.son_konum_x) > c.partial_lock_bolge_esik:
            if c.debug_partial:
                print(f"  [PL] Yüz çok uzakta "
                      f"(Δx={abs(yuz_merkez_x-pl.son_konum_x)} > "
                      f"{c.partial_lock_bolge_esik})")
            return False

        # Tarama yönüyle uyumsuzluk mu?
        dominant = self._tarama_yon.dominant_yon()
        if dominant == "SOL" and pl.cikis_yonu == "SAG":
            if c.debug_partial:
                print(f"  [PL] Yön uyumsuz: tarama={dominant} çıkış={pl.cikis_yonu}")
            return False
        if dominant == "SAG" and pl.cikis_yonu == "SOL":
            if c.debug_partial:
                print(f"  [PL] Yön uyumsuz: tarama={dominant} çıkış={pl.cikis_yonu}")
            return False

        if c.debug_partial:
            print(f"  [PL] ✓ EŞLEŞTİ! aday={pl.aday_isim} "
                  f"skor={pl.en_iyi_skor:.3f} emb={len(pl.embeddingler)}")
        self.emb_pencere.ekle_liste(pl.embeddingler)
        return True

    def _partial_lock_olustur(self, embeddingler: list,
                               en_iyi_skor: float, aday_isim: str,
                               merkez_x: int, merkez_y: int,
                               px_genislik: int):
        c = self.cfg
        if not c.partial_lock_aktif or not embeddingler:
            return

        min_umut = c.arcface_esik_ultra * c.partial_lock_skor_esigi
        if en_iyi_skor < min_umut:
            if c.debug_partial:
                print(f"  [PL] Skor yetersiz: {en_iyi_skor:.3f} < {min_umut:.3f}")
            return

        cikis_yonu = "BELIRSIZ"
        if len(self._yuz_x_gecmis) >= 2:
            x_ler  = list(self._yuz_x_gecmis)
            hareket = x_ler[-1] - x_ler[0]
            if hareket < -30:  cikis_yonu = "SOL"
            elif hareket > 30: cikis_yonu = "SAG"

        self._partial_lock = PartialLockKayit(
            embeddingler=list(embeddingler),
            en_iyi_skor=en_iyi_skor,
            aday_isim=aday_isim,
            son_konum_x=merkez_x,
            son_konum_y=merkez_y,
            cikis_yonu=cikis_yonu,
            olusturma_frame=self._frame_id,
            gecerlilik_frame=self._frame_id + c.partial_lock_max_frame,
            px_genislik=px_genislik,
        )
        if c.debug_partial:
            print(f"  [PL] OLUŞTURULDU: aday={aday_isim} "
                  f"skor={en_iyi_skor:.3f} çıkış={cikis_yonu} "
                  f"geçerli={c.partial_lock_max_frame}f")

    def _partial_lock_en_iyi_skor_al(self) -> Tuple[float, str]:
        emb = self.emb_pencere.en_iyi_embedding()
        if emb is None:
            return 0.0, ""
        isim, guven = self.db.tanimla(
            emb, self.son_mesafe_metre, self._guncel_yuz_px, debug_skor=False)
        if isim is None:
            for h_isim, h_emb in self.db.hedefler.items():
                if isinstance(h_emb, list):
                    s = max(float(np.dot(emb, e)) for e in h_emb)
                else:
                    s = float(np.dot(emb, h_emb))
                return s, h_isim
            return 0.0, ""
        return guven / 100.0, isim

    # ─────────────────────────────────────────────────────────────────────
    # YOLO ADAPTIVE POLLING  [v10.1'den]
    # ─────────────────────────────────────────────────────────────────────
    def _yolo_calistir_mi(self) -> bool:
        c = self.cfg
        if self.kilit_durumu == "ARAMA":
            interval = c.yolo_arama_interval
        elif self.kilit_durumu == "YUZ_KILITLENDI":
            interval = c.yolo_kilitli_interval
        else:
            interval = c.yolo_dogrulandi_interval
        return (self._frame_id - self._yolo_son_frame) >= interval

    # ─────────────────────────────────────────────────────────────────────
    # imgsz seçimi
    # ─────────────────────────────────────────────────────────────────────
    def _imgsz_sec(self, alan_oran=None) -> int:
        c = self.cfg
        if self.son_mesafe_metre is not None:
            m = self.son_mesafe_metre
            if m < 3.0: return c.imgsz_yakin
            if m < 6.0: return c.imgsz_orta
            if m < 8.0: return c.imgsz_uzak
            return c.imgsz_ultra
        if alan_oran is not None:
            if alan_oran > 0.05:  return c.imgsz_yakin
            if alan_oran > 0.01:  return c.imgsz_orta
            if alan_oran > 0.003: return c.imgsz_uzak
        return c.imgsz_ultra

    # ─────────────────────────────────────────────────────────────────────
    # YÜZLER BUL — ROI + multi-person  [v10.0 ROI + v10.1 yüz sayısı]
    # ─────────────────────────────────────────────────────────────────────
    def _yuz_bul_hizli(self, kare_bgr) -> Tuple[Optional[tuple], int]:
        """
        ROI varsa önce ortadan tara (yan manken koruması — v10.0).
        Görünen yüz sayısını da döndür (multi-person güvenlik — v10.1).
        """
        h, w = kare_bgr.shape[:2]

        if self.cfg.roi_aktif:
            mx = int(w * self.cfg.roi_margin_x)
            my = int(h * self.cfg.roi_margin_y)
            roi = kare_bgr[my:h - my, mx:w - mx]
            boxes_roi = self._multiscale_tespit(
                roi, self.cfg.imgsz_ultra, self.cfg.yolo_conf_arama)
            if boxes_roi:
                # ROI koordinatını tam kare koordinatına çevir
                tam_boxes = [(x1 + mx, y1 + my, x2 + mx, y2 + my, sc)
                             for x1, y1, x2, y2, sc in boxes_roi]
                self._gorunen_yuz_sayisi = len(tam_boxes)
                en = max(tam_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                return (en[1], en[2], en[3], en[0]), self._gorunen_yuz_sayisi

        # ROI'de bulunamadı ya da roi_aktif=False → tam kare
        boxes = self._multiscale_tespit(
            kare_bgr, self.cfg.imgsz_ultra, self.cfg.yolo_conf_arama)
        if not boxes:
            return None, 0
        self._gorunen_yuz_sayisi = len(boxes)
        en = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x1, y1, x2, y2 = en[:4]
        return (y1, x2, y2, x1), self._gorunen_yuz_sayisi

    # ─────────────────────────────────────────────────────────────────────
    # YOLO tespitleri
    # ─────────────────────────────────────────────────────────────────────
    def _multiscale_tespit(self, kare_bgr, imgsz, conf):
        h, w = kare_bgr.shape[:2]
        boxes = self._yolo_tespit(kare_bgr, imgsz, conf)
        if not self.cfg.multiscale_aktif:
            return boxes
        ust_y = int(h * (1 - self.cfg.multiscale_overlap))
        boxes += self._yolo_tespit(kare_bgr[:ust_y, :], imgsz, conf)
        alt_y = int(h * self.cfg.multiscale_overlap)
        boxes += [(x1, y1 + alt_y, x2, y2 + alt_y, sc)
                  for x1, y1, x2, y2, sc in
                  self._yolo_tespit(kare_bgr[alt_y:, :], imgsz, conf)]
        return self._nms(boxes)

    def _yolo_tespit(self, kare, imgsz, conf, ox=0, oy=0):
        try:
            r = self.yolo(kare, verbose=False, conf=conf, imgsz=imgsz)[0]
        except Exception:
            return []
        if r.boxes is None:
            return []
        boxes = []
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            c = float(box.conf[0])
            if (x2 - x1) < self.cfg.min_yuz_piksel:
                continue
            boxes.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy, c))
        return boxes

    def _nms(self, boxes, iou_esik=0.4):
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        sonuc = []
        while boxes:
            best = boxes.pop(0)
            sonuc.append(best)
            boxes = [b for b in boxes if self._iou(best, b) < iou_esik]
        return sonuc

    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a[:4]
        bx1, by1, bx2, by2 = b[:4]
        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / (union + 1e-6)

    # ─────────────────────────────────────────────────────────────────────
    # Ön işleme
    # ─────────────────────────────────────────────────────────────────────
    def _dijital_zoom(self, kare):
        z = self.cfg.dijital_zoom
        if z <= 1.0:
            return kare
        h, w = kare.shape[:2]
        nh, nw = int(h / z), int(w / z)
        y1, x1 = (h - nh) // 2, (w - nw) // 2
        return cv2.resize(kare[y1:y1 + nh, x1:x1 + nw], (w, h),
                          interpolation=cv2.INTER_LINEAR)

    def _on_isle(self, kare):
        kernel = np.array([[0, -0.7, 0],
                           [-0.7, 3.8, -0.7],
                           [0, -0.7, 0]], np.float32)
        keskin = np.clip(
            cv2.filter2D(kare, -1, kernel), 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(keskin, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # ─────────────────────────────────────────────────────────────────────
    # Arduino / servo / lazer
    # ─────────────────────────────────────────────────────────────────────
    def _arduino_baglan(self, port, baud=9600):
        try:
            self.arduino = serial.Serial(port, baud, timeout=1)
            time.sleep(2)
            self.arduino_bagli = True
            print(f"✓ Arduino: {port}")
        except Exception as e:
            print(f"❌ Arduino: {e}")

    def lazer_kontrol(self, durum):
        if not self.arduino_bagli:
            return
        try:
            self.arduino.write(
                b"LAZER:ON\n" if durum else b"LAZER:OFF\n")
            self.lazer_aktif = durum
        except Exception:
            pass

    def servo_gonder(self, yon, aci, mesafe_kat):
        if not self.arduino_bagli:
            return
        try:
            self.arduino.write(
                f"YON:{yon},ACI:{aci},MESAFE:{mesafe_kat}\n".encode())
        except Exception:
            pass


    # ─────────────────────────────────────────────────────────────────────
    # Konum / servo
    # ─────────────────────────────────────────────────────────────────────
    def _konum_analiz(self, konum, boyut):
        ust, sag, alt, sol = konum
        yh, yw = boyut
        mx, my = (sol + sag) // 2, (ust + alt) // 2
        return {
            'x_oran':    (mx - yw / 2) / (yw / 2),
            'y_oran':    (my - yh / 2) / (yh / 2),
            'alan_oran': ((sag - sol) * (alt - ust)) / (yw * yh),
            'merkez_x':  mx,
            'merkez_y':  my,
        }

    def _servo_hesapla(self, kd):
        x    = kd['x_oran']
        m    = self.son_mesafe_metre
        k    = (self.cfg.servo_k_uzak
                if (m and m > 5.0) else self.cfg.servo_k_yakin)
        band = self.cfg.servo_merkez_band
        if x < -band:   yon, aci = 'SOL', int(abs(x) * k)
        elif x > band:  yon, aci = 'SAG', int(abs(x) * k)
        else:           yon, aci = 'MERKEZ', 0
        if (m and m < 2.0) or kd['alan_oran'] > 0.15:  kat = 'YAKIN'
        elif (m and m < 5.0) or kd['alan_oran'] > 0.01: kat = 'ORTA'
        else:                                            kat = 'UZAK'
        return {'yon': yon, 'aci': aci, 'mesafe': kat}

    # ─────────────────────────────────────────────────────────────────────
    # Tam sıfırla  [v10.1 — PartialLock korunur]
    # ─────────────────────────────────────────────────────────────────────
    def _tam_sifirla(self):
        if self.lazer_aktif:
            self.lazer_kontrol(False)
        self.kalman.sifirla()
        self.son_mesafe_metre  = None
        self.aktif_isim        = None
        self.hedef_guven       = 0.0
        self.kilit_durumu      = "ARAMA"
        self.emb_pencere.temizle()
        if self.async_emb:
            self.async_emb.temizle()
        if self._sync_worker:
            self._sync_worker.temizle()
        self.reverif_isci.temizle()
        self.dogrulama_sayac              = 0
        self._onay_sayac                  = 0
        self._aday_isim                   = None   # [v10.3]
        self.hedef_kayip_sayac            = 0
        self.reverifikasyon_sayac         = 0
        self.reverifikasyon_basarisizlik  = 0
        self._reverif_aktif               = False
        self.mod                          = "ARANIYOR"
        self.bytetrack.sifirla()
        self._guncel_imgsz   = self.cfg.imgsz_ultra
        self._guncel_yuz_px  = 999
        self._yolo_son_frame = 0
        self._yuz_x_gecmis.clear()
        self._lazer_gecikme_sayac = 0
        # PartialLock'u silme! Tarama sırasında sıfırlanırsa hafıza kaybolur.
        # Sadece expire ile temizlenir.

    # ─────────────────────────────────────────────────────────────────────
    # ANA DÖNGÜ
    # ─────────────────────────────────────────────────────────────────────
    def baslat(self, gosterim=True, kayit=False):
        if self.db.bos:
            print("❌ Veritabanında hedef yok!")
            print("   python3 yaris_v10.py --hedef-ekle foto.jpg --isim Ali")
            return

        c = self.cfg
        kam = cv2.VideoCapture(c.kamera_indeks, cv2.CAP_V4L2)
        if not kam.isOpened():
            kam = cv2.VideoCapture(c.kamera_indeks)
        if not kam.isOpened():
            print(f"❌ Kamera {c.kamera_indeks} açılamadı!")
            return
        kam.set(cv2.CAP_PROP_FRAME_WIDTH,  c.kamera_genislik)
        kam.set(cv2.CAP_PROP_FRAME_HEIGHT, c.kamera_yukseklik)
        kam.set(cv2.CAP_PROP_FPS,          c.kamera_fps)
        kam.set(cv2.CAP_PROP_BUFFERSIZE,   2)
        kam.set(cv2.CAP_PROP_AUTOFOCUS,    0)
        kam.set(cv2.CAP_PROP_FOCUS,        10)
        if c.mjpeg_aktif:
            kam.set(cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*'MJPG'))
            print("✓ Kamera MJPEG")

        profil_bilgi = KAMERA_PROFILLERI.get(
            c.kamera_profil, {}).get("aciklama", c.kamera_profil)
        print("\n" + "=" * 70)
        print("🏁 YARIŞ v10.3 — Kümülatif Oylama + 8m Hızlandırma")
        print(f"📷 Kamera: {profil_bilgi} "
              f"({c.kamera_genislik}x{c.kamera_yukseklik} @{c.kamera_fps}fps)")
        print(f"👥 Hedefler: {list(self.db.hedefler.keys())}")
        for isim, emb in self.db.hedefler.items():
            sayi = len(emb) if isinstance(emb, list) else 1
            print(f"   {isim}: {sayi} embedding")
        print(f"🏆 Yarışma: {'✅' if c.yarisma_modu else 'kapalı'}")
        print(f"🔒 PartialLock: {'AÇIK' if c.partial_lock_aktif else 'KAPALI'} "
              f"(expire={c.partial_lock_max_frame}f, "
              f"bölge={c.partial_lock_bolge_esik}px)")
        print(f"👥 Multi-person: "
              f"1yüz={c.multiperson_fark_1yuz} "
              f"2yüz={c.multiperson_fark_2yuz} "
              f"3+yüz={c.multiperson_fark_3plus}")
        print(f"🎯 ROI: {'AÇIK' if c.roi_aktif else 'KAPALI'}")
        print(f"🛡️  ArcFace: hard_reject={c.arcface_hard_reject} | "
              f"uzak={c.arcface_esik_uzak} | ultra={c.arcface_esik_ultra}")
        print(f"📦 Emb: periyot={c.emb_al_periyot} "
              f"pencere={c.emb_pencere_max} "
              f"min_onay={c.kilitlenme_min_onay}")
        print(f"⏱️  Lazer gecikme: {c.lazer_gecikme_frame} frame")
        print(f"🔄 ReVerif: {c.reverifikasyon_periyot}f | "
              f"esik: {c.reverifikasyon_basarisizlik_esik}")
        print(f"🤖 YOLO: {c.yolo_model}  |  "
              f"SR: {self.sr.mod}  |  "
              f"Kalite: {'AÇIK' if c.kalite_aktif else 'KAPALI'}")
        print(f"⚡ Multiscale: {'AÇIK' if c.multiscale_aktif else 'KAPALI'}  |  "
              f"Async: {'AÇIK' if self.asenkron else 'KAPALI'}  |  "
              f"Kamera {c.kamera_indeks}")
        print(f"🚀 FPS opt: isle_interval={c.on_isle_interval}  |  "
              f"arama_yolo={c.arama_yolo_interval}  |  "
              f"bt_imgsz_kilitli={c.bytetrack_imgsz_kilitli}")
        print("=" * 70 + "\n")

        yazici = None
        if kayit:
            yazici = cv2.VideoWriter(
                'kayit_v10.avi',
                cv2.VideoWriter_fourcc(*'XVID'),
                20.0, (c.kamera_genislik, c.kamera_yukseklik + 140))

        fps_s, fps_b = 0, time.time()
        son_komut  = None
        konum_data = None

        try:
            while True:
                t0 = time.time()
                self._frame_id += 1

                ret, kare = kam.read()
                if not ret:
                    break

                # [v10.4] _on_isle her on_isle_interval frame'de bir çalışır.
                # Araya giren frame'lerde önceki işlenmiş kare yeniden kullanılır.
                # interval=2 → ~%50 filter2D+CLAHE CPU kazancı, tanımaya etkisi minimal.
                if (self._islenmis_kare_cache is None or
                        self._frame_id % self.cfg.on_isle_interval == 0):
                    self._islenmis_kare_cache = self._dijital_zoom(
                        self._on_isle(kare))
                kare_is = self._islenmis_kare_cache

                bulunan_konum = None
                self.hedef_bulundu = False
                self.kalman_aktif  = False

                # ── ARAMA MODU ────────────────────────────────────────────
                if self.kilit_durumu == "ARAMA":
                    self.mod = "ARANIYOR"
                    # [v10.4] ARAMA'da multiscale YOLO her arama_yolo_interval
                    # frame'de bir çalışır. Yüz yokken 3× YOLO her frame çok pahalı.
                    if self._frame_id % self.cfg.arama_yolo_interval != 0:
                        yuz_konum, yuz_sayisi = None, 0
                    else:
                        yuz_konum, yuz_sayisi = self._yuz_bul_hizli(kare_is)

                    if yuz_konum:
                        ust, sag, alt, sol = yuz_konum
                        merkez_x = (sol + sag) // 2
                        merkez_y = (ust + alt) // 2
                        yuz_px   = sag - sol
                        alan_oran = ((sag-sol)*(alt-ust)) / (
                            kare.shape[0] * kare.shape[1])
                        self._guncel_imgsz       = self._imgsz_sec(alan_oran)
                        self._guncel_yuz_px      = yuz_px
                        self._gorunen_yuz_sayisi = yuz_sayisi

                        # [v10.1] PartialLock kontrolü
                        partial_eslesti = self._partial_lock_kontrol_et(
                            merkez_x, merkez_y)

                        self.bytetrack.aktiflesir(konum_hint=yuz_konum)
                        self.aktif_konum = yuz_konum
                        if not partial_eslesti:
                            self.emb_pencere.temizle()
                        if self.async_emb:
                            self.async_emb.temizle()
                        self.dogrulama_sayac   = 0
                        self._onay_sayac       = 0
                        self._aday_isim        = None   # [v10.3]
                        self.kilit_durumu      = "YUZ_KILITLENDI"
                        self.hedef_kayip_sayac = 0
                        self.mod = ("YÜZ KİLİTLENDİ"
                                    + (" [PL]" if partial_eslesti else ""))
                        bulunan_konum       = yuz_konum
                        self.hedef_bulundu  = True
                        self._yuz_x_gecmis.clear()
                        self._partial_lock  = None  # kullanıldı, temizle

                # ── KİLİTLİ / DOĞRULANMIŞ ────────────────────────────────
                elif self.kilit_durumu in ("YUZ_KILITLENDI", "DOGRULANDI"):
                    # [v10.4] Yüz zaten kilitliyken küçük imgsz yeterli.
                    # bytetrack_imgsz_kilitli=320 → takip kalitesi korunur,
                    # model.track() süresi belirgin düşer (~3× hızlanma).
                    bt_imgsz = c.bytetrack_imgsz_kilitli
                    track_konum, re_assign = self.bytetrack.guncelle(
                        kare_is, bt_imgsz)

                    kayip_esik = (c.kayip_esik_dogrulandi
                                  if self.kilit_durumu == "DOGRULANDI"
                                  else c.kayip_esik_kilit)

                    if track_konum:
                        bulunan_konum       = track_konum
                        self.hedef_bulundu  = True
                        self.hedef_kayip_sayac = 0

                        ust, sag, alt, sol = track_konum
                        yuz_px = sag - sol
                        mx     = (sol + sag) // 2
                        my     = (ust + alt) // 2
                        self._guncel_yuz_px = yuz_px
                        self._yuz_x_gecmis.append(mx)

                        self.kalman.guncelle(mx, my, sag - sol, alt - ust)
                        alan_oran = ((sag-sol)*(alt-ust)) / (
                            kare.shape[0] * kare.shape[1])
                        self._guncel_imgsz = self._imgsz_sec(alan_oran)

                        # Embedding topla
                        if self._frame_id % c.emb_al_periyot == 0:
                            if self.asenkron:
                                self.async_emb.istek_gonder(
                                    kare_is, track_konum, self._frame_id)
                            else:
                                emb, kal = self._sync_worker._isle(
                                    kare_is, track_konum)
                                if emb is not None:
                                    self.emb_pencere.ekle(
                                        emb, kal, self._frame_id)

                        if self.asenkron:
                            emb, kal = self.async_emb.sonuc_al()
                            if emb is not None:
                                self.emb_pencere.ekle(emb, kal, self._frame_id)

                        # ── YUZ_KILITLENDI: tanıma dene ──────────────────
                        if self.kilit_durumu == "YUZ_KILITLENDI":
                            # [v10.1] Küçük yüzlerde daha az onay gerek
                            if yuz_px < 15:
                                min_onay, deneme_max = 1, 10
                            elif yuz_px < 20:
                                min_onay, deneme_max = 1, 8
                            elif yuz_px < 35:
                                min_onay, deneme_max = 2, 6
                            else:
                                min_onay    = c.kilitlenme_min_onay
                                deneme_max  = c.dogrulama_deneme_max

                            if self.emb_pencere.hazir_mi(yuz_px):
                                ort_emb = self.emb_pencere.en_iyi_embedding()
                                isim, guven = self.db.tanimla(
                                    ort_emb,
                                    self.son_mesafe_metre,
                                    yuz_piksel_genislik=yuz_px,
                                    debug_skor=c.debug_skor,
                                    gorunen_yuz_sayisi=self._gorunen_yuz_sayisi)
                                self.dogrulama_sayac += 1

                                if isim:
                                    # [v10.3] KÜMÜLATİF OYLAMA
                                    # Aynı aday → sayac artar, pencere temizlenmez
                                    # Farklı aday → aday değişir, sayac 1'e döner
                                    if isim == self._aday_isim:
                                        self._onay_sayac += 1
                                    else:
                                        self._aday_isim  = isim
                                        self._onay_sayac = 1

                                    print(f"  [{self._onay_sayac}/{min_onay}] "
                                          f"Doğrulanıyor: {isim} "
                                          f"({guven:.1f}%) px={yuz_px} "
                                          f"yüz={self._gorunen_yuz_sayisi}")

                                    if self._onay_sayac >= min_onay:
                                        self.aktif_isim  = isim
                                        self.hedef_guven = guven
                                        self.kilit_durumu = "DOGRULANDI"
                                        self.reverifikasyon_sayac        = 0
                                        self.reverifikasyon_basarisizlik = 0
                                        self._onay_sayac    = 0
                                        self._aday_isim     = None  # [v10.3]
                                        self._partial_lock  = None
                                        self._lazer_gecikme_sayac = 0  # [v10.2]
                                        self.mod = f"✓ {isim}"
                                        print(
                                            f"✅ KİLİT: {isim} ({guven:.1f}%) "
                                            f"px:{yuz_px} "
                                            f"yüz:{self._gorunen_yuz_sayisi}")
                                    # [v10.3] emb_pencere.temizle() KALDIRILDI
                                    # Pencere birikerek devam eder — daha stabil
                                else:
                                    # [v10.3] Başarısız tanımada sayacı azalt,
                                    # sıfırlama. Biriken iyi embeddingler korunur.
                                    self._onay_sayac = max(0, self._onay_sayac - 1)

                                    if self.dogrulama_sayac >= deneme_max:
                                        # Tanınamadı → PartialLock oluştur
                                        en_skor, en_isim = \
                                            self._partial_lock_en_iyi_skor_al()
                                        self._partial_lock_olustur(
                                            list(self.emb_pencere._pencere),
                                            en_skor, en_isim,
                                            mx, my, yuz_px)
                                        print(f"  ❌ {deneme_max} denemede "
                                              f"tanınamadı → ARAMA")
                                        self._tam_sifirla()
                                    else:
                                        # [v10.3] emb_pencere.temizle() KALDIRILDI
                                        self.mod = (f"DOĞRULANAMADI "
                                                    f"({self.dogrulama_sayac}/"
                                                    f"{deneme_max})")

                        # ── DOGRULANDI: reverif ───────────────────────────
                        else:
                            self.mod = f"✓ {self.aktif_isim or '?'}"

                            if re_assign and not self._reverif_aktif:
                                self.reverifikasyon_sayac = max(
                                    self.reverifikasyon_sayac,
                                    c.reverifikasyon_periyot // 2)
                                print("  ⚠️  ByteTrack re-assign")

                            self.reverifikasyon_sayac += 1
                            if (self.reverifikasyon_sayac >= c.reverifikasyon_periyot
                                    and not self._reverif_aktif):
                                self.reverifikasyon_sayac = 0
                                self._reverif_aktif = True
                                self.reverif_isci.baslat(
                                    kare_is, track_konum,
                                    self.aktif_isim, self.son_mesafe_metre,
                                    yuz_px, self._frame_id)

                            if self._reverif_aktif:
                                isim, guven, tamamlandi = \
                                    self.reverif_isci.kontrol_et(self._frame_id)
                                if tamamlandi:
                                    self._reverif_aktif = False
                                    if isim == self.aktif_isim:
                                        self.hedef_guven = guven
                                        self.reverifikasyon_basarisizlik = 0
                                        print(f"  ✅ ReVerif: {isim} "
                                              f"({guven:.1f}%)")
                                    else:
                                        self.reverifikasyon_basarisizlik += 1
                                        print(
                                            f"  ⚠️  ReVerif başarısız "
                                            f"{self.reverifikasyon_basarisizlik}"
                                            f"/{c.reverifikasyon_basarisizlik_esik}"
                                            f": {self.aktif_isim} → "
                                            f"{isim or 'YABANCI'}")
                                        self.emb_pencere.temizle()
                                        if self.async_emb:
                                            self.async_emb.temizle()
                                        if (self.reverifikasyon_basarisizlik
                                                >= c.reverifikasyon_basarisizlik_esik):
                                            print("  🔴 → ARAMA")
                                            self._tam_sifirla()
                                else:
                                    self._reverif_aktif = False

                    else:
                        # Yüz kaybedildi
                        self.hedef_kayip_sayac += 1

                        # [v10.1] Kayıp olunca PartialLock oluştur
                        if (self.kilit_durumu == "YUZ_KILITLENDI"
                                and self.hedef_kayip_sayac == 1
                                and self.emb_pencere.boyut() > 0):
                            en_skor, en_isim = self._partial_lock_en_iyi_skor_al()
                            if self.aktif_konum:
                                ust_s, sag_s, alt_s, sol_s = self.aktif_konum
                                self._partial_lock_olustur(
                                    list(self.emb_pencere._pencere),
                                    en_skor, en_isim,
                                    (sol_s + sag_s) // 2,
                                    (ust_s + alt_s) // 2,
                                    sag_s - sol_s)

                        tahmin = self.kalman.tahmin_et()
                        if tahmin and self.hedef_kayip_sayac < kayip_esik // 2:
                            bulunan_konum       = tahmin
                            self.hedef_bulundu  = True
                            self.hedef_guven    = max(20.0,
                                                      self.hedef_guven * 0.88)
                            self.kalman_aktif   = True
                            self.mod            = "KALMAN"
                        elif self.hedef_kayip_sayac > kayip_esik:
                            self._tam_sifirla()

                # ── Servo & lazer ─────────────────────────────────────────
                if bulunan_konum:
                    self.aktif_konum = bulunan_konum
                    konum_data = self._konum_analiz(
                        bulunan_konum, kare.shape[:2])
                    son_komut = self._servo_hesapla(konum_data)
                    self.konum_gecmisi.append(konum_data)

                    # [v10.1] Tarama yön belleğini güncelle
                    if son_komut:
                        self._tarama_yon.komut_ekle(son_komut['yon'])

                    if self.arduino_bagli:
                        self.servo_gonder(son_komut['yon'],
                                          son_komut['aci'],
                                          son_komut['mesafe'])
                        # [v10.2] Lazer ateşleme gecikmesi
                        if self.kilit_durumu == "DOGRULANDI":
                            self._lazer_gecikme_sayac += 1
                            if (self._lazer_gecikme_sayac >=
                                    c.lazer_gecikme_frame
                                    and not self.lazer_aktif):
                                self.lazer_kontrol(True)
                        elif self.lazer_aktif:
                            self.lazer_kontrol(False)
                            self._lazer_gecikme_sayac = 0

                # ── Debug ─────────────────────────────────────────────────
                if c.debug_mesafe and bulunan_konum:
                    ust, sag, alt, sol = bulunan_konum
                    px_g = sag - sol
                    print(f"  [DBG] f={self._frame_id} px={px_g} "
                          f"m={self.son_mesafe_metre or '?'} "
                          f"imgsz={self._guncel_imgsz} "
                          f"emb={self.emb_pencere.boyut()} "
                          f"PL={'var' if self._partial_lock else 'yok'} "
                          f"yüz={self._gorunen_yuz_sayisi} "
                          f"lazer_bkl={self._lazer_gecikme_sayac}")

                # ── FPS ───────────────────────────────────────────────────
                fps_s += 1
                if time.time() - fps_b >= 1.0:
                    self.fps_gercek = fps_s
                    fps_s = 0
                    fps_b = time.time()
                self.isleme_suresi = (time.time() - t0) * 1000

                # ── Ekran ─────────────────────────────────────────────────
                if gosterim:
                    ekran = self._ciz(kare, son_komut, konum_data)
                    if yazici:
                        yazici.write(ekran)
                    cv2.namedWindow("Yaris v10.2", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Yaris v10.2", 1400, 900)
                    cv2.imshow("Yaris v10.2", ekran)

                if gosterim and cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            if self.async_emb:
                self.async_emb.durdur()
            if self._sync_worker:
                self._sync_worker.durdur()
            self.reverif_isci.durdur()
            if kam:
                kam.release()
            if yazici:
                yazici.release()
            if gosterim:
                cv2.destroyAllWindows()
            if self.arduino_bagli:
                self.lazer_kontrol(False)
                self.arduino.write(b"DUR\n")
            print("\n🏁 Sistem durduruldu.")

    # ─────────────────────────────────────────────────────────────────────
    # EKRAN ÇİZİMİ
    # ─────────────────────────────────────────────────────────────────────
    def _ciz(self, kare, son_komut, konum_data):
        panel = np.zeros((140, kare.shape[1], 3), dtype=np.uint8)

        if self.kalman_aktif and self.hedef_bulundu:
            durum, d_renk = "KALMAN TAHMİN", (0, 200, 255)
        elif self.hedef_bulundu:
            durum  = f"KİLİTLENDİ: {self.aktif_isim or '?'}"
            d_renk = (0, 255, 0)
        else:
            durum, d_renk = "ARANIYOR...", (0, 0, 255)

        cv2.putText(panel, durum, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, d_renk, 2)
        cv2.putText(
            panel,
            (f"Guven: {self.hedef_guven:.1f}%"
             if self.hedef_bulundu
             else f"Kayip: {self.hedef_kayip_sayac}"),
            (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 0) if self.hedef_bulundu else (128, 128, 255), 1)

        # PartialLock göstergesi
        if self._partial_lock:
            cv2.putText(
                panel,
                f"PL:{self._partial_lock.aday_isim}"
                f"[{self._partial_lock.gecerlilik_frame - self._frame_id}f]",
                (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

        if self.son_mesafe_metre:
            renk_m = ((0, 255, 100) if self.son_mesafe_metre < 5 else
                      (0, 200, 255) if self.son_mesafe_metre < 8 else
                      (0, 100, 255))
            cv2.putText(panel, f"{self.son_mesafe_metre:.2f}m",
                        (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk_m, 2)

        sr_renk = (0, 200, 100) if self.sr.mod == "ONNX_SR" else (100, 180, 100)
        cv2.putText(panel, f"SR:{self.sr.mod}",
                    (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.4, sr_renk, 1)

        cv2.putText(panel, f"FPS:{self.fps_gercek}",
                    (340, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(panel, f"{self.isleme_suresi:.0f}ms",
                    (340, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)
        cv2.putText(panel, f"imgsz:{self._guncel_imgsz}",
                    (340, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 100), 1)
        cv2.putText(
            panel,
            f"px:{self._guncel_yuz_px} "
            f"emb:{self.emb_pencere.boyut()}/{self.cfg.emb_pencere_max} "
            f"yüz:{self._gorunen_yuz_sayisi}",
            (340, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        cv2.putText(
            panel,
            f"tarama:{self._tarama_yon.dominant_yon()} "
            f"lazer_bkl:{self._lazer_gecikme_sayac}/{self.cfg.lazer_gecikme_frame}",
            (340, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 200), 1)

        cv2.putText(panel,
                    "LAZER:ON" if self.lazer_aktif else "LAZER:OFF",
                    (550, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if self.lazer_aktif else (100, 100, 100), 2)
        cv2.putText(panel,
                    "ARD:OK" if self.arduino_bagli else "ARD:YOK",
                    (550, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)

        mod_str = ""
        if self.cfg.yarisma_modu: mod_str += "YARISMA "
        if self.cfg.manken_modu:  mod_str += "MANKEN"
        if mod_str:
            cv2.putText(panel, mod_str.strip(),
                        (550, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (0, 255, 200), 1)

        profil_str = self.cfg.kamera_profil.upper()
        cv2.putText(panel, f"CAM:{profil_str}",
                    (550, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200, 200, 0), 1)

        kilit_renk = ((0, 255, 0) if self.kilit_durumu == "DOGRULANDI" else
                      (0, 165, 255) if self.kilit_durumu == "YUZ_KILITLENDI"
                      else (0, 0, 255))
        cv2.putText(panel, self.kilit_durumu,
                    (550, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    kilit_renk, 1)

        if self.kilit_durumu == "YUZ_KILITLENDI" and self._onay_sayac > 0:
            pct   = self._onay_sayac / max(self.cfg.kilitlenme_min_onay, 1)
            bar_w = int(pct * 90)
            cv2.rectangle(panel, (550, 120), (640, 132), (40, 40, 40), -1)
            cv2.rectangle(panel, (550, 120), (550 + bar_w, 132),
                          (0, 165, 255), -1)

        if self.kilit_durumu == "DOGRULANDI":
            pct   = min(self.reverifikasyon_sayac /
                        max(self.cfg.reverifikasyon_periyot, 1), 1.0)
            bar_w = int(pct * 90)
            cv2.rectangle(panel, (550, 120), (640, 132), (40, 40, 40), -1)
            rv_renk = ((0, 100, 255) if self._reverif_aktif
                       else (0, 200, 100))
            cv2.rectangle(panel, (550, 120), (550 + bar_w, 132),
                          rv_renk, -1)

        if son_komut:
            cv2.putText(
                panel,
                f"{son_komut['yon']} {son_komut['aci']}d | "
                f"{son_komut['mesafe']}",
                (750, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 165, 0), 2)

        if self.aktif_konum:
            ust, sag, alt, sol = self.aktif_konum
            if self.kalman_aktif:
                renk = (0, 200, 255)
                for i in range(0, sag - sol, 15):
                    cv2.line(kare, (sol+i, ust),
                             (min(sol+i+8, sag), ust), renk, 2)
                    cv2.line(kare, (sol+i, alt),
                             (min(sol+i+8, sag), alt), renk, 2)
                for i in range(0, alt - ust, 15):
                    cv2.line(kare, (sol, ust+i),
                             (sol, min(ust+i+8, alt)), renk, 2)
                    cv2.line(kare, (sag, ust+i),
                             (sag, min(ust+i+8, alt)), renk, 2)
            else:
                renk = ((0, 255, 0) if self.kilit_durumu == "DOGRULANDI"
                        else (0, 165, 255))
                cv2.rectangle(kare, (sol, ust), (sag, alt), renk, 3)

            if konum_data:
                mx, my = konum_data['merkez_x'], konum_data['merkez_y']
                if self.lazer_aktif:
                    cv2.circle(kare, (mx, my), 9, (0, 0, 255), -1)
                    cv2.circle(kare, (mx, my), 16, (0, 0, 255), 2)
                else:
                    cv2.circle(kare, (mx, my), 5, (0, 255, 255), -1)
                if son_komut and son_komut['yon'] in ('SOL', 'SAG'):
                    hx = mx - 70 if son_komut['yon'] == 'SOL' else mx + 70
                    cv2.arrowedLine(kare, (mx, my), (hx, my),
                                    (255, 50, 50), 4)

            cv2.rectangle(kare, (sol, alt - 35), (sag, alt),
                          renk if not self.kalman_aktif else (0, 200, 255),
                          cv2.FILLED)
            isim_str = (
                f"{self.aktif_isim or '?'} "
                f"{'SR' if self.sr.gerekli_mi(sag - sol) else ''}"
            ).strip()
            cv2.putText(kare, isim_str, (sol + 6, alt - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 1)

        return np.vstack([panel, kare])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description='Yarış Sistemi v10.3')
    p.add_argument('--arduino',           type=str)
    p.add_argument('--hedef-ekle',        type=str)
    p.add_argument('--isim',              type=str)
    p.add_argument('--hedef-sil',         type=str)
    p.add_argument('--hedef-listesi',     action='store_true')
    p.add_argument('--no-augment',        action='store_true')
    p.add_argument('--model',             type=str, default='yolov8s-face.pt')
    p.add_argument('--tensorrt',          action='store_true',
                   help='TensorRT engine kullan')
    p.add_argument('--orin',              action='store_true',
                   help='Jetson Orin Nano optimizasyonları')
    p.add_argument('--kayit',             action='store_true')
    p.add_argument('--gosterim-yok',      action='store_true')
    p.add_argument('--cpu',               action='store_true')
    p.add_argument('--no-async',          action='store_true')
    p.add_argument('--no-sr',             action='store_true')
    p.add_argument('--no-multiscale',     action='store_true')
    p.add_argument('--no-kalite',         action='store_true')
    p.add_argument('--no-mjpeg',          action='store_true')
    p.add_argument('--no-roi',            action='store_true')
    p.add_argument('--no-partial-lock',   action='store_true')
    p.add_argument('--kamera',            type=int, default=2)
    p.add_argument('--kamera-profil',     type=str, default='c925',
                   choices=['brio', 'c925'])
    p.add_argument('--emb-periyot',       type=int, default=2)
    p.add_argument('--reverif-periyot',   type=int, default=50)
    p.add_argument('--reverif-timeout',   type=int, default=120)
    p.add_argument('--sr-model',          type=str, default='realesrgan_x4.onnx')
    p.add_argument('--yarisma-modu',      action='store_true')
    p.add_argument('--manken-modu',       action='store_true',
                   help='Hareketsiz hedef (baskılı yüz) için optimize parametreler')
    p.add_argument('--zoom',              type=float, default=1.0)
    p.add_argument('--lazer-gecikme',     type=int, default=15)
    p.add_argument('--partial-lock-sure', type=int, default=90)
    p.add_argument('--partial-lock-bolge',type=int, default=200)
    p.add_argument('--on-isle-interval',   type=int, default=2,
                   help='filter2D+CLAHE her kaç frame — 1=her frame, 2=önerilen (Orin)')
    p.add_argument('--arama-yolo-interval', type=int, default=2,
                   help='ARAMA modunda multiscale YOLO sıklığı — 1=her frame, 2=önerilen')
    p.add_argument('--bt-imgsz-kilitli',    type=int, default=320,
                   help='ByteTrack kilitli modda imgsz — 320 önerilen (Orin)')
    p.add_argument('--debug-mesafe',        action='store_true')
    p.add_argument('--debug-skor',        action='store_true')
    p.add_argument('--debug-partial',     action='store_true')
    args = p.parse_args()

    cfg = YarisConfig(
        kamera_indeks                = args.kamera,
        kamera_profil                = args.kamera_profil,
        yolo_model                   = args.model,
        emb_al_periyot               = args.emb_periyot,
        reverifikasyon_periyot       = args.reverif_periyot,
        reverifikasyon_timeout_frame = args.reverif_timeout,
        sr_aktif                     = not args.no_sr,
        sr_model_yolu                = args.sr_model,
        multiscale_aktif             = not args.no_multiscale,
        kalite_aktif                 = not args.no_kalite,
        mjpeg_aktif                  = not args.no_mjpeg,
        roi_aktif                    = not args.no_roi,
        partial_lock_aktif           = not args.no_partial_lock,
        partial_lock_max_frame       = args.partial_lock_sure,
        partial_lock_bolge_esik      = args.partial_lock_bolge,
        yarisma_modu                 = args.yarisma_modu,
        manken_modu                  = args.manken_modu,
        debug_mesafe                 = args.debug_mesafe,
        debug_skor                   = args.debug_skor,
        debug_partial                = args.debug_partial,
        dijital_zoom                 = args.zoom,
        tensorrt_aktif               = args.tensorrt,
        jetson_orin                  = args.orin,
        lazer_gecikme_frame          = args.lazer_gecikme,
        on_isle_interval             = args.on_isle_interval,
        arama_yolo_interval          = args.arama_yolo_interval,
        bytetrack_imgsz_kilitli      = args.bt_imgsz_kilitli,
    )

    if args.hedef_listesi:
        db = ArcFaceVeritabani(cfg)
        if db.bos:
            print("Veritabanı boş.")
        else:
            print("Kayıtlı hedefler:")
            for isim, emb in db.hedefler.items():
                sayi = len(emb) if isinstance(emb, list) else 1
                aug  = " (augmentasyonlu)" if sayi > 1 else ""
                print(f"  • {isim}  ({sayi} embedding){aug}")
        return

    if args.hedef_ekle:
        if not args.isim:
            print("❌ --isim gerekli!")
            return
        ArcFaceVeritabani(cfg, gpu=not args.cpu).hedef_ekle(
            args.hedef_ekle, args.isim,
            augmentasyon=not args.no_augment)
        return

    if args.hedef_sil:
        ArcFaceVeritabani(cfg).hedef_sil(args.hedef_sil)
        return

    sistem = YarisV10(
        arduino_port = args.arduino,
        tensorrt     = args.tensorrt,
        gpu          = not args.cpu,
        asenkron     = not args.no_async,
        cfg          = cfg,
    )
    sistem.baslat(gosterim=not args.gosterim_yok, kayit=args.kayit)


if __name__ == "__main__":
    main()
