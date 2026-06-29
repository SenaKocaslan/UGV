#!/usr/bin/env python3
"""
MASTER DRIVER v4.0 - ZED Mini + NAV2 ENTEGRE
Şerit takibi + engel kaçınma ROS2 düğümü

ZED Mini entegrasyonu:
  - Sol göz RGB  : /zed_mini/zed_node/left/image_rect_color  (bgra8)
  - Derinlik     : /zed_mini/zed_node/depth/depth_registered  (32FC1, metre)
  - Kamera bilgi : /zed_mini/zed_node/left/camera_info        (fx/fy/cx/cy)
  - message_filters.ApproximateTimeSynchronizer ile RGB+depth senkronize
  - _pixel_to_world: geometrik tahmin yerine ZED'in gerçek z değeri kullanılır
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import message_filters
import struct
import sys
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

try:
    from virtual_obstacle_manager.visual_processor import VisualProcessor
    print("[MasterDriver] visual_processor.py yuklendi")
except ImportError as e:
    print(f"HATA: visual_processor.py bulunamadi -> {e}")
    sys.exit(1)


class MasterDriverNode(Node):

    def __init__(self):
        super().__init__('master_driver_node')

        # ── QoS profilleri ────────────────────────────────────────────────
        # ZED wrapper RELIABLE veya BEST_EFFORT yayınlayabilir.
        # Güvenli seçim: BEST_EFFORT subscriber (her ikisiyle çalışır).
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE
        )

        # ── ZED Mini SUBSCRIBER'lari (message_filters ile sync) ───────────
        # Sol göz renkli görüntü (bgra8 — ZED wrapper varsayılanı)
        self.sub_rgb = message_filters.Subscriber(
            self,
            Image,
            '/zed/zed_node/rgb/color/rect/image',
            qos_profile=sensor_qos
        )

        # Derinlik haritası: 32FC1, her piksel = metre cinsinden mesafe
        # NaN  → geçersiz (çok yakın / çok uzak)
        # Inf  → sonsuz (arka plan)
        self.sub_depth = message_filters.Subscriber(
            self,
            Image,
            '/zed/zed_node/depth/depth_registered',
            qos_profile=sensor_qos
        )

        # Zaman damgaları arasında en fazla 50ms fark kabul et
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth],
            queue_size=30,
            slop=2.0
        )
        self.sync.registerCallback(self.listener_callback)
        self.get_logger().info("ZED Mini RGB + Depth subscriber hazir")

        # ── Kamera parametreleri (başlangıç değerleri) ────────────────────
        # ZED Mini @ 640x480 yaklaşık değerler.
        # camera_info callback'i ile otomatik güncellenir → en doğru yol.
        self.fx = 336.0
        self.fy = 336.0
        self.cx = 320.0
        self.cy = 240.0
        self._camera_info_received = False

        # Kamera bilgisi geldiğinde fx/fy/cx/cy'yi güncelle
        self.sub_camera_info = self.create_subscription(
            CameraInfo,
            '/zed/zed_node/rgb/color/rect/image/camera_info',
            self._camera_info_callback,
            10
        )

        # ── PUBLISHER'lar ─────────────────────────────────────────────────
        self.obstacle_pub = self.create_publisher(
            PointCloud2, '/camera/virtual_obstacles', 10)

        self.debug_pub = self.create_publisher(
            Image, '/camera/debug_image', 10)

        # ── Sürüş parametreleri ───────────────────────────────────────────
        self.max_linear_speed  = 0.3
        self.max_angular_speed = 1.0
        self.lane_kp           = 0.004
        self.obstacle_kp       = 0.8
        self.img_width         = 640.0

        # ── Güvenlik ──────────────────────────────────────────────────────
        self.safe_lane_margin = 30

        # ── Performans ────────────────────────────────────────────────────
        self.frame_count     = 0
        self.process_every_n = 2

        self.get_logger().info("MASTER DRIVER v4.0 + ZED Mini BASLADI")

    # ──────────────────────────────────────────────────────────────────────
    # KAMERA BİLGİSİ CALLBACK — fx/fy/cx/cy otomatik güncellenir
    # ──────────────────────────────────────────────────────────────────────
    def _camera_info_callback(self, msg: CameraInfo):
        if self._camera_info_received:
            return   # bir kez yeterli
        # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self._camera_info_received = True
        self.get_logger().info(
            f"Kamera intrinsics alindi: fx={self.fx:.1f} fy={self.fy:.1f} "
            f"cx={self.cx:.1f} cy={self.cy:.1f}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # ANA CALLBACK — RGB + Depth senkronize gelir
    # ──────────────────────────────────────────────────────────────────────
    def listener_callback(self, rgb_msg: Image, depth_msg: Image):

        self.get_logger().info("CALLBACK TETİKLENDİ")

        self.frame_count += 1
        if self.frame_count % self.process_every_n != 0:
            return

        # ── RGB dönüştür ─────────────────────────────────────────────────
        # ZED wrapper bgra8 yayınlar → bgr8'e çevir
        try:
            frame_bgra = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgra8')
            frame      = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
        except Exception:
            try:
                frame = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            except Exception as e:
                self.get_logger().error(f"RGB donusturme hatasi: {e}")
                return

        # ── Depth dönüştür ───────────────────────────────────────────────
        # 32FC1 → float32 numpy dizisi, değerler metre cinsinden
        try:
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        except Exception as e:
            self.get_logger().warn(f"Depth donusturme hatasi: {e} — fallback")
            depth_frame = None

        # ── Görüntü işleme ───────────────────────────────────────────────
        try:
            obstacles, lane_info = self.vision.process_frame(frame, depth_frame)

            debug_img = lane_info.get('debug_frame')

            if debug_img is not None:
                cv2.imshow("UGVC-10 | Robotun Gozu", debug_img)
                cv2.imshow("Ham Goruntu", frame)
                cv2.waitKey(1)

            self._publish_obstacles(obstacles)
            self._publish_debug(debug_img)

        except Exception as e:
            self.get_logger().error(f"Islem hatasi: {e}")
            import traceback
            traceback.print_exc()

    # ──────────────────────────────────────────────────────────────────────
    # ENGEL KOORDİNATI: Piksel + Gerçek Derinlik → Dünya (metre)
    #
    #   ZED Mini ile:
    #     z  = depth_frame[py, px]          ← gerçek mesafe (metre)
    #     x  = (px - cx) * z / fx           ← sağ-sol  (metre)
    #     y  = (py - cy) * z / fy           ← yukarı-aşağı (metre)
    #
    #   depth_value None ise eski geometrik yönteme (cam_height) geri düşer.
    # ──────────────────────────────────────────────────────────────────────
    def _pixel_to_world(self, px, py, depth_value=None):
        if depth_value is not None and np.isfinite(depth_value) and depth_value > 0:
            # ── ZED Mini: pin-hole projeksiyon, gerçek z ─────────────────
            z = float(depth_value)
            x = (px - self.cx) * z / self.fx
            y = (py - self.cy) * z / self.fy
        else:
            # ── Fallback: kamera yüksekliği varsayımı ────────────────────
            cam_height = 0.60
            x = (px - self.cx) * cam_height / self.fx
            z = (py - self.cy) * cam_height / self.fy
            y = 0.0
        return x, y, z

    # ──────────────────────────────────────────────────────────────────────
    def _publish_obstacles(self, obstacles):
        msg = self._build_cloud(obstacles)
        self.obstacle_pub.publish(msg)

    # ──────────────────────────────────────────────────────────────────────
    def _build_cloud(self, points):
        msg = PointCloud2()
        msg.header.frame_id = 'base_link'
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.height          = 1
        msg.width           = max(len(points), 1)
        msg.fields          = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step   = 12
        msg.is_dense     = True

        if points:
            buf = b''
            for p in points:
                # p = (x_px, side_norm, z_metre)
                # z_metre → visual_processor'dan gelen ZED derinliği
                px      = float(p[0])
                z_depth = float(p[2])   # ZED'den okunan gerçek derinlik
                py_est  = self.cy       # engelin y pikseli bilinmiyorsa merkez
                x, y, z = self._pixel_to_world(px, py_est, z_depth)
                buf += struct.pack('fff', x, y, z)
        else:
            buf = struct.pack('fff', 0.0, 0.0, 0.0)

        msg.row_step = len(buf)
        msg.data     = buf
        return msg

    # ──────────────────────────────────────────────────────────────────────
    def _publish_debug(self, debug_frame):
        if debug_frame is None:
            return
        try:
            dm = self.bridge.cv2_to_imgmsg(debug_frame, 'bgr8')
            dm.header.stamp    = self.get_clock().now().to_msg()
            dm.header.frame_id = 'camera'
            self.debug_pub.publish(dm)
        except Exception as e:
            self.get_logger().warn(
                f"Debug yayini hatasi: {e}",
                throttle_duration_sec=5.0)


# ──────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MasterDriverNode()
        node.bridge = CvBridge()
        node.vision = VisualProcessor()

        print("Calisiyor... (Ctrl+C ile dur)")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Ctrl+C -> cikis")
    except Exception as e:
        print(f"HATA: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node:
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print("Program bitti")


if __name__ == '__main__':
    main()