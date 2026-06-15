import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs_py.point_cloud2 import create_cloud
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import struct
import cv2  # Görselleştirme için OpenCV eklendi

class HybridLaneDetectorNode(Node):
    def __init__(self):
        super().__init__('hybrid_lane_detector_node')
        
        # YOLOv8 modeli
        self.model = YOLO("/home/irem/Desktop/best.pt")
        
        # CvBridge nesnesi (Hata düzeltildi: Parantez eklendi)
        self.bridge = CvBridge() 

        # PointCloud2 yayıncısı. Konu Adı: /lane_points_filtered
        self.pc2_pub = self.create_publisher(PointCloud2, '/lane_points_filtered', 10)
        
        # ZED RGB ve PointCloud konularına abonelikler
        self.rgb_sub = message_filters.Subscriber(self, Image, '/zed/zed_node/rgb/color/rect/image')
        self.pc2_sub = message_filters.Subscriber(self, PointCloud2, '/zed/zed_node/point_cloud/cloud_registered')

        # Zaman senkronizasyonu (Hata düzeltildi: Approximatetime -> ApproximateTime)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.pc2_sub], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info("Şerit tespit düğümü ve debug ekranı başlatıldı.")

    def sync_callback(self, rgb_msg, pc2_msg):
        # Görüntüyü OpenCV formatına dönüştür
        cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

        # Görselleştirme (Debug) için orijinal görüntünün bir kopyasını oluşturuyoruz
        debug_image = cv_image.copy()
        # Transparan maske boyaması yapabilmek için boş bir katman oluşturuyoruz
        mask_overlay = np.zeros_like(cv_image)

        # YOLOv8 çıkarımı
        results = self.model(cv_image, verbose=False)

        lane_points_3d = []

        x_offset = next(f.offset for f in pc2_msg.fields if f.name == 'x')
        y_offset = next(f.offset for f in pc2_msg.fields if f.name == 'y')
        z_offset = next(f.offset for f in pc2_msg.fields if f.name == 'z')

        for result in results:
            # Hata düzeltildi: results.masks yerine result.masks
            if result.masks is not None:
                # Hata düzeltildi: result.mask.xy yerine result.masks.xy
                for mask, box in zip(result.masks.xy, result.boxes):s
                    cls = int(box.cls[0])

                    # Şerit sınıfı (Sınıf ID'si 0 varsayılıyor)
                    if cls == 0:
                        # --- DEBUG: Maskeyi ve Tespit Kutusunu Ekrana Çizme ---
                        # Maske piksellerini OpenCV'nin çizebileceği formata (int32) getiriyoruz
                        pts = mask.astype(np.int32)
                        
                        # Şeridin içini yeşil renge boya (Katman üzerine)
                        cv2.fillPoly(mask_overlay, [pts], (0, 255, 0))
                        
                        # Şeridin etrafına ince yeşil bir kontur çiz
                        cv2.polylines(debug_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                        
                        # YOLO Bounding Box (Kutu) koordinatlarını al ve çiz
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2) # Mavi kutu
                        cv2.putText(debug_image, f"Serit: {box.conf[0]:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        # -----------------------------------------------------

                        for point in mask:
                            # Hata düzeltildi: int(point[0], int(point[1])) parantez hatası giderildi
                            u, v = int(point[0]), int(point[1])

                            # Sınır kontrolü
                            if u >= pc2_msg.width or v >= pc2_msg.height or u < 0 or v < 0:
                                continue

                            point_offset = (v * pc2_msg.row_step) + (u * pc2_msg.point_step)
                            try:
                                x = struct.unpack_from('f', pc2_msg.data, point_offset + x_offset)[0]
                                y = struct.unpack_from('f', pc2_msg.data, point_offset + y_offset)[0]
                                z = struct.unpack_from('f', pc2_msg.data, point_offset + z_offset)[0]

                                if np.isnan(x) or np.isnan(y) or np.isnan(z):
                                    continue
                                if np.isinf(x) or np.isinf(y) or np.isinf(z):
                                    continue

                                # Geçerli 3D noktayı listeye ekle
                                lane_points_3d.append([x, y, z])

                            except Exception:
                                continue

        # --- DEBUG: Transparan Maskeyi Orijinal Görüntüyle Birleştirme ---
        # %60 orijinal görüntü, %40 yeşil şerit maskesi olacak şekilde harmanlıyoruz
        debug_image = cv2.addWeighted(mask_overlay, 0.4, debug_image, 0.6, 0)
        
        # Gerçek zamanlı pencereyi göster
        cv2.imshow("YOLOv8 Şerit Tespit Debug Ekranı", debug_image)
        cv2.waitKey(1) # OpenCV pencerelerinin donmaması için 1ms bekleme şarttır
        # -----------------------------------------------------------------

        # Eğer geçerli şerit noktası varsa PointCloud2 mesajı üret ve yayınla
        if len(lane_points_3d) > 0:
            header = Header()
            header.stamp = rgb_msg.header.stamp
            header.frame_id = pc2_msg.header.frame_id

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]

            filtered_pc2_msg = create_cloud(header, fields, lane_points_3d)
            self.pc2_pub.publish(filtered_pc2_msg)

def main(args=None):
    # ROS2 iletişim altyapısını başlat
    rclpy.init(args=args)
    # Hata düzeltildi: Sınıf çağrısına parantez eklendi, nesne üretildi
    node = HybridLaneDetectorNode()

    try: 
        # Kamera verisi geldiği sürece düğümü açık tut
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    # Kapatırken OpenCV pencerelerini temizle
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()