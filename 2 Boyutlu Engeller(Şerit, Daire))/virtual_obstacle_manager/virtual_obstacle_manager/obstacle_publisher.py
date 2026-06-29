import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import struct
import math

class ObstaclePublisher(Node):
    def __init__(self):
        super().__init__('obstacle_publisher')

        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)

        self.publisher_ = self.create_publisher(PointCloud2, '/virtual_obstacles', 10)
        self.timer = self.create_timer(1.0, self.publish_points)

        # --- Çizgi parametreleri ---
        self.x_start   = -5.0   # Başlangıç x
        self.x_end     = 5.0  # Bitiş x
        self.x_step    = 0.05  #atif = sola doğru
        self.y_near    =  -3.0   # Her çizginin yakın ucu
        self.y_far     =  3.0   # Her çizginin uzak ucu
        self.z_val     =  0.0   # Sabit yükseklik

        self.get_logger().info('Engel yayıncı başlatıldı: /virtual_obstacles')

    def _generate_points(self):
        """
        Her x değeri için [x, y_near, z] ve [x, y_far, z] ikilisini üretir.
        Sonuç: karşılıklı, kesintisiz dikey çizgi segmentleri.
        """
        # while + float yerine index bazlı hesaplama → floating point birikimi yok
        n_steps = int(round((self.x_end - self.x_start) / self.x_step))
        points = []
        for i in range(n_steps + 1):
            x = round(self.x_start + i * self.x_step, 6)
            points.append([x, self.y_near, self.z_val])
            points.append([x, self.y_far,  self.z_val])
        return points

    def publish_points(self):
        now = self.get_clock().now()
        if now.nanoseconds == 0:
            self.get_logger().warn('Gazebo saati bekleniyor (sim_time = 0)...')
            return

        points = self._generate_points()

        msg = PointCloud2()
        msg.header.stamp    = now.to_msg()
        msg.header.frame_id = 'map'
        msg.height     = 1
        msg.width      = len(points)
        msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step   = 12
        msg.row_step     = 12 * len(points)
        msg.is_dense     = True

        buffer = bytearray()
        for p in points:
            buffer += struct.pack('fff', p[0], p[1], p[2])
        msg.data = bytes(buffer)

        self.publisher_.publish(msg)
        self.get_logger().info(f'{len(points)} nokta yayınlandı ({len(points)//2} çizgi segmenti)')

def main():
    rclpy.init()
    node = ObstaclePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
