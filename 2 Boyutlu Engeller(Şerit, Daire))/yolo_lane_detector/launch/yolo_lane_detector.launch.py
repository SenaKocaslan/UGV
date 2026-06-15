import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    pkg_yolo_lane = get_package_share_directory('yolo_lane_detector')
    zed_wrapper_launch_path = os.path.join(get_package_share_directory('zed_wrapper'), 'launch', 'zed_camera.launch.py')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    
    nav2_params_file = os.path.join(pkg_yolo_lane, 'config', 'nav2_params.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')


    #1. ZED MINI SÜRÜCÜSÜ (Gerçek kamera verisi ve Odometri için) 
    zed_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(zed_wrapper_launch_path),
        launch_arguments={
            'camera_model': 'zedm',
            'publish_tf': 'true',
            'publish_map_tf': 'true', # bu ayar zedin kendi iç haritalandırmasının kullnamasını sağlıyor
            'pos_tracking_enabled': 'true', # KONUM TAKİBİNİ LAUNCH İÇİNDEN KESİN OLARAK AÇIYORUZ
            'odometry_frame': 'odom',
            'map_frame': 'map',
            'base_frame': 'zed_camera_link'
        }.items()
    )


    #2. STATİK TF YAYINCI (Robotun gövdesi ile kamera arasındaki fark)
    # Robotun olmadığını varsayıyoruz, kamerayı base_link'in 20cm üzerinde kabul ediyoruz.
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0.2', '0', '0', '0', 'zed_camera_link', 'zed_left_camera_frame']
    )
    
    # 3. NAV2 (Navigasyon ve Maliyet Haritası)
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'false',
            'params_file': nav2_params_file
        }.items()
    )

    #4. YOLO LANE DETECTOR NODE (özel yazılan düğüm)
    lane_detector_node = Node(
        package='yolo_lane_detector',
        executable='lane_detector',
        name='lane_detector_node',
        output='screen',
        parameters=[{'use_sim_time': False}]
    )
    

    return LaunchDescription([
        # Önce kamera başlar
        zed_camera,
        
        # Kamera başladıktan sonra TF'i yayınla
        static_tf,

        # 5 saniye sonra YOLO düğümünü başlat (Kamera feed'inin oturması için)
        TimerAction(period=5.0, actions=[lane_detector_node]),

        # 10 saniye sonra Nav2'yi başlat
        TimerAction(period=10.0, actions=[nav2])
    ])
