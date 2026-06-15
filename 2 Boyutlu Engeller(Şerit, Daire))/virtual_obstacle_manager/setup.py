import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'virtual_obstacle_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch dosyaların varsa burayı aktif edebilirsin
        # (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='irem',
    maintainer_email='irem@todo.todo',
    description='Sanal engel yöneticisi paketi',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'master_driver = virtual_obstacle_manager.master_driver:main',
        ],
    },
)
