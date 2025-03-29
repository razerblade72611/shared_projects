import os
from glob import glob
from setuptools import setup

package_name = 'plen_ros_helpers'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    package_dir={'': 'src'},  # assuming Python modules are under src/plen_ros_helpers
    data_files=[
        # Install package.xml
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install any launch or config files if present
        # e.g., ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='...',
    maintainer_email='...',
    description='Helper nodes and scripts for PLEN ROS2 control',
    license='MIT',
    entry_points={
        'console_scripts': [
            # Register executable scripts provided by this package
            'gazebo_tools_test = plen_ros_helpers.gazebo_tools_test:main',
            'plen_td3 = plen_ros_helpers.plen_td3:main'
        ],
    },
)

