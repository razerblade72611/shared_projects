from setuptools import setup
import os
from glob import glob

package_name = 'plen_ros_helpers'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    package_dir={'': 'src'},
    data_files=[
        # Install the ament resource index file
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # Install package.xml
        ('share/' + package_name, ['package.xml']),
        # Install launch files (if any)
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Install config files (including the macro JSON file)
        (os.path.join('share', package_name, 'config'), glob('config/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Maurice Rahme',
    author_email='mrahme97@gmail.com',
    maintainer='Maurice Rahme',
    maintainer_email='mrahme97@gmail.com',
    description='Visualise and control a PLEN2 robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joint_test = plen_ros_helpers.joint_test:main',
            'joint_trajectory_test = plen_ros_helpers.joint_trajectory_test:main',
            'plen_td3 = plen_ros_helpers.plen_td3:main',
            'walk_eval = plen_ros_helpers.walk_eval:main',
            'macro_runner = plen_ros_helpers.macro_runner:main',  # ✅ Added macro_runner
        ],
    },
)

