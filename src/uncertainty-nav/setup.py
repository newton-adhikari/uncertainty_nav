from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'uncertainty_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'urdf'),   glob('urdf/*.xacro')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'uncertainty_agent = uncertainty_nav.uncertainty_agent_node:main',
            'particle_filter   = uncertainty_nav.particle_filter_node:main',
            'rviz_uncertainty   = uncertainty_nav.rviz_uncertainty_node:main',
        ],
    },
)
