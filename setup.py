# -*- coding: utf-8 -*-
import sys

from setuptools import find_packages
from setuptools import setup

if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise Exception("Must be using Python 3")

    setup(
        name='HW4Graph',
        python_requires='>=3.8.0',
        version=open('version.txt', 'r').read(),
        description='SCC Implementation for AdjLists HW4 Tristan',
        url='https://github.com/Dariusrussellkish/Algorithms-HW4.git',
        author='Darius Russell Kish',
        packages=find_packages(),
        install_requires=['networkx'],
        setup_requires=["pytest-runner"],
        tests_require=["pytest"],
        include_package_data=True,
        zip_safe=False,
        scripts=[]
    )
