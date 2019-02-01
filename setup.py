from setuptools import setup, find_packages

# Distribute: python setup.py sdist upload

setup(
    name='pywmi',
    version='0.3.21',
    description='Essential tools and interfaces for WMI',
    url='http://github.com/samuelkolb/pywmi',
    author='Samuel Kolb',
    author_email='samuel.kolb@me.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=['pysmt<0.8', 'numpy', 'future', 'typing', 'matplotlib', 'pillow', 'polytope', 'tabulate', 'problog',
                      'graphviz', 'sympy', 'scipy', 'autodora>=0.3.3', 'deprecated'],
    extras_require={
        'sdd': ["pysdd"]
    },
    setup_requires=['pytest-runner'],
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pywmi-install = pywmi.install:main",
            "pywmi-cli = pywmi.__main__:parse"
        ]
    }
)
