"""
Setup script for datasketch package
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
try:
    with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
        long_description_content_type = 'text/markdown'
except FileNotFoundError:
    with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
        long_description = f.read()
        long_description_content_type = 'text/x-rst'

# Get the code version
version = {}
with open(path.join(here, "hnsw/version.py")) as fp:
    exec(fp.read(), version)
__version__ = version['__version__']
# now we have a `__version__` variable

setup(
    name='hnsw-enhanced',
    version=__version__,
    description='HNSW algorithm implementation with comprehensive Chinese documentation',
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    url='https://github.com/HankyZhang/datasketch-enhanced',
    project_urls={
        'Source': 'https://github.com/HankyZhang/datasketch-enhanced',
        'Original': 'https://github.com/ekzhu/datasketch',
    },
    author='HankyZhang (Enhanced Version)',
    author_email='your.email@example.com',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Database',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='hnsw similarity-search machine-learning chinese-docs',
    packages=find_packages(include=['hnsw*', 'hybrid_hnsw*']),
    install_requires=[
        'numpy>=1.11',
        'scikit-learn>=0.24',  # Added for MiniBatchKMeans in method3
    ],
    extras_require={
        'benchmark': [
            'matplotlib>=3.1.2',
            'scikit-learn>=0.21.3',
            'pandas>=0.25.3',
        ],
        'test': [
            'pytest',
        ],
    },
)
