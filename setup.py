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
with open(path.join(here, "datasketch/version.py")) as fp:
    exec(fp.read(), version)
__version__ = version['__version__']
# now we have a `__version__` variable

setup(
    name='datasketch-enhanced',
    version=__version__,
    description='Enhanced datasketch with comprehensive Chinese documentation and comments',
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
    keywords='database datamining',
    packages=find_packages(include=['datasketch*']),
    install_requires=[
        'numpy>=1.11',
        'scipy>=1.0.0',
    ],
    extras_require={
        'cassandra': [
            'cassandra-driver>=3.20',
        ],
        'redis': [
            'redis>=2.10.0',
        ],
        'benchmark': [
            'pyhash>=0.9.3',
            'matplotlib>=3.1.2',
            'scikit-learn>=0.21.3',
            'scipy>=1.3.3',
            'pandas>=0.25.3',
            'SetSimilaritySearch>=0.1.7',
            'pyfarmhash>=0.2.2',
            'nltk>=3.4.5',
        ],
        'test': [
            'cassandra-driver>=3.20',
            'redis>=2.10.0',
            'mock>=2.0.0',
            'mockredispy',
            'coverage',
            'pymongo>=3.9.0',
            'nose>=1.3.7',
            'nose-exclude>=0.5.0',
            'pytest',
        ],
        'experimental_aio': [
            "aiounittest ; python_version>='3.6'",
            "motor ; python_version>='3.6'",
        ],
    },
)
