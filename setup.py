from pathlib import Path

from setuptools import setup


readme = Path('README.md').read_text(encoding='utf-8')

setup(
    name='srdatagen',
    version='0.1.0',
    description=(
        'Synthetic VQA data generation for SpatialReasoner.'
    ),
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Wufei Ma',
    author_email='wufeim@gmail.com',
    url="https://github.com/wufeim/SpatialReasonerDataGen",
    project_urls={
        "Documentation": "https://spatial-reasoner.github.io/",
    },
    packages=["srdatagen"],
    package_dir={"srdatagen": "srdatagen"},
    include_package_data=True,
    python_requires=">=3.8",
    license="CC-BY-4.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Programming Language :: Python",
        "Topic :: Software Development",
    ],
)
