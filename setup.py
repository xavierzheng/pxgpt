"""Setup script for PXGPT."""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [
        line.split("#")[0].strip()
        for line in f
        if line.strip() and not line.lstrip().startswith("#")
    ]

setup(
    name="pxgpt",
    version="0.4.0",
    description="Plant analysis tool with multiple LLM provider support",
    author="PXGPT Team", 
    packages=find_packages(),
    install_requires=requirements,
    # Test-only, deliberately NOT in requirements.txt: that file becomes
    # install_requires above, so anything added there is forced on every user
    # who installs pxGPT. Install with:  pip install -e ".[dev]"
    extras_require={
        "dev": [
            "pytest",
            "packaging",   # tests/test_ops_local_vllm_setup.py parses PEP 508
            # test_version_single_source.py introspects this file to check that
            # test-only deps stay out of install_requires. Declared explicitly
            # because a Python 3.12+ venv no longer ships setuptools, so the
            # test passed on a developer box and failed in a clean environment.
            "setuptools",
        ],
    },
    python_requires=">=3.10",
    entry_points={
        'console_scripts': [
            'pxgpt=pxgpt.main:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License", 
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
