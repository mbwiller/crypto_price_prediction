# setup.py
import setuptools

setuptools.setup(
    name="crypto_price_prediction_mdp",      # your package name on PyPI (or local)
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description="An advanced MDP for minute‑level crypto price prediction",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/crypto-price-prediction-mdp",
    packages=setuptools.find_packages(),      # automatically pick up crypto_mdp/
    include_package_data=True,               # include files from MANIFEST.in
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy",
        "scikit-learn",
        "ta-lib",
        "gym"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
