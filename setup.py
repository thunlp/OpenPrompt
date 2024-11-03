import setuptools
import os

def get_requirements(filename="requirements.txt"):
    """Reads dependencies from a requirements.txt file."""
    with open(filename, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    print("Requirements:", requirements)
    return requirements

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="openprompt",
    version="1.0.1",
    description="An open source framework for prompt-learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ning Ding, Shengding Hu, Weilin Zhao, Yulin Chen",
    author_email="dingn18@mails.tsinghua.edu.cn",
    license="Apache",
    url="https://github.com/thunlp/OpenPrompt",
    keywords=["PLM", "prompt", "AI", "NLP"],
    python_requires=">=3.6.0",
    install_requires=get_requirements(),
    packages=setuptools.find_packages(),
    extras_require={
        "optional": ["sklearn"],
        "torch": ["torch"]  # You can list other optional dependencies here as needed.
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
