import setuptools
import os
import importlib

def get_requirements(path):
    ret = []
    with open(os.path.join(path, "requirements.txt"), encoding="utf-8") as freq:
        for line in freq.readlines():
            ret.append( line.strip() )
    return ret


path = os.path.dirname(os.path.abspath(__file__))
requires =  get_requirements(path)
print("requirements:")
print(requires)

with open('README.md', 'r') as f:
    setuptools.setup(
        name = 'openprompt',
        version = '1.0.0',
        description = "An open source framework for prompt-learning.",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        author = 'Ning Ding, Shengding Hu, Weilin Zhao, Yulin Chen',
        author_email = 'dingn18@mails.tsinghua.edu.cn',
        license="Apache",
        url="https://github.com/thunlp/OpenPrompt",
        keywords = ['PLM', 'prompt', 'AI', 'NLP'],
        python_requires=">=3.6.0",
        install_requires=requires,
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ]
    )

required_list = ["torch"]
for package in required_list:
    try:
        m = importlib.import_module(package)
    except ModuleNotFoundError:
        print("\n"+"="*30+"  WARNING  "+"="*30)
        print(f"{package} is not found on your environment, please install it manually.")
        print("We do not install it for you because the environment sometimes needs special care.")

optional_list = ["sklearn"]
for package in optional_list:
    try:
        m = importlib.import_module(package)
    except ModuleNotFoundError:
        print("\n"+"="*30+"  WARNING  "+"="*30)
        print(f"{package} is not found on your environment, please install it if the specific script needs.")

