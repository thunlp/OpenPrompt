import setuptools
import os
import importlib

requires = """
transformers>=4.10.0
sentencepiece==0.1.96
# scikit-learn>=0.24.2
tqdm>=4.62.2
tensorboardX
nltk
yacs
dill
datasets
rouge==1.0.0
pyarrow
scipy
"""

def get_requirements():
    ret = [x for x in requires.split("\n") if len(x)>0]
    print("requirements:", ret)
    return ret



# path = os.path.dirname(os.path.abspath(__file__))
# requires =  get_requirements(path)
# print("requirements:")
# print(requires)

with open('README.md', 'r') as f:
    setuptools.setup(
        name = 'openprompt',
        version = '1.0.1',
        description = "An open source framework for prompt-learning.",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        author = 'Ning Ding, Shengding Hu, Weilin Zhao, Yulin Chen',
        author_email = 'dingn18@mails.tsinghua.edu.cn',
        license="Apache",
        url="https://github.com/thunlp/OpenPrompt",
        keywords = ['PLM', 'prompt', 'AI', 'NLP'],
        python_requires=">=3.6.0",
        install_requires=get_requirements(),
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

