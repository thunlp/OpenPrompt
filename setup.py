import setuptools
import os

def get_requirements(path):
    ret = []
    with open(os.path.join(path, "requirements.txt"), encoding="utf-8") as freq:
        for line in freq.readlines():
            ret.append( line.strip() )
    return ret


path = os.path.dirname(os.path.abspath(__file__))
requires =  get_requirements(path)
print("aaaaa")
print(requires)

with open('README.md', 'r') as f:
    setuptools.setup(
        name = 'openprompt',
        version = '0.1.2',
        description = "An open source framework for prompt-learning.",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        author = 'Ning Ding, Shengding Hu, Weilin Zhao, Yulin Chen',
        author_email = 'dingn18@mails.tsinghua.edu.cn',
        license="Apache",
        url="https://github.com/thunlp/OpenPrompt",
        keywords = ['PLM', 'prompt', 'AI', 'NLP'],
        python_requires=">=3.8.0",
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