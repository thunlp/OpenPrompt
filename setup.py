import setuptools

with open('README.md', 'r') as f:
    setuptools.setup(
        name = 'openprompt',
        version = '1.0',
        description = "An open source framework for prompt-learning.",
        author = 'Ning Ding, Shengding Hu, Weilin Zhao',
        author_email = 'dingn18@mails.tsinghua.edu.cn',
        license="Apache",
        keywords = ['PLM', 'prompt', 'AI', 'NLP'],
        python_requires=">=3.6.0",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Intended Audience :: Researchers",
            "Intended Audience :: Students",
            "Intended Audience :: Developers",

        ]
    )