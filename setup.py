import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="text_generator",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for generating text from questions/queries inputs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/text_generator",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["openai", "PyPDF2", "transformers", "torch"],
)
