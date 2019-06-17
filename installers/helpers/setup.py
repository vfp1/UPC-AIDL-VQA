import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='vqaHelpers',
     version='0.2',
     scripts=['vqaHelpers/vqaIngestion.py', 'vqaHelpers/vqaEval.py'] ,
     author="Victor Pajuelo Madrigal",
     author_email="victor@eogora.com",
     description="A VQA helper developed by"
                 "Aishwarya Agrawal and packaged by"
                 "Victor Pajuelo Madrigal",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/vfp1/VQA",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )