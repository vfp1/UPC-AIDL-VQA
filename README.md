# UPC-AIDL-VQA
An attempt to address the VQA challenge for the Artificial Intelligence for Deep Learning postgraduate

## Legacy
This project derives from previous attempts on addressing VQA. Following there is a list of legacy code:

* [VQA helpers adapted to Python 3.5](https://github.com/vfp1/VQA)

## Installers
This repository needs the installing of a package contain within it. Install it in your environment following
the following instructions.

### English model from Spacy

```python
python -m spacy download en
```

### Ngrok installation
```bash
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip
```

#### Preparing the datasets
The datasets are given as in the [VQA Challenge](https://visualqa.org/download.html). Those are uploaded to a Google Drive 
link, which will not be posted publicly here. However, those are zipped. To unzip, do the following after **vqaHelpers** installation. 
Grab a cup of coffe, it will take quite a while, it is **30GB** after all:

``` python
from vqaHelpers import vqaIngestion
# Path to the root folder of the Google Drive data
path = r"G:\My Drive\Studies\UPC-AIDL\VQA\data"

# Unzip the Images
vqaIngestion.VQADataset().imageUnzip(path)
```

