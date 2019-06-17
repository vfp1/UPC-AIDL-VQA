from vqaHelpers.vqaIngestion import VQADataset

# Path to the root folder of the Google Drive data
path = r"G:\My Drive\Studies\UPC-AIDL\VQA\data"

# Unzip the Images
VQADataset().imageUnzip(path)