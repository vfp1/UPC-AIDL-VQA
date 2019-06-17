from vqaHelpers import vqaIngestion
# Path to the root folder of the Google Drive data
path = r"G:\My Drive\Studies\UPC-AIDL\VQA\data"

# Unzip the Images
vqaIngestion.VQADataset().imageUnzip(path)