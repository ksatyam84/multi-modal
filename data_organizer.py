import os
import csv
import requests

csv_file = "RAW_DATA/mymoviedb_EAS510.csv"
output_folder = "/data"

os.makedirs(output_folder, exist_ok=True)
