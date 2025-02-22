import os
import csv
import requests
import pandas as pds


csv_file = "RAW_DATA/mymoviedb_EAS510.csv"
output_folder = "/datasets/all_poster_images"

os.mkdir(output_folder, exist_ok=True)


data = pd.read_csv(csv_file)
print(data)






"""# Read the CSV file
with open(csv_file, newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header if exists

    for row in reader:
        # Assuming CSV format: Category, Image_URL
        category, image_url = row[0], row[1]

        # Create a folder for the category
        category_folder = os.path.join(output_folder, category)
        os.makedirs(category_folder, exist_ok=True)

        # Download and save the image
        try:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                image_name = os.path.basename(image_url)
                image_path = os.path.join(category_folder, image_name)

                with open(image_path, "wb") as img_file:
                    for chunk in response.iter_content(1024):
                        img_file.write(chunk)
                print(f"Downloaded: {image_path}")
            else:
                print(f"Failed to download {image_url}")
        except requests.RequestException as e:
            print(f"Error fetching {image_url}: {e}")

print("Processing complete.")"""
