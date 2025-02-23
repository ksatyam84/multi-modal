import os
import csv
import requests
import pandas as pd


csv_file = "RAW_DATA/mymoviedb_EAS510.csv"
ROOT_FOLDER = "datasets/"

"""
try:
    os.makedirs(output_folder, exist_ok=True)
    print(f"Directory '{output_folder}' created or already exists.")
except FileExistsError:
    print(f"A file with the name '{output_folder}' already exists.")"""

data = pd.read_csv(csv_file) #created a Pandas Dataframe object

def download_all_image_files(local_path, url_array):

    for url in url_array:
        try: 
            response = requests.get(url, stream=True)
            print(response)
            response.raise_for_status()
            
            filename = local_path +"/" + url.split("/")[-1]
            #print(f"filename:, {filename}")
            
            with open(filename, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    
            print(f"Downloaded: {url}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")

    print(f"Download completed for path: {local_path}")


def get_genres(input_dataframe):

    indiv_genres = get_indiv_elements_from_column("Genre", input_dataframe)

    return indiv_genres


def get_titles(input_dataframe):

    indiv_titles = get_indiv_elements_from_column("Title", input_dataframe)
    
    return indiv_titles
    
def get_img_urls(input_dataframe):

    indiv_img_urls = get_indiv_elements_from_column("Poster_Url", input_dataframe)
    
    return indiv_img_urls

def get_indiv_elements_from_column(name, input_dataframe):

    indiv_elements = []

    for element_set in input_dataframe.get(name):
        if not isinstance(element_set, float):
            for element_item in element_set.split(', '):
                if element_item not in indiv_elements:
                    indiv_elements.append(element_item)
    

    return indiv_elements


def print_elements(name, indiv_elements):

    print(f"Number of Individual {name}: {len(indiv_elements)}")
    print(f"\n Here they are: \n ")
    for element in indiv_elements:
        index = indiv_elements.index(element)
        print(f"Index: {index+1}, {element}")

def mk_element_dir(name, path=ROOT_FOLDER):

    new_path = path + name

    try:
        os.makedirs(new_path, exist_ok=True)
        print(f"Directory {new_path} created successfully")
    except FileExistsError:
        print(f"A file with the name '{new_path}' already exists.")


def mk_genre_image_dict(input_dataframe):

    element_dict = []
    genres = get_genres(input_dataframe)
    
    for genre in genres:
        element_dict.append(genre)

    for element_set in input_dataframe.get("Genres"):
        print(element_dict)

    return element_dict

   
def mkdir_elements(sample_name, label_array, input_dataframe):
    
    train_sample_name = sample_name + "/Train"
    test_sample_name = sample_name + "/Test"


    try:
        os.makedirs(ROOT_FOLDER + sample_name, exist_ok=True)
        print(f"Directory {sample_name} created successfully")

        mk_element_dir(train_sample_name)
        for label in label_array: 
            mk_element_dir(label,ROOT_FOLDER + train_sample_name + "/")


        mk_element_dir(test_sample_name)
        for label in label_array: 
            mk_element_dir(label,ROOT_FOLDER + test_sample_name + "/")


    except FileExistsError:
        print(f"A file with the name '{sample_name}' already exists.")


def collect_element_urls(element_name, input_dataframe, label_col_name, url_col_name):
    
    element_urls = []
    tl_movies = 0
    i = 0

    for genre_str in input_dataframe.get(label_col_name):
        #print(genre_str)
        if not isinstance(genre_str, float):

            if element_name in genre_str:
                element_urls.append(input_dataframe.get(url_col_name)[i])
                tl_movies+=1
        i+=1
    
    print(f"i: {i}, {element_name} movies: {tl_movies}, {element_name} urls: {len(element_urls)}")

    return element_urls



def ld_img_dir(path, label_array):


    for label in label_array:

        url_array = collect_element_urls(label, data, "Genre", "Poster_Url")

        train_url_array = url_array[:int(len(url_array)*.8)]
        test_url_array = url_array[int(len(url_array)*.8):]

        print(f"train_url_array length: {len(train_url_array)}")
        print(f"test_url_array length: {len(test_url_array)}")  
    
        #print(path+"/Train/"+label)

        download_all_image_files(path+"/Train/"+label, train_url_array)

        #print(path+"/Test/"+label)
        download_all_image_files(path+"/Test/"+label, test_url_array)

    print(f"Full sample image set download complete")
    


def main():
    # Code to be executed when the script is run directly   
    #print(data)

    #print_elements("Genres",get_genres(data))
    #mk_genre_image_dict(data)
    #mkdir_elements("SampleV0", get_genres(data), data)
    label_array = ['Action', 'Documentary', 'Romance']

    

    ld_img_dir(ROOT_FOLDER + "SampleV0", label_array)
    #print_elements("Titles",get_titles(data))
    #clearprint_elements("Urls",get_img_urls(data))

    #print(data)
    #if "Action" in data.get("Genre")[0]:
       # print(data.get("Poster_Url")[0])
    




if __name__ == "__main__":
    main()
