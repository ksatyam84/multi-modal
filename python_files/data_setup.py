import os
import csv
import requests
import pandas as pd

import matplotlib.pyplot as plt

import torch

import time

import torchvision
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

ROOT_FOLDER = ""
NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int, num_workers = NUM_WORKERS):

    """ Creates training and testing DataLoaders.
        
        Takes in a training directory and testing directory path and turns
        them into PyTorch Datasets and then into PyTorch DataLoaders.
        
        Args:
            train_dir: Path to training directory.
            test_dir: Path to testing directory.
            transform: torchvision transforms to perform on training and testing data.
            batch_size: Number of samples per batch in each of the DataLoaders.
            num_workers: An integer for number of workers per DataLoader.

        Returns:
            A tuple of (train_dataloader, test_dataloader, class_names).
            Where class_names is a list of the target classes.

        Example usage:
            train_dataloader, test_dataloader, class_names = 
                = create_dataloaders(train_dir=path/to/train_dir,
                                    test_dir=path/to/test_dir,
                                    transform=some_transform,
                                    batch_size=32,
                                    num_workers=4)

        If we'd like to make DataLoader 's we can now use the function within data_setup.py like so:
        
        Let's put our TinyVGG() model class into a script, note sometimes people just name it models.py and put all their baseline models (three or four) in one file.
"""
    train_data = datasets.ImageFolder(train_dir, transform = transform)
    test_data = datasets.ImageFolder(test_dir, transform = transform)

    classes = train_data.classes

    train_dataloader = DataLoader(train_data, batch_size, shuffle=True, num_workers = NUM_WORKERS, pin_memory= True)

    test_dataloader = DataLoader(test_data, batch_size, shuffle=True, num_workers = NUM_WORKERS, pin_memory= True)

    return train_dataloader, test_dataloader, classes


def set_root_path(path):

    ### Set the root path for the project, BEFORE CALLING ANY FUNCTION TO BUILD CUSTOM DATA SAMPLES.
    # Args:
    #     path (str): The root directory path to be set.
    # Returns:
    #     str: The root directory path.

    ROOT_FOLDER = path

    return ROOT_FOLDER

def download_all_image_files(local_path, url_array):

    ### Download all images from the given URLs and save them to the specified local path.
    # Args: 
    #     local_path (str): The local directory path where the images will be saved.
    #     url_array (list): A list of URLs pointing to the images to be downloaded.
    # Returns: 
    #     None

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
    
    ### Get a list of unique genres from the input dataframe.
    # Args:
    #     input_dataframe (pd.DataFrame): The input dataframe containing movie data.
    # Returns:
    #     list: A list of unique genres.

    indiv_genres = get_indiv_elements_from_column("Genre", input_dataframe)

    return indiv_genres

def get_titles(input_dataframe):
    
    ### Get a list of unique titles from the input dataframe.
    # Args:
    #     input_dataframe (pd.DataFrame): The input dataframe containing movie data.
    # Returns:
    #     list: A list of unique titles.

    indiv_titles = get_indiv_elements_from_column("Title", input_dataframe)
    
    return indiv_titles
    
def get_img_urls(input_dataframe):
    
    ### Get a list of unique image URLs from the input dataframe.
    # Args:
    #     input_dataframe (pd.DataFrame): The input dataframe containing movie data.
    # Returns:      
    #     list: A list of unique image URLs.

    indiv_img_urls = get_indiv_elements_from_column("Poster_Url", input_dataframe)
    
    return indiv_img_urls

def get_indiv_elements_from_column(name, input_dataframe):
    
    ### Get a list of unique elements from a specified column in the input dataframe.
    # Args:
    #     name (str): The name of the column to extract elements from.
    #     input_dataframe (pd.DataFrame): The input dataframe containing movie data.
    # Returns:
    #     list: A list of unique elements from the specified column.

    indiv_elements = []

    for element_set in input_dataframe.get(name):
        if not isinstance(element_set, float):
            for element_item in element_set.split(', '):
                if element_item not in indiv_elements:
                    indiv_elements.append(element_item)
    

    return indiv_elements

def print_elements(name, indiv_elements):
    
    ### Print the number of unique elements and their indices from a specified list.
    # Args:
    #     name (str): The name of the list to be printed.
    #     indiv_elements (list): The list of unique elements to be printed.
    # Returns:
    #     None

    print(f"Number of Individual {name}: {len(indiv_elements)}")
    print(f"\n Here they are: \n ")
    for element in indiv_elements:
        index = indiv_elements.index(element)
        print(f"Index: {index+1}, {element}")

def mk_element_dir(name, path=ROOT_FOLDER):
    
    ### Create a directory for a specified element.
    # Args:
    #     name (str): The name of the element for which the directory will be created.
    #     path (str): The path where the directory will be created. Defaults to ROOT_FOLDER.
    # Returns:
    #     None

    new_path = path + name

    try:
        os.makedirs(new_path, exist_ok=True)
        print(f"Directory {new_path} created successfully")
    except FileExistsError:
        print(f"A file with the name '{new_path}' already exists.")

def mk_genre_image_dict(input_dataframe):
    
    ### Create a dictionary of genres and their corresponding image URLs from the input dataframe.
    # Args:
    #     input_dataframe (pd.DataFrame): The input dataframe containing movie data.
    # Returns:
    #     list: A list of dictionaries, each containing a genre and its corresponding image URL.

    element_dict = []
    genres = get_genres(input_dataframe)
    
    for genre in genres:
        element_dict.append(genre)

    for element_set in input_dataframe.get("Genres"):
        print(element_dict)

    return element_dict
 
def mkdir_elements(sample_name, label_array, input_dataframe):
    
    ### Create directories for a sample and its corresponding labels.
    # Args:
    #     sample_name (str): The name of the sample for which directories will be created.
    #     label_array (list): A list of labels for which directories will be created.
    #     input_dataframe (pd.DataFrame): The input dataframe containing movie data.
    # Returns:
    #     None

    
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
    
    ### Collect URLs of a specific element from the input dataframe.
    # Args:
    #     element_name (str): The name of the element for which URLs will be collected.
    #     input_dataframe (pd.DataFrame): The input dataframe containing movie data.
    #     label_col_name (str): The name of the column containing labels.
    #     url_col_name (str): The name of the column containing URLs.
    # Returns:
    #     list: A list of URLs corresponding to the specified element.

    
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
    
    ### Downoad images from a specified directory and organize them into training and testing sets.
    # Args:
    #     path (str): The path to the directory containing images.
    #     label_array (list): A list of labels corresponding to the images.
    # Returns:
    #     None


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
    
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    # Your code here
    image_transform = transforms.Compose([transforms.ToDtype(torch.uint8, scale=True), transforms.Resize(size=(224, 224),
                                         antialias=True), transforms.RandomPerspective(distortion_scale=0.2, p=.4), transforms.ToImage(),transforms.ToDtype(torch.float32, scale=True), transforms.Normalize([.5, .5, .5], [.1, .1, .1])])
    
    train_dataloader, test_dataloader, class_names = create_dataloaders("/Users/kkodweis/Github-Repos/EAS510-BasicsAI/multi-modal/datasets/SampleV0/Train", "/Users/kkodweis/Github-Repos/EAS510-BasicsAI/multi-modal/datasets/SampleV0/Test", image_transform, batch_size=32)

    i = 0
    for batch in tqdm(train_dataloader, "Train Batches: "):
        # Process the batch
        i+=1
        #inputs, labels = batch
        #print("Input batch shape:", inputs.shape)
        #print("Label batch shape:", labels.shape)

    j = 0
    for batch in tqdm(test_dataloader, "Test Batches: "):
        # Process the batch
        j+=1
        #inputs, labels = batch
        #print("Input batch shape:", inputs.shape)
        #print("Label batch shape:", labels.shape)

    print(f"Number of Train Batchs: {i}")

    print(f"Number of Test Batchs: {j}")

    #print(class_names)

if __name__ == "__main__":
    main()