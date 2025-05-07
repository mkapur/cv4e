## Project
## https://docs.google.com/document/d/1uehymxhfCPdapyxz7G6hPSSoyOQoOCS5cXVJ1xlDu5k/edit?tab=t.0
#%% imports and cleanup
## open metadata and print num images by category in dataset
import json
import os # to use listdir and other pwd
from collections import Counter ##
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from PIL import Image

#%% setup
headpath = r'c:/Users/kapur/Dropbox/other projects/cv4e/data/osu-small-animals-lila'

# Replace 'metadata.json' with the path to your JSON metadata file
metadata_file = r'c:\Users\kapur\Dropbox\other projects\cv4e\data\osu-small-animals.json'

# Open and load the JSON file
# this is enormous so don't call 'metadata' on its own 
with open(metadata_file, 'r') as file:
    metadata = json.load(file)

metadata.keys() # check the keys in the metadata
metadata['categories'] # check the info in the metadata

# Filter out blank images (those without 'category_id')
metadataImg = metadata['images'] # to load the images only
metadataAnn = metadata['annotations'] ## to load the labels only
category_strings = metadata['categories'] ## where the string labels are stored

# Assuming the metadata contains a list of images with a 'category' field
categories = [image['category_id'] for image in metadataAnn if 'category_id' in image]
len(categories) # check how many images are valid, should be 118k+


#%% Count the number of images in each category
category_counts = Counter(categories)

# Print the results
for category, count in category_counts.items():
    print(f"Category: {category}, Number of Images: {count}")

# Create a mapping from category_id to category name
category_mapping = {cat['id']: cat['name'] for cat in category_strings}

# Map category IDs in category_counts to their names
mapped_category_counts = {category_mapping.get(cat_id, f"ID {cat_id}"): count for cat_id, count in category_counts.items()}

filtered_category_counts = {category: count for category, count in mapped_category_counts.items() if count > 0}


sorted_category_counts = dict(sorted(filtered_category_counts.items(), 
key=lambda item: item[1]))
# Extract sorted categories and counts
categories2 = list(sorted_category_counts.keys())
counts = list(sorted_category_counts.values())

plt.bar(categories2, np.log(counts), color='indianred')
plt.xlabel('Category Name')
plt.ylabel('log Number of Images')
plt.title('Number of Images per Category')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()
 
#%%  how many unique locations exist in the dataset?
locations = [image['location'] for image in metadataImg if 'location' in image]
locations[:5]
len(set(locations)) # check how many unique locations exist in the dataset
Counter(locations) # frequency of each location

#%%  Pick 10 images at random and look at them to make sure the labels are correct…
## ten random integer image ID
random_integers = [random.randint(1, len(metadataImg)) for _ in range(10)]  # Random integers between 1 and 100
print(random_integers)

#%% make filepaths for the 10 random images and open them
## here are the full filepaths for the 10 random images
 
""" filepaths = [f"{headpath}/{metadataAnn[i]['image_id']}" for i in random_integers]
print(filepaths)
img = Image.open(filepaths[0])
img.show() """

#%% Map the string categories to category_id for the 10 random images
mapped_categories = [
    category_mapping.get(metadataAnn[i]['category_id'], f"Unknown ID {metadataAnn[i]['category_id']}")
    for i in random_integers if i < len(metadataAnn)
]

## loop thru the images, open them and print the category name
for"""  i in range(len(random_integers)):
    img = Image.open(filepaths[i])
    img.show()
    p """rint(f"Image {i + 1}: Category ID = {metadataAnn[random_integers[i]]['category_id']}, Category Name = {mapped_categories[i]}")

#%% how many images have multiple labels?
## going to count the number of image ids that are duplicated in the metadataAnnDF
## bring these into a dataframe to make it easier to work with
metadataAnnDF=pd.DataFrame(metadataAnn)
## count the number of entries in metatadataAnnDF for each image id
counts = metadataAnnDF['image_id'].value_counts()

metadataAnnDF['category_id'].value_counts() # check how many image ids are blank

# Filter for image_ids with counts greater than 1
duplicate_image_ids = counts[counts > 1]
len(duplicate_image_ids) ## none??

#%% are there any images with no labels?
metadataAnnDF['image_id'].isna().sum() 
(metadataAnnDF['image_id'] == '').sum()# check how many unique image ids are in the labels
#%% are there labels that correspond to images not in the dataset?
## this means we have to look at the metadataImgDF and 
# see if there are any image ids that are not in the metadataAnnDF
metadata.keys()
metadataCatDF=pd.DataFrame(metadata['categories'])

# Get unique IDs from metadataCatDF and metadataAnnDF
cat_ids = set(metadataCatDF['id']) ## makes a SET
ann_category_ids = set(metadataAnnDF['category_id'])

# Find the intersection
intersection_ids = cat_ids.intersection(ann_category_ids)
print(f"Intersection of IDs: {intersection_ids}")
print(f"Number of intersecting IDs: {len(intersection_ids)}")
 # Find IDs in cat_ids that are not in ann_category_ids
missing_ids = cat_ids.difference(ann_category_ids)
print(f"IDs in cat_ids but not in ann_category_ids: {missing_ids}")
print(f"Number of missing IDs: {len(missing_ids)}")


#%%  day 3 - start training with the YOLO classification framework
## choose the categories you want to train on. 
## first cut just randomly choose 80.20.
from sklearn.model_selection import train_test_split

# Drop blank images (NA or empty image_id)
filtered_metadata = metadataAnnDF[metadataAnnDF['image_id'].notna() & (metadataAnnDF['image_id'] != '')]

# Split into training and test sets (80% train, 20% test)
train_set, test_set = train_test_split(filtered_metadata, test_size=0.2, random_state=42)

# Print the sizes of the sets, these are DFs
print(f"Training set size: {len(train_set)}")
print(f"Test set size: {len(test_set)}")
type(train_set)
train_set.head(5) # check the first 5 rows of the training set

#f = open(os.path.join('..','documents')

# %% 
script_dir = os.path.dirname(os.path.abspath(__file__)) ## a la "here"

# Create YOLO-compatible directories relative to the script's directory

train_dir = os.path.join(script_dir, "..", "data/osu-small-animals-lila/yolo_data/train")
val_dir = os.path.join(script_dir, "..", "data/osu-small-animals-lila/yolo_data/val")
#os.makedirs(train_dir, exist_ok=True)
#os.makedirs(val_dir, exist_ok=True)

# YOLO reads the classes from the basedir. so make those folders, too
#os.makedirs(os.path.join(train_dir, "0"), exist_ok=True)
# check the path to the first image in the training set
#headpath
# Resize dimensions (e.g., 640x640 for YOLO)
resize_dim = (224, 224)
train_set.head()
#os.path.join(headpath, train_set['image_id'].iloc[0]) # check the path to the first image in the training set
#f"{headpath}/{ train_set['image_id'].iloc[0]}" # check the path to the first image in the training set
# Function to resize and save images
def resize_and_save_image(src_path, dest_path):
    with Image.open(src_path) as img:
        img_resized = img.resize(resize_dim)
        img_resized.save(dest_path)

# Create resized images and shortcuts for training images
for filepath in train_set['image_id'][0:10000]:
    filepath2 = f"{headpath}/{filepath}"  # check the path to the first image in the training set
    dest_dir = f"{headpath}/yolo_data/train/{os.path.basename(os.path.dirname((filepath)))}"  # check the path to the first image in the training set
    os.makedirs(dest_dir, exist_ok=True)  # Create the directory if it doesn't exist
    dest_path = f"{headpath}/yolo_data/train/{os.path.basename(os.path.dirname((filepath)))}/{os.path.basename(filepath)}"  # check the path to the first image in the training set
    resize_and_save_image(filepath2, dest_path)

# Create resized images and shortcuts for validation images
for filepath in test_set['image_id'][0:10000]:
    filepath2 = f"{headpath}/{filepath}"  # check the path to the first image in the training set
    dest_dir = f"{headpath}/yolo_data/val/{os.path.basename(os.path.dirname((filepath)))}"  # check the path to the first image in the training set
    os.makedirs(dest_dir, exist_ok=True)  # Create the directory if it doesn't exist
    dest_path = f"{headpath}/yolo_data/val/{os.path.basename(os.path.dirname((filepath)))}/{os.path.basename(filepath)}"  # check the path to the first image in the training set
    resize_and_save_image(filepath2, dest_path)    

# %%'c:/Users/kapur/Dropbox/other projects/cv4e/data/osu-small-animals-lila/yolo_data/val/'
## data has the root dir with test and train folders in it
yolo classify train model=yolo11n-cls data='c:/Users/kapur/Dropbox/other projects/cv4e/data/osu-small-animals-lila/yolo_data/' epochs=50 patience=10
# %%Evaluation
""" What does the framework say your accuracy was?
Compute the same thing yourself
Plot your train/val curves… how do they compare to the examples we showed earlier?  Does it look like your training went well?
Plot a confusion matrix (NB: yolo plots this for you, make sure they agree)
Pick an image that you think should be “easy”, first from the training set, which makes it extra easy.  Run the model on that image.  Did the model get it right?
 """
