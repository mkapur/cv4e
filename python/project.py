## Project
## https://docs.google.com/document/d/1uehymxhfCPdapyxz7G6hPSSoyOQoOCS5cXVJ1xlDu5k/edit?tab=t.0
#%% imports and cleanup
## open metadata and print num images by category in dataset
import json
import os # to use listdir and other pwd
from collections import Counter##
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
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
# %%
#%%  Pick 10 images at random and look at them to make sure the labels are correct…
## ten random integer image ID
random_integers = [random.randint(1, len(metadataImg)) for _ in range(10)]  # Random integers between 1 and 100
print(random_integers)
headpath = r'c:/Users/kapur/Dropbox/other projects/cv4e/data/osu-small-animals-lila'

#%% make filepaths for the 10 random images and open them
## here are the full filepaths for the 10 random images
 
filepaths = [f"{headpath}/{metadataAnn[i]['image_id']}" for i in random_integers]
print(filepaths)
img = Image.open(filepaths[0])
img.show()

#%% Map the string categories to category_id for the 10 random images
mapped_categories = [
    category_mapping.get(metadataAnn[i]['category_id'], f"Unknown ID {metadataAnn[i]['category_id']}")
    for i in random_integers if i < len(metadataAnn)
]

## loop thru the images, open them and print the category name
for i in range(len(random_integers)):
    img = Image.open(filepaths[i])
    img.show()
    print(f"Image {i + 1}: Category ID = {metadataAnn[random_integers[i]]['category_id']}, Category Name = {mapped_categories[i]}")

