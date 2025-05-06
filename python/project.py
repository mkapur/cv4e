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
[metadataAnn[i]['id'] for i in random_integers]
# %% now let's look at the images
# Assuming 'file_name' in metadataImg contains the relative path to the image
for random_id in random_integers:
    # Find the image entry with the matching 'id'
    image_entry = next((img for img in metadataImg if img['id'] == random_id), None)
    
    #if image_entry:
        # Get the file path of the image
        #image_path = os.path.join(r'c:\Users\kapur\Dropbox\other projects\cv4e\data\Images', image_entry['file_name'])  # Replace 'path_to_images_directory' with the actual directory
    image_path = os.path.join(r'c:\Users\kapur\Dropbox\other projects\cv4e\data\Images', metadataAnn[random_id]['id'])
        # Check if the image file exists
    print(f"Opening image: {image_path}")
        
        # Open the image (e.g., using PIL or OpenCV)
      
        img = Image.open(image_path)
        img.show()
    else:
        print(f"No image found for id: {random_id}")




# %%
