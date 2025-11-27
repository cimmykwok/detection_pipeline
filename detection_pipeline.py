import ast
import mmcv
import numpy as np
from argparse import ArgumentParser
from mmdet.apis import init_detector, inference_detector
import cv2 
import os
import torch
import shutil
import re 
import json
from argparse import ArgumentParser
from mmocr.apis.inferencers import MMOCRInferencer
from output_file import SpottedID, DetectedTU, DetectedItems  # Import the classes from the other file
from detected_object import DetectedObject  # Import the classes from the other file
from config import settings  # Import the settings dictionary

def is_bbox_contained(bbox_A, bbox_B):
    """Check if bbox A is contained within bbox B."""
    x1_A, y1_A, x2_A, y2_A = bbox_A
    x1_B, y1_B, x2_B, y2_B = bbox_B
    
    return (x1_A >= x1_B and
            y1_A >= y1_B and
            x2_A <= x2_B and
            y2_A <= y2_B)

def get_bounding_boxes(polygons):
    bboxes = []
    for polygon in polygons:
        # Assuming polygon is a flat list of coordinates
        x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
        y_coords = [polygon[i] for i in range(1, len(polygon), 2)]

        
        min_x = min(x_coords)
        min_y = min(y_coords)
        max_x = max(x_coords)
        max_y = max(y_coords)
        bboxes.append([min_x, min_y, max_x, max_y])  # Format: [min_x, min_y, max_x, max_y]
    return bboxes

def assign_values():
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    values = {}
    current_value = 10  # Start at 10 for A
    
    for char in alphabet:
        if current_value % 11 == 0:  # If current value is a multiple of 11, increment
            current_value += 1
        
        values[char] = current_value  # Assign the value
        current_value += 1  # Increment to the next value
    return values

def replace_ukn(text):
    # Replace all instances of '<UKN>' with an empty string
    #print(f"Original text: {text}")  # Debug statement
    replaced_text = text.replace('<UKN>', '')
    #print(f"Replaced text: {replaced_text}")  # Debug statement
    return replaced_text

def is_valid_container_id(container_id):
    # Validate input length
    regex = r'^[A-Z]{4}[0-9]{7}$'
    
    # Check basic regex format
    if not re.match(regex, container_id):
        print(f"Not valid format: {container_id}")  # Correct print statement
        return False
    
    # Assign numerical values for letters (A=10, ..., Z, skipping multiples of 11)
    char_values = {}
    value = 10
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if value % 11 == 0:
            value += 1  # Skip multiples of 11
        char_values[char] = value
        value += 1
    
    # Assign weights: 2^i % 11
    weights = [(2 ** i) % 11 for i in range(10)]
    
    # Convert characters to their values
    values = []
    for i, char in enumerate(container_id[:10]):
        if char.isalpha():  # For letters
            if char not in char_values:
                raise ValueError(f"Invalid character {char} in container ID.")
            values.append(char_values[char])
        elif char.isdigit():  # For digits
            values.append(int(char))
        else:
            raise ValueError(f"Invalid character {char} in container ID.")
    
    # Calculate weighted sum
    weighted_sum = sum(value * weights[i] for i, value in enumerate(values))
    
    # Compute check digit
    check_digit = weighted_sum % 11
    return check_digit == int(container_id[10:])  # Compare the calculated digit with the provided one

def replace_first_four_if_not_all_alpha(s, logo_mapping):
    # Check if logo_mapping is empty or None
    if logo_mapping == '' or logo_mapping is None:
        return s  # Return the original string if logo_mapping is empty or None
    
    # check if the prefix is correct
    if logo_mapping != s[:4]:
        temp_id = logo_mapping + s[4:]
        #print(temp_id)  # Consider commenting or removing this line in production
        if is_valid_container_id(temp_id):  # Use temp_id directly here
            return temp_id
        
    # Initialize prefix length and alpha length
    prefix_length = 0
    alpha_length = 0
    alpha_count = 0

    # Iterate over the first four characters
    for i, c in enumerate(s):
        #print(f"i: {i}")  # Correct print statement
        prefix_length += 1  # Count each character
        if c.isalpha():
            alpha_length += 1  # Count alphabetic characters
            alpha_count += 1
            #print(c)
            if i == 3:
                alpha_length = 4
        if prefix_length >= 4:  # Stop after the fourth character
            break

    # If we found at least one alphabetic character and not all four are alphabetic
    if alpha_length > 0 and alpha_count != prefix_length:
        s = logo_mapping + s[alpha_length:]  # Replace prefix with logo_mapping
    return s
            
    # Replace the prefix (first up to 4 characters) with `logo_mapping` if there are any alphabetic characters
    if alpha_length > 0 and alpha_count!=4:
        s = logo_mapping + s[prefix_length:]
    return s

def calculate_check_digit(container_number):
    values = assign_values()  # Get the value mapping
    total_sum = 0

    # Check the length of the container_number
    if len(container_number) == 11:
        input_container = container_number[:-1]  # Ignore the last character (check digit)
    else:
        input_container = container_number  # Use the full string if it is 10 characters
    
    for i, char in enumerate(input_container):
        if char.isalpha():  # For letters
            value = values[char]
        else:  # For digits
            value = int(char)
        
        # Multiply by the power of 2
        total_sum += value * (2 ** i)

    # Step 3: Calculate check digit
    integer_part = total_sum // 11  # Floor division to get integer part
    multiplied = integer_part * 11
    check_digit = total_sum - multiplied

    # Adjust if check digit is 10
    if check_digit == 10:
        check_digit = 0

    return input_container + str(check_digit)  # Append the check digit and return it

# Setup configuration
config_file = settings["config_file"]
checkpoint_file = settings["checkpoint_file"]
text_detection_config = settings["text_det_config"]
text_recognition_config = settings["text_rec_config"]
text_detection_weight = settings["text_det_weight"]
text_recognition_weight = settings["text_rec_weight"]
output_dir = settings["output_dir"]
container_dir = settings["container_dir"]
id_text_dir = settings["id_text_dir"]
output_result = settings["ouput_result_filename"]
area_threshold = settings["area_threshold"]

# Define the valid labels we want to check against
valid_container_labels = settings["valid_container_labels"]
valid_id_text_labels = settings["valid_id_text_labels"]
valid_logo_mapping = settings["valid_logo_mapping"]

score_threshold = settings["score_threshold"]
largest_n_containers_no = settings["largest_n_containers_no"]
img = settings["input_image"]
input_folder = 'images'

id_text_objects = []
container_objects = []
logo_objects = []

# Initialize the detector
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Get the directory of the original image
image_dir = os.path.dirname(img)

# Get the name of the image file (without the extension)
img_name = os.path.basename(img)  # 'GX010007_done_29469.jpg'
img_name_without_extension = os.path.splitext(img_name)[0]  # 'GX010007_done_29469'

# Create a new folder with the image name under image_dir
output_dir = os.path.join(image_dir, output_dir)  # Create output directory under image_dir
image_name_dir = os.path.join(output_dir, img_name_without_extension)  # './images/GX010007_done_29469'
id_text_dir = os.path.join(image_name_dir, id_text_dir)
container_dir = os.path.join(image_name_dir, container_dir)

# Read logo mapping from JSON file
with open(valid_logo_mapping, 'r') as jsonfile:
    logo_mapping = json.load(jsonfile)

# Delete the directory if it exists
if os.path.exists(image_name_dir):
    shutil.rmtree(image_name_dir)  # Remove the entire directory

# Create the new directory
os.makedirs(image_name_dir, exist_ok=True)
os.makedirs(id_text_dir, exist_ok=True)
os.makedirs(container_dir, exist_ok=True)

# Get the class names from model metadata
classes = model.dataset_meta.get('classes')
#print("Check classes:", classes)

# Perform inference on the image
#result = inference_detector(model, input_folder)
result = inference_detector(model, img)
pred_instances = result.pred_instances

# Check if predictions were made
if pred_instances is not None and pred_instances.bboxes.numel() > 0:
    # Convert tensors to NumPy arrays for easier handling
    boxes = pred_instances.bboxes.cpu().numpy()  # Bounding boxes
    labels = pred_instances.labels.cpu().numpy()  # Labels
    scores = pred_instances.scores.cpu().numpy()  # Scores for the detections
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # Area = (x2 - x1) * (y2 - y1)

    # Filter boxes based on score > 0.8
    high_score_indices = np.where(scores > score_threshold)[0]
    
    # Select the filtered boxes, labels, and scores
    filtered_boxes = boxes[high_score_indices]
    filtered_labels = labels[high_score_indices]
    filtered_scores = scores[high_score_indices]
    filtered_areas = areas[high_score_indices]  # Use areas here to store the correct values

    # Loop through filtered results and create DetectedObject instances
    for i in range(len(filtered_boxes)):
        detected_obj = DetectedObject(
            name=classes[filtered_labels[i]],  # Use class name corresponding to the label
            object_id=i + 1,                   # Unique ID (could also be generated differently)
            box=filtered_boxes[i],         # Add the box as a list
            label=classes[filtered_labels[i]],  # Use the corresponding class name
            score=filtered_scores[i],       # Keep scores in list form
            area=filtered_areas[i]          # Keep areas in list form
        )
        # Append to the detected_objects list
        # Collect 'id_text' boxes and container boxes
        if classes[filtered_labels[i]] in valid_id_text_labels:
            id_text_objects.append(detected_obj)
        elif classes[filtered_labels[i]] in valid_container_labels:
            container_objects.append(detected_obj)
        elif classes[filtered_labels[i]] in logo_mapping:
            logo_objects.append(detected_obj)

    # Iterate through the detected objects
    for i in id_text_objects:
        # Check if the label 'id_text' is in the object's labels
        for j in container_objects:
            if is_bbox_contained(i.box, j.box):
                # Update the parent_id (or append to parent_ids if it's a list)
                i.add_parent_id(j.object_id)

    # Iterate through the detected objects
    for i in logo_objects:
        # Check if the label 'id_text' is in the object's labels
        for j in container_objects:
            if is_bbox_contained(i.box, j.box):
                # Update the parent_id (or append to parent_ids if it's a list)
                j.add_logo(logo_mapping[i.label])  

    # Filter container objects to only include those with area greater than the threshold
    filtered_objects = [obj for obj in container_objects if obj.area > area_threshold]

    sorted_objects = sorted(container_objects, key=lambda x: x.area, reverse=True)

    # Sort the filtered objects by area
    # sorted_objects = sorted(filtered_objects, key=lambda x: x.area, reverse=True)

    # OOCR - Initialize the inferencer
    ocr = MMOCRInferencer(
        det=text_detection_config,
        det_weights=text_detection_weight,
        rec=text_recognition_config,
        rec_weights=text_recognition_weight,
    )
    
    detected_items = DetectedItems()

    # Ensure you have the relevant variables from the sorted container_objects
    if len(container_objects) > 0:  # Ensure there are container objects present
        # Sort objects by area (we directly use the area from the DetectedObject instance)
        sorted_container_objects = sorted(container_objects, key=lambda obj: obj.area, reverse=True)

        # Get the top N container objects based on area
        top_container_objects = sorted_container_objects[:largest_n_containers_no]

        #print(top_container_objects)
        # Load original image for cropping
        image = cv2.imread(img)
        height, width = image.shape[:2] 

        # crop all images and store them
        cropped_images = []
        for obj in top_container_objects:

            x1, y1, x2, y2 = obj.box.astype(int)  # Convert box coordinates to integers
            # Check bounds
            # Ensure cropping coordinates are within bounds
            x1 = max(0, min(x1, width))  
            y1 = max(0, min(y1, height)) 
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            cropped_image = image[y1:y2, x1:x2]  # Crop the image
            print(f"Original Image Shape: {cropped_image.shape}")
            print(f"Cropping coordinates: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
            cropped_images.append(cropped_image)  # Store the cropped image

            # Create a filename for the cropped image and save it
            cropped_image_filename = os.path.join(container_dir, f'id_{obj.object_id}_{obj.label}.png')
            cv2.imwrite(cropped_image_filename, cropped_image)

        # run inference on all cropped images at once if your model supports batch inference
        predictions = ocr(cropped_images)  # Assuming this works with your inference method

        # process the result for each cropped image
        for i, prediction in enumerate(predictions['predictions']):
            obj = top_container_objects[i]  # Link predictions back to the original object
            rec_scores = np.array(prediction.get('rec_scores', []))  # Convert to NumPy array
            rec_texts = [text.upper() for text in prediction.get('rec_texts', [])] 
            det_polygons = prediction.get('det_polygons', [])  # List of detected polygons

            # Convert polygons to bounding boxes
            det_bboxes = get_bounding_boxes(det_polygons)

            # Filter using scores array for threshold
            high_score_indices = np.where(rec_scores > score_threshold)[0]

            # process logo
            predicted_id_prefix = ''
            if obj.logo_owner_code is not None:
                predicted_id_prefix = obj.logo_owner_code

            if len(high_score_indices) > 0:
                container = DetectedTU(obj.label, obj.box)
                longest_potential_id = ""
                longest_potential_id_bbox = None  # To store the corresponding bounding box
                highest_score = -1  # Initialize to a low value, assuming scores are non-negative

                for idx in high_score_indices:
                    potential_id = ""
                    score = rec_scores[idx] # Assuming 'score' is available in the bounding boxes

                    # Check if it's a valid container ID
                    if len(rec_texts[idx]) == 11 and is_valid_container_id(rec_texts[idx]):
                        potential_id = rec_texts[idx]
                    else:
                        if len(rec_texts[idx]) == 11 and rec_texts[idx][:4].isalpha() and rec_texts[idx][4:].isdigit():
                           # Check for a recognizable pattern that may indicate a potential ID
                            if(predicted_id_prefix is not None):
                                processed_text = replace_ukn(rec_texts[idx].strip())
                                if predicted_id_prefix != processed_text[:4]:
                                    temp_id = predicted_id_prefix + processed_text[4:]
                                    #print(f"predicted_id_prefix: {predicted_id_prefix}")  # Correct print 
                                    #print(f"temp_id: {temp_id}")  # Correct print 
                                    if is_valid_container_id(temp_id):  # Use temp_id directly here
                                        potential_id = temp_id
                                        #print(f"potential_id: {potential_id}")  # Correct print
                                    else:
                                        potential_id = rec_texts[idx]
                                else:
                                    #calculate check digit if still not valid
                                    potential_id = calculate_check_digit(processed_text[:10])
                            else:
                                #calculate check digit if still not valid
                                potential_id = calculate_check_digit(processed_text[:10])
                        else:
                            # Check for a recognizable pattern that may indicate a potential ID
                            if len(rec_texts[idx]) >= 4 and rec_texts[idx][:4].isalpha() and rec_texts[idx][:4].isupper():
                                potential_id = rec_texts[idx]
                            # If it's purely numeric, use the prefix
                            elif rec_texts[idx].isdigit():
                                potential_id = predicted_id_prefix + rec_texts[idx]
                            elif predicted_id_prefix is not None and len(rec_texts[idx]) > 4:
                                potential_id = replace_first_four_if_not_all_alpha(replace_ukn(rec_texts[idx].strip()), predicted_id_prefix)
                            # Handle the case where it doesn't fit any above conditions
                            else:
                                potential_id = predicted_id_prefix + replace_ukn(rec_texts[idx].strip())

                            #print(f"potential_id: {potential_id}")  # Correct print 
                            # only add check digit if the length is 10
                            if len(potential_id) == 10:
                                # Calculate the check digit
                                potential_id = calculate_check_digit(potential_id)
                    
                    # Add potential ID to the container
                    container.add_other_potential_id(rec_texts[idx], det_bboxes[idx])
                    
                    # Check if the current potential_id is the longest one found so far
                    current_length = len(potential_id)
                    
                    if current_length > len(longest_potential_id):
                        # Found a longer potential ID
                        longest_potential_id = potential_id
                        longest_potential_id_bbox = det_bboxes[idx]
                        highest_score = score  # Update highest_score
                    elif current_length == len(longest_potential_id):
                        # Found another ID with the same length, compare scores
                        if score > highest_score:
                            longest_potential_id = potential_id
                            longest_potential_id_bbox = det_bboxes[idx]
                            highest_score = score  # Update highest_score if this ID has a greater score

                # After the loop, if we found the longest potential ID, update the container
                if longest_potential_id:
                    container.update_potential_id(longest_potential_id.upper(), longest_potential_id_bbox)

                # Add the container to detected items
                detected_items.add_tu(container)

                # Print the stored information for verification
                print(f"Added id_{obj.object_id}_{obj.label}.png - {container.type} with potential ID: {container.predicted_container_id}")
            else:
                print("No detection scores exceed the threshold.")

        # Crop and save the id_text objects
        for i, obj in enumerate(id_text_objects):
            x1, y1, x2, y2 = obj.box.astype(int)  # Convert box coordinates to integers
            cropped_image = image[y1:y2, x1:x2]  # Crop the image

            # Create a filename for the cropped image
            cropped_image_filename = os.path.join(id_text_dir, f'id_{obj.object_id}_{obj.label}.png')
            
            # Save the cropped image
            cv2.imwrite(cropped_image_filename, cropped_image)

        #print(id_text_objects)

    else:
        print("No container objects detected.")

    detected_items.save_to_file(output_dir, output_result)
        
else:
    print("No predictions found for this sample.")
