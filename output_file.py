import os
import json
import numpy as np
import datetime

class SpottedID:
    def __init__(self, recognised_text, text_bbox):
        self.recognised_text = recognised_text
        self.text_bbox = text_bbox

    def to_dict(self):
        return {
            "recognised_text": self.recognised_text,
            "text_bbox": self.text_bbox
        }
    

class DetectedTU:
    def __init__(self, tu_type, bbox, container_id=None, text_bbox=None):
        self.type = tu_type  # 'container' or 'trailer'
        self.bbox = bbox  # [min_x, min_y, max_x, max_y]
        
        # Store as a dictionary with keys
        self.predicted_container_id = {
            "tu_id": container_id,  # This will hold the valid container ID
            "text_bbox": text_bbox  # This is the text bounding box
        } if container_id else None
        
        self.other_potential_ids = []  # List to hold other potential IDs

    def add_other_potential_id(self, recognised_text, text_bbox):
        """Add an additional potential ID."""
        self.other_potential_ids.append(SpottedID(recognised_text, text_bbox))

    def update_potential_id(self, container_id, text_bbox):
        """Update the container ID and its bounding box."""
        self.predicted_container_id = {
            "tu_id": container_id,
            "text_bbox": text_bbox
        }

    def to_dict(self):
        # Prepare to return the structure as required
        return {
            "type": self.type,
            "bbox": self.bbox.tolist() if isinstance(self.bbox, np.ndarray) else self.bbox,  # Convert to list if ndarray
            "predicted_tu_id": self.predicted_container_id,
            "potential_ids": [id.to_dict() for id in self.other_potential_ids]
        }
    
class DetectedItems:
    def __init__(self):
        self.detected_tus = []

    def add_tu(self, detected_tu):
        """Add a detected TU to the internal list."""
        self.detected_tus.append(detected_tu)

    def generate_timestamped_filename(self, base_name, extension='json'):
        """Generate a timestamped filename in the format base_name_YYYYMMDD_HHMMSS.extension."""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # Format timestamp
        return f"{base_name}_{timestamp}.{extension}"  # Create a new filename

    def save_to_file(self, folder_path, file_name):
        """Save detected items to a JSON file."""
        # Ensure the output directory exists
        os.makedirs(folder_path, exist_ok=True)

        # Generate a timestamped filename
        timestamped_filename = self.generate_timestamped_filename(file_name)  
        file_path = os.path.join(folder_path, timestamped_filename)  # Construct the full file path

        # Save the JSON data to a file
        with open(file_path, 'w') as json_file:
            # Ensure everything is serializable
            json.dump({"detected_tus": [tu.to_dict() for tu in self.detected_tus]}, json_file, indent=4)
        
        print(f"Data successfully saved to {file_path}")