class DetectedObject:
    def __init__(self, name, object_id=1, box=None, label=None, score=None, area=None, parent_ids=None, logo_owner_code=None):
        self.name = name                          # Name of the detected object
        self.object_id = object_id                # Unique identifier for the object
        self.box = box                            # Single bounding box
        self.label = label                        # Single label
        self.score = score                        # Single score
        self.area = area                          # Single area value
        self.parent_ids = parent_ids if parent_ids is not None else []  # List of parent container IDs
        self.logo_owner_code = logo_owner_code

    def add_parent_id(self, parent_id):
        """Add a parent ID to the list of parent IDs."""
        if parent_id not in self.parent_ids:
            self.parent_ids.append(parent_id)

    def add_logo(self, logo_owner_code):
        self.logo_owner_code = logo_owner_code

    def __str__(self):
        """Provide a string representation of the detected object."""
        return (f"DetectedObject(name={self.name}, logo_owner_code={self.logo_owner_code}, object_id={self.object_id}, "
                f"box={self.box}, label={self.label}, score={self.score}, "
                f"area={self.area}, parent_ids={self.parent_ids})")