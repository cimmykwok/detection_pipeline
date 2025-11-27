# config.py
settings = {
    # demo image
    "input_image": 'GX010007_done_10032.jpg', 
    # config files for models
    "config_file": 'model/rtmdet_s/config.py',
    "checkpoint_file": 'model/rtmdet_s/epoch_444.pth',
    "text_det_config": 'model/dbnet_resnet18/config.py',
    "text_det_weight": "model/dbnet_resnet18/epoch_100.pth",
    "text_rec_config": 'model/svtr_s/config.py',
    "text_rec_weight": 'model/svtr_s/epoch_360.pth',
    # Output configuration
    "output_dir": 'output',  # Base output directory
    "container_dir": 'container',  # Subdirectory for container outputs
    "id_text_dir": 'id_text',  # Subdirectory for ID text outputs
    "ouput_result_filename": 'output_result',  # Base name for output result files
    # Valid labels for filtering
    "valid_container_labels": ['tank_container', 'semi_trailer', 'container', 'trailer'],
    "valid_id_text_labels": ['id_text'],
    "valid_logo_mapping": 'logo_mapping.json',
    # Processing parameters
    "score_threshold": 0.5,  # Minimum score for valid detections (will affect the performance....)
    "largest_n_containers_no": 15, # Number of largest containers to retrieve
    "area_threshold": 10000
}