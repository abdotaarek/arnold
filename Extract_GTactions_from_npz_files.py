import numpy as np

def extract_actions_from_npz(npz_path):
    # Load the npz file
    npz_file = np.load(npz_path, allow_pickle=True)

    # Access the ground truth data
    gt_data = npz_file['gt']

    # Create a list to store the formatted step data
    formatted_data = []

    # Iterate over each step in the ground truth data
    for step in gt_data:
        # Extract the specified fields from each step
        position = step['position_rotation_world'][0]
        rotation = step['position_rotation_world'][1]
        gripper_open = step['gripper_open']

        
        formatted_data.append({'position': position, 'rotation': rotation, 'gripper_open': gripper_open})

   
    print(formatted_data)


npz_path = '/media/local/atarek/arnold/data_root/open_drawer/test/Steven-open_drawer-0-0-0.0-0.5-2-Mon_Jan_30_06:06:37_2023.npz'  # Replace with your NPZ file path
extract_actions_from_npz(npz_path)

