from SFXlearner.src.generate_data import (
    slice_guitarset,
    slice_idmt_smt_guitar,
    generate_dataset_sox,
)
import os
import argparse

guitarset10_path = "dataset/"
guitarset10_path_sliced = "dataset/guitarset10_sliced/"
guitarset_path_rendered = "dataset/guitarset10_rendered/"

def create_dirs():
    os.makedirs(guitarset10_path_sliced, exist_ok=True)
    os.makedirs(guitarset_path_rendered, exist_ok=True)

def slice(duration=5):
    slice_guitarset(
    guitarset10_path, save_dir=guitarset10_path_sliced, duration=duration
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description='Your program description')
    parser.add_argument('--dataset', type=str, help='Size of dataset to train with', default="standard")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Use small dataset if specified in the arguments
    args = parse_arguments()
    if args.dataset == "small":
        guitarset10_path_sliced = "dataset/guitarset10_sliced_small/"
        guitarset_path_rendered = "dataset/guitarset10_rendered_small_test/" 
    
    # Create directories if they do not exist
    create_dirs()

    # Slice dataset if it hasn't been done
    new_path_sliced = f"{guitarset10_path_sliced}guitarset_5.0s_clean"
    if not os.path.isdir(new_path_sliced):
        print("--Slicing dataset")
        slice()

    # Augment data using if it's not been done
    if not os.path.isdir(f"{guitarset_path_rendered}gen_multiFX_03262024"): 
        print("--Slicing dataset")
        guitarset10_path_sliced = new_path_sliced
        generate_dataset_sox([guitarset10_path_sliced], guitarset_path_rendered, methods=[1, 5], valid_split=0.2)
    else:
        print("--Dataset already created")