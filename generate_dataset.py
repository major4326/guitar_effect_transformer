from SFXlearner.src.generate_data import (
    slice_guitarset,
    slice_idmt_smt_guitar,
    generate_dataset_sox,
)
import os
import argparse

# Define paths to sliced and rendered datasets
dataset_path = "dataset/"
guitarset10_path_sliced = "dataset/guitarset10_sliced/"
guitarset_path_rendered = "dataset/guitarset10_rendered/"
idmt_smt_path_sliced = "dataset/idmt_smt_sliced/"
idmt_smt_path_rendered = "dataset/idmt_smt_rendered/"

def create_dirs(path_sliced, path_rendered):
    os.makedirs(path_sliced, exist_ok=True)
    os.makedirs(path_rendered, exist_ok=True)
    
def create_dataset(path_sliced, path_rendered, dataset, duration=5):
    """
    Slice and generate dataset using sox
    """
    global dataset_path

    # Create directories if they do not exist
    create_dirs(path_sliced, path_rendered)

    # Slice dataset if it hasn't been done
    if dataset == "guitarset":
        new_path_sliced = f"{path_sliced}guitarset_5.0s_clean"
        valid_split = 0.2
    else:
        new_path_sliced = f"{path_sliced}IDMT-SMT-GUITAR_5s"
        valid_split = 1
        dataset_path = f"{dataset_path}IDMT-SMT-GUITAR_V2/dataset4"

    if not os.path.isdir(new_path_sliced):
        print("--Slicing dataset")
        slice_guitarset(
            dataset_path, save_dir=path_sliced, duration=duration
        )

    # Augment data using if it's not been done
    if not os.path.isdir(f"{path_rendered}gen_multiFX"): 
        print("--Generating the dataset")
        path_sliced = new_path_sliced
        generate_dataset_sox([path_sliced], path_rendered, methods=[1, 5], valid_split=valid_split)
    else:
        print("--Dataset already created")

def parse_arguments():
    """
    Parse terminal arguments

    Arguments
    ----------
    dataset: Which dataset to generate ('guitarset' or 'idmt_smt')
    size: Size of the dataset to generate ('standard' or 'small')
    """
    parser = argparse.ArgumentParser(description='Your program description')
    parser.add_argument('--dataset', type=str, help='Which dataset to train on', default="guitarset")
    parser.add_argument('--size', type=str, help='Size of dataset to train with', default="standard")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Use small dataset if specified in the arguments
    args = parse_arguments()
    if args.size == "small":
        guitarset10_path_sliced = "dataset/guitarset10_sliced_small/"
        guitarset_path_rendered = "dataset/guitarset10_rendered_small_test/" 
        idmt_smt_path_sliced = "dataset/idmt_smt_sliced_small/"
        idmt_smt_path_rendered = "dataset/idmt_smt_rendered_small_test/" 

    # Defined sliced and rendered path
    if args.dataset == "guitarset":
        path_sliced = guitarset10_path_sliced
        path_rendered = guitarset_path_rendered
    else: 
        path_sliced = idmt_smt_path_sliced
        path_rendered = idmt_smt_path_rendered

    # Generate dataset
    create_dataset(path_sliced, path_rendered, args.dataset)