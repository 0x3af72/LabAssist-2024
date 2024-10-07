import os
import random
import csv

def prepare(
    train_f, test_f, valid_f,
    train_p, test_p, valid_p,
    base_path, datasets,
    filtered=None
):
    all_files = []

    # Collect all files from the given datasets
    for i, row in enumerate(datasets):
        dataset, label = row
        files = os.listdir(dataset)
        files = [(f"{base_path}/{label}/{f}", i) for f in files if (not filtered in f if filtered else True)]
        all_files += files

    # Shuffle the files
    random.shuffle(all_files)

    # Calculate split indices
    total_files = len(all_files)
    train_end = int(total_files * train_p)
    test_end = train_end + int(total_files * test_p)

    # Split the files
    train_files = all_files[:train_end]
    test_files = all_files[train_end:test_end]
    valid_files = all_files[test_end:]

    # Function to write files to a CSV
    def write_to_csv(file_list, file_name):
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=" ")
            for file in file_list:
                writer.writerow(file)

    # Write the file lists to the respective CSV files
    write_to_csv(train_files, train_f)
    write_to_csv(test_files, test_f)
    write_to_csv(valid_files, valid_f)

if __name__ == "__main__":
    prepare(
        "train.csv", "test.csv", "valid.csv",
        0.7, 0.15, 0.15,
        "datasets",
        [
            # ("/datasets/buretting", "buretting"),
            # ("/datasets/pipetting", "pipetting"),
            # ("/datasets/swirling", "swirling"),
            # ("/datasets/standard", "standard")
            ("datasets/tapping_vf", "tapping_vf"),
            ("datasets/vf_shaking_correct", "vf_shaking_correct"),
            ("datasets/vf_shaking_wrong", "vf_shaking_wrong")
        ],
        "MOV"
    )
