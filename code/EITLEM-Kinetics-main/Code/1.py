import torch
import os

def print_train_pair_columns(dataset_path, Type):
    # Load train pair information
    file_path = os.path.join(dataset_path, Type, f"{Type}TrainPairInfo")
    if not os.path.exists(file_path):
        print(f"Error: file does not exist -> {file_path}")
        return

    train_pair = torch.load(file_path)
    print(f"Successfully loaded {Type} train pair information, data type: {type(train_pair)}\n")

    # Handle dict type (explicit column names)
    if isinstance(train_pair, dict):
        print("Included columns (keys):")
        for key in train_pair.keys():
            print(f"- {key}")
        # Print the first row as an example
        if train_pair:
            first_key = next(iter(train_pair.keys()))
            print(f"\nExample data (first element of {first_key}): {train_pair[first_key][0]}")

    # Handle list type (implicit columns)
    elif isinstance(train_pair, list):
        print(f"{len(train_pair)} entries in total; structure of each entry (implicit columns):")
        if train_pair:
            first_sample = train_pair[0]
            for i, value in enumerate(first_sample):
                print(f"- Column {i+1}: {value} (example value)")
            # Infer common scenarios
            if len(first_sample) == 3:
                print("\nInferred column meaning: protein ID -> molecule ID -> label value (kcat/Km)")

    # Other types (e.g. tensors)
    else:
        print("The data type is a tensor or another structure; printing its shape:")
        print(f"Shape: {train_pair.shape} (if 2D, the first dimension = number of samples, the second = number of columns)")

# Call the function (modify according to your actual path and type)
dataset_path = "../Data/"  # root directory of your data
Type = "KCAT"  # or "KM"
print_train_pair_columns(dataset_path, Type)
