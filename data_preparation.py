from PIL import Image
from pathlib import Path, PurePath
from tqdm import tqdm
import os
import shutil
import numpy as np
import pandas as pd
import zipfile


def unzip_file(path):
    print("Unzipping...")
    file = zipfile.ZipFile("Pest Images.zip")
    file.extractall(path)
    print("Unzipping done.\n")


def check_dimensions(path):
    container = []

    for directory in path.rglob("*"):
        if "ipynb" in directory.as_posix():
            continue
        if directory.is_file():
            class_ = {}
            class_["name"] = PurePath(directory.parent).parts[-1]
            class_["path"] = directory
            size = Image.open(directory).size
            class_["width"] = size[0]
            class_["height"] = size[1]
            container.append(class_)

    df = pd.DataFrame(container)
    print("Each class' dimension")
    print(df.groupby("name").describe())
    print("")

    return df


def split_data(
    dataframe,
    seed=None,
    train_frac=0.7,
    val_frac=0.3,
    test_frac=0.1,
):
    np.random.seed(seed)
    MASK_INDEX = dataframe.groupby("name").sample(frac=1 - train_frac).index
    train = (
        dataframe.iloc[dataframe.index.difference(MASK_INDEX)]
        .copy()
        .reset_index(drop=True)
    )

    val_test = dataframe.iloc[MASK_INDEX].copy().reset_index(drop=True)

    TEST_MASK = val_test.groupby("name").sample(frac=test_frac / val_frac).index
    val = (
        val_test.iloc[val_test.index.difference(TEST_MASK)]
        .copy()
        .reset_index(drop=True)
    )
    test = val_test.iloc[TEST_MASK].copy().reset_index(drop=True)
    return train, test, val


def copy_to_target(source_path, target_path, train_data, test_data, val_data):
    print("Copying files...")
    for path, target in (
        (train_data.path, "train"),
        (test_data.path, "test"),
        (val_data.path, "val"),
    ):

        # Split Type folder
        split_type = target_path / target
        print()

        for source in tqdm(path):
            # Class folder
            class_folder = split_type / source.relative_to(source_path).parent
            class_folder.mkdir(exist_ok=True, parents=True)
            shutil.copy(source, class_folder)


def main():
    print("Process started...")
    shutil.rmtree("Data", ignore_errors=True)
    CWD = Path()
    RAW_DATA_PATH = CWD / "Data" / "Raw Data"
    SPLIT_DATA = CWD / "Data" / "Split Data"
    RAW_DATA_PATH.mkdir(exist_ok=True, parents=True)
    SPLIT_DATA.mkdir(exist_ok=True, parents=True)

    unzip_file(RAW_DATA_PATH)

    df = check_dimensions(RAW_DATA_PATH)
    train, test, val = split_data(
        dataframe=df, seed=0, train_frac=0.7, val_frac=0.3, test_frac=0.1
    )

    # Train count
    print("Train Count")
    print(train.name.value_counts())
    print("")

    # Validation count
    print("Validation Count")
    print(val.name.value_counts())
    print("")

    # Test count
    print("Test Count")
    print(test.name.value_counts())
    print("")

    copy_to_target(
        source_path=RAW_DATA_PATH,
        target_path=SPLIT_DATA,
        train_data=train,
        test_data=test,
        val_data=val,
    )
    print("Data split done successfully.")


if __name__ == "__main__":
    main()
