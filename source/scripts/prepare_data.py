from pathlib import Path


def print_expected_layout():
    root = Path("datasets")
    message = f"""
Expected dataset layout:

{root / 'modelnet40_ply_hdf5_2048'}
  - ply_data_train0.h5
  - ...
  - ply_data_test0.h5
  - shape_names.txt

{root / 'scanobjectnn' / 'main_split'}
  - training_objectdataset.h5
  - test_objectdataset.h5
  - classes.txt

Download notes:
- ModelNet40 H5 files are commonly distributed with PointNet / DGCNN repositories.
- ScanObjectNN should use one consistent official split for both models.
- Keep raw datasets outside the submission package if the course instructions require it.
"""
    print(message.strip())


if __name__ == "__main__":
    print_expected_layout()
