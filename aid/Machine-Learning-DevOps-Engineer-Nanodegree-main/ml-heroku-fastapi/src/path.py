"""
Function to be used by the scripts of this
folder in order to create the paths pointing
the data folder
"""
import os


def data_folder_path(subfolder, file):
    """[summary]

    Args:
        subfolder ([str]): [subfolder name]
        file ([str]): [file name]

    Returns:
        [str]: [file complete path]
    """
    root = os.path.dirname(os.getcwd())
    return os.path.join(root, "Uda_MLOps_Project_3", "data", subfolder, file)
