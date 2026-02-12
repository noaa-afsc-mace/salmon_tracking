import pytest
import os
import shutil
from mot_split_test_train import do_it

def build_dest_dir(dest_dir):
    """
    Create the dest directory
    """

    # Create the dest directory at the start of each test
    if os.path.exists(dest_dir):
        # Clean the directory before the test
        for filename in os.listdir(dest_dir):
            file_path = os.path.join(dest_dir, filename)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
    else:
        os.makedirs(dest_dir)

def breakdown_dest_dir(dest_dir):
    """
    Deletes dest directory
    """

    # Cleanup the dest directory after each test
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

@pytest.fixture
def setup_existing_data(request):
    """
    Fixture to point to an existing MOT data directory for testing.
    """
    mot_annotations_dir = "tracking/tracking_data_generation/merge_split/tests/test_data/mot_train_test/source"
    save_dir = "tracking/tracking_data_generation/merge_split/tests/test_data/mot_train_test/dest"
    train_test_split_csv = "tracking/tracking_data_generation/merge_split/tests/test_data/testing_split.csv"

    return {
        "train_test_split_csv": train_test_split_csv,
        "save_dir": save_dir,
        "mot_annotations_dir": mot_annotations_dir,
        "exclude_empty": request.param
    }

@pytest.mark.parametrize("setup_existing_data", [
    True,  
    False
], indirect=True)
def test_directory_contents(setup_existing_data):
    """
    Tests that no duplicate files exist in either directory
    """

    train_test_split_csv = setup_existing_data["train_test_split_csv"]
    save_dir = setup_existing_data["save_dir"]
    mot_annotations_dir = setup_existing_data["mot_annotations_dir"]
    exclude_empty = setup_existing_data['exclude_empty']

    build_dest_dir(save_dir)

    # run split
    do_it(mot_annotations_dir, save_dir, train_test_split_csv, exclude_empty)

    if exclude_empty:
        assert os.listdir(os.path.join(save_dir, "test")) == []
        assert os.listdir(os.path.join(save_dir, "train")) == ['2019_t17_vid13_0_1799']
    else:
        assert os.listdir(os.path.join(save_dir, "test")) == ['2019_t15_vid6_1740_1799']
        assert os.listdir(os.path.join(save_dir, "train")) == ['2019_t17_vid13_0_1799']
    
    breakdown_dest_dir(save_dir)






    
    