import pytest
from convert_gt import process_gt_data

@pytest.mark.parametrize(
    "gt_input, frame_reduction_factor, expected_output",
    [
        # Test basic frame reduction with distinct bboxes
        (
            ["1,1,0,0,10,10,0,0", "2,1,0,0,20,20,0,0", "3,1,0,0,30,30,0,0"],
            1,
            ["1,1,0,0,10,10,0,0", "2,1,0,0,20,20,0,0", "3,1,0,0,30,30,0,0"]
        ),
        # Test basic reduction
        (
            ["1,1,0,0,10,10,0,0", "2,1,0,0,20,20,0,0", "3,1,0,0,30,30,0,0"],
            2,
            ["1,1,0,0,10,10,0,0", "2,1,0,0,30,30,0,0"]
        ),
        
        # Test removing short tracks (less than 2 frames) with distinct bboxes
        (
            ["1,1,0,0,10,10,0,0", "2,2,0,0,20,20,0,0"],
            2,
            []  # Track 2 is removed because it has only 1 frame
        ),
        
        # Test multiple tracks with some removed with distinct bboxes
        (
            ["1,1,0,0,10,10,0,0", "2,1,0,0,20,20,0,0", "3,2,0,0,30,30,0,0", "3,1,0,0,40,30,0,0"],
            2,
            ["1,1,0,0,10,10,0,0", "2,1,0,0,40,30,0,0"]  # Track 2 removed (only 1 frame left after reduction)
        ),

        # Test large frame numbers with reduction and distinct bboxes
        (
            ["2,1,0,0,50,50,0,0", "3,1,0,0,100,100,0,0", "4,1,0,0,150,150,0,0", "6,1,0,0,150,151,0,0", "7,1,0,0,152,150,0,0", "8,1,0,0,153,150,0,0"],
            3,
            ["2,1,0,0,150,150,0,0", "3,1,0,0,152,150,0,0"]
        ),

        # Complex example with multiple annotations, tracks, and distinct bboxes
        # Test large frame numbers with reduction and distinct bboxes
        (
            ["1,1,0,0,50,52,0,0", "1,2,0,0,50,52,0,0", "2,1,0,0,50,50,0,0", "2,2,0,0,50,50,0,0", 
             "3,1,0,0,100,100,0,0", "4,1,0,0,150,150,0,0", "5,2,0,0,150,151,0,0", "5,3,0,0,150,151,0,0",
             "6,1,0,0,150,151,0,0", "7,1,0,0,152,150,0,0", "7,3,0,0,152,150,0,0", 
             "8,1,0,0,152,150,0,0", "8,3,0,0,152,150,0,0", "9,1,0,0,156,150,0,0", "9,3,0,0,157,150,0,0"],
            4,
            ["1,1,0,0,50,52,0,0", "1,2,0,0,50,52,0,0", "2,2,0,0,150,151,0,0", "2,3,0,0,150,151,0,0",
            "3,1,0,0,156,150,0,0", "3,3,0,0,157,150,0,0"]
        ),
    ]
)
def test_process_gt_data(gt_input, frame_reduction_factor, expected_output):
    # Mock file reading by writing test data to a temp file

    # Run function
    output = process_gt_data(gt_input, frame_reduction_factor)

    # Assert results
    assert output == expected_output
