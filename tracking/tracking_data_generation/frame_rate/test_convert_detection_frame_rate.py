import pytest
from convert_detection_frame_rate import generate_file_map

@pytest.mark.parametrize(
    "file_list, frame_reduction_factor, expected_output",
    [
        # Test basic frame reduction without any missing frames
        (
            ["clip_1.txt", "clip_2.txt", "clip_3.txt", "clip_4.txt"],
            1,
            {1: ("clip_1.txt", "clip_1.txt"),
             2: ("clip_2.txt", "clip_2.txt"),
             3: ("clip_3.txt", "clip_3.txt"),
             4: ("clip_4.txt", "clip_4.txt")}
        ),
        # Test basic reduction
        (
            ["clip_1.txt", "clip_2.txt", "clip_3.txt", "clip_4.txt"],
            2,
            {1: ("clip_1.txt", "clip_1.txt"),
             3: ("clip_3.txt", "clip_2.txt")}
        ),
        # Test file extension mismatch, should raise FileExistsError
        (
            ["clip_1.txt", "clip_2.jpg", "clip_3.txt", "clip_4.txt"],
            2,
            pytest.raises(FileExistsError)
        ),
        # Test frame number mismatch after reduction (we expect frames to reduce by 2)
        (
            ["clip_5.txt", "clip_6.txt", "clip_7.txt", "clip_8.txt"],
            2,
            {5: ("clip_5.txt", "clip_3.txt"),
             7: ("clip_7.txt", "clip_4.txt")}
        ),
        # Test with non-sequential frame numbers
        (
            ["clip_1.txt", "clip_4.txt", "clip_7.txt", "clip_8.txt"],
            3,
            {1: ("clip_1.txt", "clip_1.txt"),
             4: ("clip_4.txt", "clip_2.txt"),
             7: ("clip_7.txt", "clip_3.txt")}
        ),
        # Test with files having frames out of order
        (
            ["clip_2.txt", "clip_1.txt", "clip_3.txt", "clip_4.txt"],
            1,
            {1: ("clip_1.txt", "clip_1.txt"),
             2: ("clip_2.txt", "clip_2.txt"),
             3: ("clip_3.txt", "clip_3.txt"),
             4: ("clip_4.txt", "clip_4.txt")}
        ),
    ]
)
def test_generate_file_map(file_list, frame_reduction_factor, expected_output):
    if isinstance(expected_output, dict):
        file_map = generate_file_map(file_list, frame_reduction_factor)
        # Assert that the generated file map matches the expected output
        assert file_map == expected_output
    else:
        # If the expected output is an exception, assert it raises the expected error
        with expected_output:
            generate_file_map(file_list, frame_reduction_factor)
