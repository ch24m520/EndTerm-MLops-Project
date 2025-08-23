import os

def test_params_yaml_exists():
    assert os.path.exists("params.yaml")

def test_model_folder_present_after_train():
    # This only asserts if you've already trained
    if os.path.exists("models/best"):
        assert os.path.isdir("models/best")
