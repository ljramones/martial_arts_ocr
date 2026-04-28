from martial_arts_ocr.config import allowed_file, get_config, get_processed_path, get_upload_path


def test_testing_config_loads():
    config = get_config("testing")

    assert config.TESTING is True
    assert "sqlite:///:memory:" in config.DATABASE_URL


def test_paths_are_under_data_directory():
    upload_path = get_upload_path("sample.png")
    processed_path = get_processed_path("sample.json")

    assert any(parent.name == "data" for parent in upload_path.parents)
    assert any(parent.name == "data" for parent in processed_path.parents)
    assert allowed_file("scan.tiff")
    assert not allowed_file("notes.txt")
