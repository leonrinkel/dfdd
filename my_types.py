from typing import Type, TypedDict

class DatasetLabel(TypedDict):
    age: str
    gender: str
    skin_type: str

class DatasetItem(TypedDict):
    subject_id: str
    file_path: str
    label: DatasetLabel

class WarpTask(TypedDict):
    input_file_path: str
    output_path: str
    number_of_frames: int
    relative_padding: float
    log_file_name: str
    predictions_file_name: str
