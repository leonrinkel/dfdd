import os
import json
import logging

from my_types import DatasetItem

KEY_FILES = "files"
KEY_LABEL = "label"
KEY_AGE = "age"
KEY_GENDER = "gender"
KEY_SKIN_TYPE = "skin-type"

def load_dataset(
    dataset_path: str,
    annotations_path: str,
) -> list[DatasetItem]:
    logger = logging.getLogger("dataset_loader")

    with open(annotations_path) as annotations_file:
        annotations = json.load(annotations_file)

    dataset_items: list[DatasetItem] = []
    for subject_id, annotation in annotations.items():
        assert KEY_FILES in annotation
        assert KEY_LABEL in annotation
        assert KEY_AGE in annotation[KEY_LABEL]
        assert KEY_GENDER in annotation[KEY_LABEL]
        assert KEY_SKIN_TYPE in annotation[KEY_LABEL]

        for file in annotation[KEY_FILES]:
            file_path = os.path.abspath(os.path.join(dataset_path, file))
            if not os.path.isfile(file_path):
                logger.warning(
                    "ignoring dataset item because file does not exist: %s",
                    file_path)
                continue

            dataset_items.append({
                "subject_id": subject_id,
                "file_path": file_path,
                "label": {
                    "age": annotation[KEY_LABEL][KEY_AGE],
                    "gender": annotation[KEY_LABEL][KEY_GENDER],
                    "skin_type": annotation[KEY_LABEL][KEY_SKIN_TYPE],
                },
            })

    return dataset_items
