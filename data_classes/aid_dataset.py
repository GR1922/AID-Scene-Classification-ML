import glob
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


class AIDDataset:
    """
    Gestione dataset AID.
    Assunzione struttura:
        dataset_dir/
            class_1/
                img1.jpg ...
            class_2/
                ...
    """

    def __init__(self, dataset_dir: str, seed: int = 42):
        self.dataset_dir = Path(dataset_dir)
        self.seed = seed

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        self.class_names = sorted([p.name for p in self.dataset_dir.iterdir() if p.is_dir()])
        self.num_classes = len(self.class_names)

        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}
        self.id_to_class = {i: name for name, i in self.class_to_id.items()}

    def load_paths_and_labels(self):
        image_paths = []
        labels = []

        for cname in self.class_names:
            folder = self.dataset_dir / cname
            files = sorted(glob.glob(str(folder / "*.jpg"))) \
                + sorted(glob.glob(str(folder / "*.jpeg"))) \
                + sorted(glob.glob(str(folder / "*.png")))

            image_paths.extend(files)
            labels.extend([self.class_to_id[cname]] * len(files))

        image_paths = np.array(image_paths)
        labels = np.array(labels)

        return image_paths, labels

    def train_val_split(self, test_size=0.2):
        X, y = self.load_paths_and_labels()

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.seed,
            stratify=y
        )
        return X_train, X_val, y_train, y_val
