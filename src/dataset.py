import os
from torch.utils.data import Dataset
from PIL import Image

# === Loader interface ===

class BaseLoader:
    def load(self, root_dir, split, label_map):
        """Should return a list of (image_path, label) tuples."""
        raise NotImplementedError

# === Loader for flat structure ===
class FlatFolderLoader(BaseLoader):
    def load(self, root_dir, split='Train', label_map=None):
        image_list = []
        data_dir = os.path.join(root_dir, split)

        for fname in os.listdir(data_dir):
            if not fname.endswith(".png"):
                continue
            class_str = fname.split(' ')[0]
            label = label_map.get(class_str, -1)
            if label != -1:
                path = os.path.join(data_dir, fname)
                image_list.append((path, label))
        return image_list

# === Loader for nested structure ===
class NestedFolderLoader(BaseLoader):
    def load(self, root_dir, split=None, label_map=None):
        image_list = []
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = label_map.get(class_name, -1)
            if label == -1:
                continue
            for fname in os.listdir(class_dir):
                if fname.endswith(".png"):
                    path = os.path.join(class_dir, fname)
                    image_list.append((path, label))
        return image_list 

class CustomDataset(Dataset):
    def __init__(self, label_map=None, transform=None):
        self.transform = transform
        self.image_list = []

        # Use provided label_map or default one
        self.label_map = label_map or {
            'good': 0,
            'cut': 1,
            'color': 2,
            'fold': 3,
            'glue': 4,
            'poke': 5
        }

        # Create reverse mapping for optional reference
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}

    def add_data(self, loader, root_dir, split='Train', filter_label_names=None):
        """
        Adds data from the loader, optionally filtering out items with specified label names.

        Args:
            loader: Data loader with a load method.
            root_dir: Root directory for the dataset.
            split: Dataset split (e.g., 'Train', 'Test').
            filter_label_names: Optional list of label names to exclude (e.g., ['good', 'cut']).
        """
        new_data = loader.load(root_dir, split=split, label_map=self.label_map)

        if filter_label_names:
            # Convert label names to label values
            filter_label_values = {
                self.label_map[name] for name in filter_label_names if name in self.label_map
            }
            new_data = [item for item in new_data if item[1] not in filter_label_values]

        self.image_list.extend(new_data)
        self._rebuild_index()

    def _rebuild_index(self):
        self.labels = [label for _, label in self.image_list]
        self.label_set = set(self.labels)
        self.label_to_indices = {
            label: [i for i, l in enumerate(self.labels) if l == label]
            for label in self.label_set
        }

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path, label = self.image_list[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
    
