import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------------
# Class definitions for each split configuration
# ------------------------------------------------------------------
SPLIT_CONFIG = {
    1: {
        'closed': ['TUM', 'STR', 'LYM', 'NORM'],
        'open':   ['cSTR', 'DEB', 'BACK']
    },
    2: {
        'closed': ['TUM', 'STR'],
        'open':   ['LYM', 'NORM', 'cSTR', 'DEB', 'BACK']
    },
    3: {
        'closed': ['TUM', 'STR', 'LYM'],
        'open':   ['NORM', 'cSTR', 'DEB', 'BACK']
    }
}

# Harmonized class names across Kather-16 and Kather-19
# Following the harmonization protocol with expert pathologists
CLASS_HARMONIZATION = {
    # Kather-16 original names -> harmonized names
    '01_TUMOR':     'TUM',
    '02_STROMA':    'STR',
    '03_COMPLEX':   'cSTR',
    '04_LYMPHO':    'LYM',
    '05_DEBRIS':    'DEB',
    '06_MUCOSA':    'NORM',
    '07_ADIPOSE':   'ADI',
    '08_EMPTY':     'BACK',
    # Kather-16 without numbers (fallback)
    'TUMOR':        'TUM',
    'STROMA':       'STR',
    'COMPLEX':      'cSTR',
    'LYMPHO':       'LYM',
    'DEBRIS':       'DEB',
    'MUCOSA':       'NORM',
    'ADIPOSE':      'ADI',
    'EMPTY':        'BACK',
    # Kather-19 original names -> harmonized names
    'ADI':          'ADI',
    'BACK':         'BACK',
    'DEB':          'DEB',
    'LYM':          'LYM',
    'MUC':          'DEB',    # mucus merged into debris
    'MUS':          'STR',    # smooth muscle merged into stroma
    'NORM':         'NORM',
    'STR':          'STR',
    'TUM':          'TUM',
}


# ------------------------------------------------------------------
# Dataset class
# ------------------------------------------------------------------
class KatherDataset(Dataset):
    """
    Dataset class for Kather-16 and Kather-19 colorectal cancer datasets.

    Supports:
    - Closed-set samples (known classes labeled 0..C-1)
    - Open-set samples (unknown classes labeled C)
    - Train/val/test splits (70/15/15 stratified)
    """

    def __init__(self, root_dir, split_config, mode='train',
                 transform=None, is_source=True):
        """
        root_dir:     path to dataset root
        split_config: dict with 'closed' and 'open' class lists
        mode:         'train', 'val', or 'test'
        transform:    image transforms
        is_source:    if True, only load closed-set classes
                      if False, load both closed and open-set classes
        """
        self.root_dir = root_dir
        self.split_config = split_config
        self.mode = mode
        self.transform = transform
        self.is_source = is_source

        self.closed_classes = split_config['closed']
        self.open_classes = split_config['open']
        self.num_closed = len(self.closed_classes)

        # class to index mapping
        self.class_to_idx = {cls: i for i, cls in
                             enumerate(self.closed_classes)}
        # open-set class index = num_closed
        self.open_idx = self.num_closed

        # load all image paths and labels
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """
        Load image paths and labels from directory structure.
        Expected structure: root_dir/CLASS_NAME/image.png
        """
        all_samples = []

        for class_folder in sorted(os.listdir(self.root_dir)):
            class_path = os.path.join(self.root_dir, class_folder)
            if not os.path.isdir(class_path):
                continue

            # harmonize class name
            harmonized = CLASS_HARMONIZATION.get(
                class_folder.upper(), class_folder.upper())

            # determine label
            if harmonized in self.closed_classes:
                label = self.class_to_idx[harmonized]
            elif harmonized in self.open_classes:
                if self.is_source:
                    continue  # skip open-set for source domain
                label = self.open_idx
            else:
                continue  # skip unknown classes

            # collect images
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    img_path = os.path.join(class_path, img_file)
                    all_samples.append((img_path, label))

        # stratified split 70/15/15
        if len(all_samples) == 0:
            self.samples = []
            return

        paths = [s[0] for s in all_samples]
        labels = [s[1] for s in all_samples]

        # first split: 70% train, 30% temp
        train_paths, temp_paths, train_labels, temp_labels = \
            train_test_split(paths, labels, test_size=0.30,
                             stratify=labels, random_state=42)

        # second split: 50% of temp = val (15%), 50% = test (15%)
        val_paths, test_paths, val_labels, test_labels = \
            train_test_split(temp_paths, temp_labels, test_size=0.50,
                             stratify=temp_labels, random_state=42)

        if self.mode == 'train':
            self.samples = list(zip(train_paths, train_labels))
        elif self.mode == 'val':
            self.samples = list(zip(val_paths, val_labels))
        elif self.mode == 'test':
            self.samples = list(zip(test_paths, test_labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ------------------------------------------------------------------
# Transforms
# ------------------------------------------------------------------
def get_transforms(img_size=224, mode='train'):
    """
    Standard transforms for training and evaluation.
    Style robustness is handled by RSAM — no color augmentation needed.
    """
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return transform


# ------------------------------------------------------------------
# DataLoader builder
# ------------------------------------------------------------------
def build_dataloaders(args):
    """
    Build source and target dataloaders for all modes.

    Returns dict with:
        source_train, source_val,
        target_train, target_val, target_test
    """
    split_config = SPLIT_CONFIG[args.split]

    train_transform = get_transforms(args.img_size, mode='train')
    eval_transform = get_transforms(args.img_size, mode='eval')

    # source dataloaders (closed-set only)
    source_train = DataLoader(
        KatherDataset(args.source_path, split_config,
                      mode='train', transform=train_transform,
                      is_source=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    source_val = DataLoader(
        KatherDataset(args.source_path, split_config,
                      mode='val', transform=eval_transform,
                      is_source=True),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # target dataloaders (closed + open-set)
    target_train = DataLoader(
        KatherDataset(args.target_path, split_config,
                      mode='train', transform=train_transform,
                      is_source=False),
        batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    target_val = DataLoader(
        KatherDataset(args.target_path, split_config,
                      mode='val', transform=eval_transform,
                      is_source=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    target_test = DataLoader(
        KatherDataset(args.target_path, split_config,
                      mode='test', transform=eval_transform,
                      is_source=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    return {
        'source_train': source_train,
        'source_val':   source_val,
        'target_train': target_train,
        'target_val':   target_val,
        'target_test':  target_test
    }


# ------------------------------------------------------------------
# Test with dummy data
# ------------------------------------------------------------------
if __name__ == '__main__':
    from Settings import get_args
    import tempfile
    import numpy as np
    from PIL import Image as PILImage

    args = get_args()

    # create dummy dataset structure for testing
    print("Creating dummy dataset for testing...")
    with tempfile.TemporaryDirectory() as tmpdir:

        split_config = SPLIT_CONFIG[args.split]
        all_classes = (split_config['closed'] +
                       split_config['open'])

        # create dummy images for each class
        for cls in all_classes:
            cls_dir = os.path.join(tmpdir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(20):
                img = PILImage.fromarray(
                    np.random.randint(0, 255, (224, 224, 3),
                                      dtype=np.uint8))
                img.save(os.path.join(cls_dir, f'img_{i}.png'))

        transform = get_transforms(args.img_size, mode='train')

        # test closed-set (source)
        source_ds = KatherDataset(
            tmpdir, split_config,
            mode='train', transform=transform, is_source=True)
        print(f"Source train samples: {len(source_ds)}")

        # test open+closed (target)
        target_ds = KatherDataset(
            tmpdir, split_config,
            mode='train', transform=transform, is_source=False)
        print(f"Target train samples: {len(target_ds)}")

        # test dataloader
        loader = DataLoader(target_ds, batch_size=8, shuffle=True)
        imgs, labels = next(iter(loader))
        print(f"Batch images shape: {imgs.shape}")
        print(f"Batch labels: {labels}")
        print(f"Unique labels: {labels.unique()}")

        print("\nDataLoader tested successfully.")
        print(f"Split {args.split} config:")
        print(f"  Closed classes: {split_config['closed']}")
        print(f"  Open classes:   {split_config['open']}")
        print(f"  Num closed (C): {len(split_config['closed'])}")