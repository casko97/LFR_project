"""
IMPLEMENTING DATALOADER FOR RSICD DATASET
- FOR FULL-SUPERVISED FINE-TUNE: TAKE IMAGE FROM "VAL" THEN SPLIT TO "REAL" TRAIN AND VAL
- FOR SEMI-SUPERVISED FINE-TUNE: TAKE IMAGE FROM "TRAIN" BUT NO CAPTION + "REAL" TRAIN SET FOR TRAINING
- FOR TRAINING MODE: RETURN PAIR OF IMAGE-TEXT
- FOR TESTING MODE: RETURN IMAGE AND IT'S FILENAME (FOR SEARCHING CORRECT CLASS)
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
from PIL import Image 
import numpy as np
import torch
from torchvision import transforms
import os
from utils import _search_class, tokenize_texts, check_duplicates
from prompt_utils import display_class_name


CAPTION_POLICIES = frozenset({"all_unique", "random_one"})


def _validate_rsicd_paths(
    image_dir: str | Path, json_path: str | Path
) -> tuple[Path, Path]:
    """Resolve and validate the common image-directory and JSON inputs."""
    image_dir = Path(image_dir)
    json_path = Path(json_path)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"RSICD image directory does not exist: {image_dir}")
    if not json_path.is_file():
        raise FileNotFoundError(f"RSICD metadata file does not exist: {json_path}")
    return image_dir, json_path


def _load_rsicd_split(json_path: Path, split: str) -> list[dict]:
    """Load one JSON split while preserving its declared record order."""
    with json_path.open(encoding="utf-8") as file:
        try:
            images = json.load(file)["images"]
        except (json.JSONDecodeError, KeyError) as error:
            raise ValueError(f"Invalid RSICD metadata file: {json_path}") from error
    return [item for item in images if item.get("split") == split]


def _load_transformed_image(image_dir: Path, filename: str, transform: Callable):
    """Open an RSICD image as RGB and apply caller-supplied preprocessing."""
    with Image.open(image_dir / filename) as image:
        return transform(image.convert("RGB"))


def _validated_record_filename(
    image_dir: Path,
    record: Mapping[str, object],
    dataset_name: str,
) -> str:
    """Validate and return the filename shared by RSICD dataset variants."""
    filename = record.get("filename")
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"{dataset_name} record has no valid filename: {record!r}")
    image_path = image_dir / filename
    if not image_path.is_file():
        raise FileNotFoundError(f"RSICD image does not exist: {image_path}")
    return filename


def _validated_text_tokens(tokenizer: Callable, texts: Sequence[str]) -> torch.Tensor:
    """Tokenize text rows and enforce the common OpenCLIP tensor contract."""
    tokens = tokenizer(list(texts))
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
        raise TypeError(
            "Tokenizer must return a [num_texts, context_length] torch.Tensor"
        )
    return tokens


class RSICDZeroShotDataset(Dataset):
    """A configured RSICD split for zero-shot classification.

    Labels are loaded once from the class-list files, rather than searched for
    each image access. ``transform`` is supplied by the caller so the dataset
    always uses the preprocessing required by the selected OpenCLIP model.
    """

    def __init__(
        self,
        image_dir: str | Path,
        json_path: str | Path,
        class_dir: str | Path,
        transform: Callable,
        split: str,
    ) -> None:
        self.image_dir, self.json_path = _validate_rsicd_paths(image_dir, json_path)
        self.class_dir = Path(class_dir)
        self.transform = transform

        if not self.class_dir.is_dir():
            raise FileNotFoundError(f"RSICD class directory does not exist: {self.class_dir}")

        self.class_names = [path.stem for path in sorted(self.class_dir.iterdir()) if path.is_file()]
        if not self.class_names:
            raise ValueError(f"No class-list files found in: {self.class_dir}")

        self.filename_to_class = self._build_filename_to_class()
        self.images_info = _load_rsicd_split(self.json_path, split)
        if not self.images_info:
            raise ValueError(f"No {split!r}-split images found in: {self.json_path}")

        self._validate_test_images()

    def _build_filename_to_class(self) -> dict[str, str]:
        filename_to_class: dict[str, str] = {}
        for class_file in sorted(path for path in self.class_dir.iterdir() if path.is_file()):
            class_name = class_file.stem
            for raw_filename in class_file.read_text(encoding="utf-8").splitlines():
                filename = raw_filename.strip()
                if not filename:
                    continue
                previous_class = filename_to_class.setdefault(filename, class_name)
                if previous_class != class_name:
                    raise ValueError(
                        f"Filename {filename!r} appears in multiple class lists: "
                        f"{previous_class!r} and {class_name!r}"
                    )
        return filename_to_class

    def _validate_test_images(self) -> None:
        missing_labels = []
        missing_images = []
        for item in self.images_info:
            filename = item.get("filename")
            if not filename:
                raise ValueError(f"Test metadata entry has no filename: {item!r}")
            if filename not in self.filename_to_class:
                missing_labels.append(filename)
            if not (self.image_dir / filename).is_file():
                missing_images.append(filename)

        if missing_labels or missing_images:
            details = []
            if missing_labels:
                details.append(f"missing labels ({len(missing_labels)}): {missing_labels[:3]}")
            if missing_images:
                details.append(f"missing images ({len(missing_images)}): {missing_images[:3]}")
            raise ValueError("Invalid RSICD test resources; " + "; ".join(details))

    def __len__(self) -> int:
        return len(self.images_info)

    def __getitem__(self, idx: int):
        filename = self.images_info[idx]["filename"]
        image_tensor = _load_transformed_image(self.image_dir, filename, self.transform)
        return image_tensor, self.filename_to_class[filename], filename


class RSICDImageOnlyDataset(Dataset):
    """Load a JSON split strictly as images and never expose its captions."""

    def __init__(
        self,
        image_dir: str | Path,
        json_path: str | Path,
        transform: Callable,
        split: str,
        max_samples: int | None = None,
    ) -> None:
        self.image_dir, self.json_path = _validate_rsicd_paths(image_dir, json_path)
        self.transform = transform
        records = _load_rsicd_split(self.json_path, split)
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            records = records[:max_samples]
        if not records:
            raise ValueError(f"No {split!r}-split images found in: {self.json_path}")

        self.filenames = [
            _validated_record_filename(self.image_dir, record, "RSICD")
            for record in records
        ]

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        filename = self.filenames[index]
        image = _load_transformed_image(self.image_dir, filename, self.transform)
        return image, filename


class RSICDPseudoPairDataset(Dataset):
    """Materialize selected pseudo image/class pairs as CLIP prompt pairs."""

    def __init__(
        self,
        image_dir: str | Path,
        pseudo_pairs: Sequence[Mapping[str, object]],
        transform: Callable,
        tokenizer: Callable,
        prompt_templates: Sequence[str],
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.records = list(pseudo_pairs)
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"RSICD image directory does not exist: {self.image_dir}")
        if not self.records:
            raise ValueError("Pseudo-pair dataset has no records")
        if not prompt_templates:
            raise ValueError("At least one pseudo prompt template is required")
        if any(template.count("{}") != 1 for template in prompt_templates):
            raise ValueError("Every pseudo prompt template must contain exactly one '{}' placeholder")

        self.filenames: list[str] = []
        self.caption_tokens: list[torch.Tensor] = []
        for record in self.records:
            filename = _validated_record_filename(
                self.image_dir, record, "Pseudo-pair"
            )
            class_name = record.get("class_name")
            if not isinstance(class_name, str):
                raise ValueError("Pseudo records require a string class_name value")
            display_name = display_class_name(class_name)
            tokens = _validated_text_tokens(
                tokenizer,
                [template.format(display_name) for template in prompt_templates],
            )
            self.filenames.append(filename)
            self.caption_tokens.append(tokens)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        filename = self.filenames[index]
        image = _load_transformed_image(self.image_dir, filename, self.transform)
        return image, self.caption_tokens[index], filename, "pseudo"


class RSICDSupervisedDataset(Dataset):
    """Captioned RSICD records for OpenCLIP supervised fine-tuning."""

    def __init__(
        self,
        image_dir: str | Path,
        records: list[dict],
        transform: Callable,
        tokenizer: Callable,
        caption_policy: str = "all_unique",
        sampling_seed: int = 0,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.tokenizer = tokenizer
        self.records = records
        self.caption_policy = caption_policy
        self.sampling_seed = sampling_seed
        self.epoch = 0

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"RSICD image directory does not exist: {self.image_dir}")
        if not self.records:
            raise ValueError("Supervised RSICD dataset has no records")
        if self.caption_policy not in CAPTION_POLICIES:
            choices = ", ".join(sorted(CAPTION_POLICIES))
            raise ValueError(
                f"Unsupported caption_policy {self.caption_policy!r}; "
                f"choose from: {choices}"
            )

        self.filenames: list[str] = []
        self.caption_tokens: list[torch.Tensor] = []
        for record in self.records:
            filename = _validated_record_filename(self.image_dir, record, "RSICD")

            captions = list(
                dict.fromkeys(
                    " ".join(sentence["tokens"])
                    for sentence in record.get("sentences", [])
                    if sentence.get("tokens")
                )
            )
            if not captions:
                raise ValueError(f"RSICD image has no captions: {filename}")

            tokens = _validated_text_tokens(self.tokenizer, captions)
            self.filenames.append(filename)
            self.caption_tokens.append(tokens)

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic random-caption schedule for one epoch."""
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = epoch

    def selected_caption_tokens(self, idx: int) -> torch.Tensor:
        """Return all captions or one stateless, seed-controlled sample."""
        tokens = self.caption_tokens[idx]
        if self.caption_policy == "all_unique":
            return tokens

        sample_seed = self.sampling_seed + self.epoch * len(self) + idx
        generator = torch.Generator().manual_seed(sample_seed)
        caption_index = torch.randint(
            tokens.shape[0], (1,), generator=generator
        ).item()
        return tokens[caption_index:caption_index + 1]

    def __getitem__(self, idx: int):
        image_tensor = _load_transformed_image(
            self.image_dir, self.filenames[idx], self.transform
        )
        return image_tensor, self.selected_caption_tokens(idx), self.filenames[idx]


def build_supervised_train_val_datasets(
    image_dir: str | Path,
    json_path: str | Path,
    train_transform: Callable,
    validation_transform: Callable,
    tokenizer: Callable,
    train_fraction: float,
    seed: int,
    source_split: str,
    train_caption_policy: str = "all_unique",
) -> tuple[RSICDSupervisedDataset, RSICDSupervisedDataset]:
    """Split one captioned JSON source split for fine-tuning."""
    image_dir, json_path = _validate_rsicd_paths(image_dir, json_path)
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")

    records = _load_rsicd_split(json_path, source_split)
    if len(records) < 2:
        raise ValueError(f"RSICD JSON {source_split!r} split must contain at least two records")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(records), generator=generator).tolist()
    train_size = int(len(records) * train_fraction)
    train_records = [records[index] for index in indices[:train_size]]
    validation_records = [records[index] for index in indices[train_size:]]

    return (
        RSICDSupervisedDataset(
            image_dir,
            train_records,
            train_transform,
            tokenizer,
            caption_policy=train_caption_policy,
            sampling_seed=seed,
        ),
        RSICDSupervisedDataset(
            image_dir,
            validation_records,
            validation_transform,
            tokenizer,
            caption_policy="all_unique",
            sampling_seed=seed,
        ),
    )


class SourceTaggedDataset(Dataset):
    """Attach a supervision-source label to a caption dataset."""

    def __init__(self, dataset: Dataset, source: str) -> None:
        self.dataset = dataset
        self.source = source

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, captions, filename = self.dataset[index]
        return image, captions, filename, self.source

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.dataset, "set_epoch", None)
        if setter is not None:
            setter(epoch)


def set_dataset_epoch(dataset: Dataset, epoch: int) -> None:
    """Propagate an epoch into tagged/concatenated caption datasets."""
    setter = getattr(dataset, "set_epoch", None)
    if setter is not None:
        setter(epoch)
    children = getattr(dataset, "datasets", None)
    if children is not None:
        for child in children:
            set_dataset_epoch(child, epoch)


def collate_supervised_captions(batch):
    """Collate image-text pairs, preserving optional supervision sources."""
    if not batch:
        raise ValueError("Cannot collate an empty image-text batch")
    if any(len(sample) not in {3, 4} for sample in batch):
        raise ValueError(
            "Samples must contain image, tokens, filename, and optional source"
        )
    normalized = [
        sample if len(sample) == 4 else (*sample, "caption") for sample in batch
    ]
    images, caption_batches, filenames, sources = zip(*normalized)
    caption_to_image = torch.cat(
        [
            torch.full((captions.shape[0],), image_index, dtype=torch.long)
            for image_index, captions in enumerate(caption_batches)
        ]
    )
    return {
        "images": torch.stack(images),
        "texts": torch.cat(caption_batches),
        "caption_to_image": caption_to_image,
        "filenames": list(filenames),
        "sources": list(sources),
    }

#========= DATALOADER CLASS ========#
class rsicd_loader(Dataset):
    """ Load image and caption from RSICD data.
    Depending on training mode, either single or multiple captions will be loaded
    Depending on which phase [train or test], different split will be loaded
    Mode = train:  load from Split B and return (image, caption)
    Mode = test:   load from Split C and ONLY return image
    
    Args:
        image_dir   : path to the image directory
        json_dataset: .json file contains list of images and captions
        multi_cap   : option to choose single or multiple captions
        fully_sup   : option for fully-supervised or NOT
        mode       : ['train' or 'test']
    Return:
        if mode = train: image (Tensor), caption
        if mode = test: image, image's filename
    """
    
    def __init__(self, image_dir, json_path, text_class, split, num_captions = None, tokenizer = None):
        self.image_dir = image_dir
        self.json_path = json_path
        self.text_class = text_class
        self.split = split
        self.num_captions = num_captions
        self.tokenizer = tokenizer 

        # === CLIP-style preprocessing ===
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        
        with open(self.json_path, 'r') as file:
            self.data_info = json.load(file)
            self.data_info = self.data_info['images']
        
        if self.split == "sup_and_unsup":
            # Combine both train and val splits. These will have the train or split in them
            self.images_info = [item for item in self.data_info if item.get('split') in ['train', 'val']]
        
        else:
            if self.split == "unsup":
                self.images_info = [item for item in self.data_info if item.get('split') == 'train']

            elif self.split == "sup":
                self.images_info = [item for item in self.data_info if item.get('split') == 'val']

            elif self.split == "test":
                self.images_info = [item for item in self.data_info if item.get('split') == 'test']           
            else:
                raise ValueError(f"Invalid split: {self.split}") 

            # NOTE overfit to one example
            #self.images_info = self.images_info[:2]
            #self.images_info = [first_entry]

        #print(len(self.images_info))
        #print(self.images_info[0])
        #print(self.images_info[0].keys())

    def process_captions(self, img_info):
        """Helper to extract, tokenize, deduplicate, and select captions."""
        all_captions = [" ".join(s["tokens"]) for s in img_info["sentences"]]


        # Tokenize all
        tokenized = tokenize_texts(tuple(all_captions), self.tokenizer)

        # Optional sanity check
        # check_duplicates(tokenized, all_captions)

        # Remove duplicates
        unique_captions = list({tuple(t.tolist()): t for t in tokenized}.values())

        # Select subset (if needed)
        selected = unique_captions[:self.num_captions] if len(unique_captions) <= self.num_captions \
                else random.sample(unique_captions, self.num_captions)

        caption_tensor = torch.stack(selected)
        if self.num_captions == 1:
            caption_tensor = caption_tensor[0].unsqueeze(0)

        return caption_tensor

    def __len__(self):
        return len(self.images_info)
    
    def __getitem__(self, idx):
        #print(f'Working wiht idx: {idx}')
        
        img_info = self.images_info[idx]
        img_filename = img_info['filename']
        img_path = os.path.join(self.image_dir, img_filename)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        if self.split == "sup_and_unsup":
            split = img_info['split']  # This is either 'train' (A) or 'val' (B)

            if split == 'train':
                # Split A: return only image
                return image
            
            elif split == 'val':
                caption_tensor = self.process_captions(img_info)
                return image, caption_tensor
        
        if self.split == "sup":
            
            caption_tensor = self.process_captions(img_info)
            return image, caption_tensor
        
        elif self.split == "unsup":
            return image

        elif self.split == 'test':
            self.label = _search_class(img_filename, self.text_class)
            return image, self.label


#============ MAIN FOR TEST ========#
if __name__ == '__main__':
    
    image_dir = 'data/RSICD_images'
    
    json_path = 'data/dataset_rsicd.json'
    
    data = rsicd_loader(image_dir, json_path, split="B")
    a = data[0]

    #plt.imshow(a[0])
    
    #for i, (img, text) in enumerate(data):
    #    plt.imshow(img)
    #    plt.title(text)
    #    plt.savefig('test_loader.png')
    #    break
