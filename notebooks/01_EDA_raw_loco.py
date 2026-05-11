import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    from os import path
    from pathlib import Path
    from typing import Any, Dict, List, Tuple

    import marimo as mo
    import matplotlib.pyplot as plt

    import numpy as np
    import cv2

    from src import paths
    from src.utils import file_ops

    return (
        Any,
        Dict,
        List,
        Path,
        Tuple,
        cv2,
        file_ops,
        mo,
        np,
        path,
        paths,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Investigate dataset class information
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Get dataset configuration
    """)
    return


@app.cell
def _(file_ops, paths):
    dataset_name = 'loco'

    # Get the loco dataset config path
    config_path = paths.CONFIG_DIR / 'data_config.yaml'
    config = file_ops.load_yaml_config(config_path=config_path)

    # Get annotation dirpath
    raw_dataset_dir = paths.RAW_DATA_DIR / dataset_name
    annotation_dir = paths.RAW_DATA_DIR / dataset_name / 'annotations'
    assert annotation_dir.exists()

    # Get dataset information from the config file
    dataset_config = config['datasets'][dataset_name]
    annotation_filenames = dataset_config['annotations']
    splits = dataset_config.get('splits', [])
    return annotation_dir, annotation_filenames, raw_dataset_dir, splits


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check that every annotation file contains identical class information
    """)
    return


@app.cell
def _(List, Path, Tuple, file_ops):
    def extract_classes(json_path: Path) -> List[Tuple[int, str]]:
        """Extract classes from a COCO format annotation file"""
        data = file_ops.load_json_config(json_path)

        categories = data.get('categories', [])
        if len(categories) == 0:
            raise ValueError(f"Invalid COCO format JSON annotation file {json_path}. "
                             "No categories found.")

        if not isinstance(categories, list):
            raise ValueError("Invalid data for categories. ",
                             f"Expected list but found {type(categories).__name__}")

        classes = [(cat['id'], cat['name']) for cat in categories]

        return classes

    return (extract_classes,)


@app.cell
def _(List, Path, annotation_dir, annotation_filenames, extract_classes):
    def validate_class_compatibility(annotation_dir: Path, annotation_filenames: List[str]) -> bool:
        # Checks that all annotation files contain identical class information

        class_dict = {}
        for anno_fname in annotation_filenames:
            anno_fpath = annotation_dir / anno_fname
            classes = extract_classes(anno_fpath)
            classes.sort()
            class_dict[anno_fname] = classes

        comparison_classes = list(class_dict.values())[0]
        all_compatible = True
        for anno_fname in annotation_filenames:
            all_compatible = all_compatible and (comparison_classes == class_dict[anno_fname])

        if not all_compatible:
            raise ValueError("FATAL: Class information is not identical between annotation files")
        else:
            print("SUCCESS: Class information identical between annotation files!")

        for cl in comparison_classes:
            print(cl)

        return all_compatible

    validate_class_compatibility(annotation_dir, annotation_filenames)
    return


@app.cell
def _(Dict, List, Path, annotation_dir, file_ops, path, splits):
    def check_data_leakage(splits_config: Dict[str, List[str]], annotation_dir: Path, directory_overlap_check=False) -> None:
        """Check for data leakage (image overlap) across splits..."""
        split_images = {}
        split_image_dirs = {}

        # Gather all unique image paths per split
        for split_name, filenames in splits_config.items():
            if not filenames:
                continue

            image_paths = set()
            image_dirs = set()
            for fname in filenames:
                data = file_ops.load_json_config(annotation_dir / fname)
                for img in data.get('images', []):
                    image_paths.add(img['path'])
                    image_dirs.add(path.dirname(img['path']))

            split_images[split_name] = image_paths
            print(f"\t{split_name} split contains {len(image_paths)} unique images.")

            split_image_dirs[split_name] = image_dirs

        # Check disjointness between all available splits
        if 'train' in split_images and 'val' in split_images:
            overlap = split_images['train'].intersection(split_images['val'])
            if overlap:
                raise ValueError(f"FATAL: Data leakage detected! {len(overlap)} images overlab between Train and Val")
            else:
                print("SUCCESS: Train and Val splits are perfectly disjoint.")

            dir_overlap = split_image_dirs['train'].intersection(split_image_dirs['val'])
            if directory_overlap_check and dir_overlap:
                raise ValueError(f"FATAL: Directory leakage detected! {len(dir_overlap)} directories overlap between Train and Val")
            elif directory_overlap_check:
                print("SUCCESS: Train and Val splits are perfectly contained in disjoint directories.")

    check_data_leakage(splits, annotation_dir, directory_overlap_check=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Investigate annotation file format
    """)
    return


@app.cell
def _(List, Path, annotation_dir, annotation_filenames, file_ops):
    def explore_coco_file(json_path: Path) -> None:
        """Checks implemented dataset COCO format dataset features"""

        print(f"Checking annotation JSON file: {json_path}...")

        data = file_ops.load_json_config(json_path)

        info = data.get('info', [])
        print(f"\t'info' implemented: {len(info) > 0}")
        if len(info) > 0:
            print(info)

        licenses = data.get('licenses', [])
        print(f"\t'licenses' implemented: {len(licenses) > 0}")
        if len(licenses) > 0:
            print(licenses)

        images = data.get('images', [])
        if len(images) == 0:
            raise ValueError(f"Invalid format for COCO format annotation file {json_path}. No images found!")
        print(f"\tNum images: {len(images)}")
        print(f"\tImage info sample: {images[0]}")

        annotations = data.get('annotations', [])
        if len(annotations) == 0:
            raise ValueError(f"Invalid format for COCO format annotation file {json_path}. No annotations found!")
        print(f"\tNum annotations: {len(annotations)}")
        print(f"\tAnnotation info sample: {annotations[0]}")

        categories = data.get('categories', [])
        if len(categories) == 0:
            raise ValueError(f"Invalid format for COCO format annotation file {json_path}. No categories found!")
        print(f"\tNum categories: {len(categories)}")
        print(f"\tCategory info sample: {categories[0]}")

    def explore_all_files(annotation_dir: Path, annotation_filenames: List[str]) -> None:
        for anno_fname in annotation_filenames:
            explore_coco_file(annotation_dir / anno_fname)

    explore_all_files(annotation_dir, annotation_filenames)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bounding box anomaly check
    """)
    return


@app.cell
def _(Any, Dict, List, Path, Tuple, file_ops):
    def check_invalid_bboxes(json_path: Path) -> Tuple[int, List[Dict[str, Any]]]:
        """Checks for invalid bounding boxes in a COCO format annotation file"""
        data = file_ops.load_json_config(json_path)

        annotations = data.get('annotations', [])
        if len(annotations) == 0:
            raise ValueError(f"Invalid JSON annotation file. No annotations were found")

        invalid_boxes = []

        for ann in annotations:
            bbox = ann['bbox']
            width, height = bbox[2:]
            if width <= 0 or height <= 0:
                invalid_boxes.append(ann)

        return (len(annotations), invalid_boxes)

    return (check_invalid_bboxes,)


@app.cell
def _(List, Path, annotation_dir, annotation_filenames, check_invalid_bboxes):
    def global_invalid_bboxes_check(annotation_dir: Path, annotation_filenames: List[str]) -> None:
        """Checks invalid bounding boxes in a list of annotation files"""
        total_invalids = 0
        for anno_fname in annotation_filenames:
            anno_fpath = annotation_dir / anno_fname
            (num_annotations, invalid_boxes) = check_invalid_bboxes(anno_fpath)
            percent_invalid = 100.0 * len(invalid_boxes) / num_annotations
            print(f"{anno_fname} contains: ",
                  f"Total bboxes = {num_annotations} | ",
                  f"Invalid bboxes: {len(invalid_boxes)}")
            total_invalids += len(invalid_boxes)

        assert total_invalids == 0

        print("SUCCESS: No invalid bounding boxes detected!")

    global_invalid_bboxes_check(annotation_dir, annotation_filenames)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check for annotation completeness and missing images
    """)
    return


@app.cell
def _(
    List,
    Path,
    annotation_dir,
    annotation_filenames,
    file_ops,
    raw_dataset_dir,
):
    def validate_annotation_file_completeness(json_path: Path, data_dirpath: Path) -> bool:
        """
        Validates the completeness of the annotation file.

        Checks include:
            1.) That all images in the image list have annotations.
            2.) That all images in the annotation data are listed.
            3.) That all image ids or annotation ids are unique.
            4.) That all image paths referenced are unique
        """
        data = file_ops.load_json_config(json_path)

        print(f"Validating completeness of JSON annotation file: {json_path}...")

        # Build a set of image ids and image paths while checking for duplicates
        images = data.get('images', [])
        if len(images) == 0:
            raise ValueError(f"Invalid format for COCO format annotation file {json_path}. No images found!")

        image_ids = set()
        image_paths = set()
        duplicate_ids = []
        duplicate_paths = []
        for image in images:
            image_id = image['id']
            assert image_id is not None and image_id != ''

            image_path = image['path']
            assert image_path is not None and image_path != ''

            if image_id in image_ids:
                duplicate_ids.append(image_id)
            else:
                image_ids.add(image_id)

            if image_path in image_paths:
                duplicate_paths.append(image_path)
            else:
                image_paths.add(image_path)

        # Build a set of annotation ids and a list of image ids referenced in the annotations
        annotations = data.get('annotations', [])
        if len(annotations) == 0:
            raise ValueError(f"Invalid format for COCO format annotation file {json_path}. No annotations found!")

        annotated_image_ids = []
        annotation_ids = set()
        duplicate_annotation_ids = []
        for anno in annotations:
            annotation_id = anno['id']
            assert annotation_id is not None and annotation_id != ''

            annotation_image_id = anno['image_id']
            assert annotation_image_id is not None and annotation_image_id != ''

            if annotation_id in annotation_ids:
                duplicate_annotation_ids.append(annotation_id)
            annotation_ids.add(annotation_id)
            annotated_image_ids.append(annotation_image_id)

        annotated_image_set = set(annotated_image_ids)

        # Duplicates check
        unique_image_ids = len(duplicate_ids) == 0
        unique_annotation_ids = len(duplicate_annotation_ids) == 0

        images_annotated = image_ids.issubset(annotated_image_set) # Check 1
        if not images_annotated:
            print("\tWARNING: Not all images are annotated!")

        annotated_images_referenced = annotated_image_set.issubset(image_ids) # Check 2
        if not annotated_images_referenced:
            print("\nWARNING: Not all images referenced by annotations have info!")

        unique_ids = unique_image_ids & unique_annotation_ids # Check 3
        if not unique_ids:
            print("\nWARNING: Contains duplicate ids!")

        unique_paths = len(duplicate_paths) == 0 # Check 4
        if not unique_paths:
            print("\nWARNING: Contains duplicate image paths!")

        # Redundant but for completeness and readability
        same_image_set = annotated_image_set == image_ids

        is_complete = images_annotated and annotated_images_referenced and unique_ids and unique_paths and same_image_set

        return is_complete

    def check_missing_images(json_path: Path, data_dirpath: Path) -> List[Path]:
        """Checks that every image referenced in the annotation file exists and returns ones missing."""

        data = file_ops.load_json_config(json_path)

        # Get the relative image paths
        images = data.get('images', [])
        if len(images) == 0:
            raise ValueError(f"Invalid format for COCO format annotation file {json_path}. No images found!")

        missing_image_paths = []
        for image in images:
            image_rel_path = image['path'].lstrip("/")    # Strip potential leading / making it a root path
            image_fpath = data_dirpath / image_rel_path
            if not image_fpath.exists():
                missing_image_paths.append(image_fpath)

        return missing_image_paths

    def validate_annotation_files(annotation_filenames: List, annotation_dir: Path, data_dirpath: Path) -> None:
        """Validates completeness of annotation and image data for all annotation files"""

        all_missing_images = []
        incomplete_annotation_files = []
        for anno_fname in annotation_filenames:
            print(f"Checking JSON annotation file: {anno_fname}...")
            anno_fpath = annotation_dir / anno_fname

            is_complete = validate_annotation_file_completeness(anno_fpath, data_dirpath)
            if not is_complete:
                incomplete_annotation_files.append(anno_fname)
                print(f"\tWARNING: {anno_fname} is not complete!")
            else:
                print(f"All information complete in {anno_fname}")

            missing_images = check_missing_images(anno_fpath, data_dirpath)
            all_missing_images.extend(missing_images)
            if len(missing_images) == 0:
                print(f"\tNo missing images in {anno_fname}")
            else:
                print(f"\tWARNING: {len(missing_images)} found in {anno_fname}")

        assert len(all_missing_images) == 0 and len(incomplete_annotation_files) == 0
        print("\nSUCCESS: Annotation files and data complete!")

    validate_annotation_files(annotation_filenames, annotation_dir, raw_dataset_dir)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visual validation of bounding box format
    """)
    return


@app.cell
def _(Tuple, cv2, np):
    def draw_bbox(image: np.ndarray,
                  bbox: Tuple[float, float, float, float],
                  color: Tuple[int, int, int]=(0, 255, 0),
                  thickness=2) -> None:
        x = int(bbox[0])
        y = int(bbox[1])
        w = int(bbox[2])
        h = int(bbox[3])

        cv2.rectangle(image, (x, y), (x+w, y+h), color=color, thickness=2)

    return (draw_bbox,)


@app.cell
def _(
    Any,
    List,
    Path,
    annotation_dir,
    annotation_filenames,
    cv2,
    draw_bbox,
    file_ops,
    plt,
    raw_dataset_dir,
):
    def plot_single_bbox(json_path: Path, data_dirpath: Path) -> Any:
        """Plots 1 bounding box in 1 image for visualisation"""

        data = file_ops.load_json_config(json_path)

        image_dict = {}
        images = data.get('images', [])
        for img in images:
            image_dict[img['id']] = img['path']

        class_dict = {}
        classes = data.get('categories', [])
        for cl in classes:
            class_dict[cl['id']] = cl['name']

        annotations = data.get('annotations', [])
        anno = annotations[0]
        class_id = anno['category_id']
        class_name = class_dict[class_id]
        print(class_id, class_name)

        image_id = anno['image_id']
        bbox = anno['bbox']

        image_fpath = data_dirpath / image_dict[image_id].lstrip('/')
        assert image_fpath.exists()
        img = cv2.imread(str(image_fpath))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        draw_bbox(img, bbox, color=(0, 255, 0), thickness=3)

        fig, ax = plt.subplots()
        ax.imshow(img)

        return fig

    def plot_single_bbox_samples(raw_dataset_dir: Path,
                                 annotation_dir: Path,
                                 annotation_filenames: List[str]) -> List[Any]:
        figs = []
        for anno_fname in annotation_filenames:
            figs.append(plot_single_bbox(annotation_dir / anno_fname, raw_dataset_dir))

        return figs

    plot_single_bbox_samples(raw_dataset_dir, annotation_dir, annotation_filenames)
    return


@app.cell
def _(
    Any,
    List,
    Path,
    annotation_dir,
    annotation_filenames,
    cv2,
    draw_bbox,
    file_ops,
    plt,
    raw_dataset_dir,
):
    def plot_image_bounding_boxes(json_path: Path, data_dirpath: Path) -> Any:
        """Plots all bounding boxes in 1 image for visualisation"""

        data = file_ops.load_json_config(json_path)

        image_dict = {}
        images = data.get('images', [])
        for img in images:
            img_id = img['id']
            if img_id in image_dict:
                raise ValueError(f"Error: Found duplicate image id {img_id} in {json_path}")
            image_dict[img_id] = img['path']

        class_dict = {}
        classes = data.get('categories', [])
        for cl in classes:
            class_dict[cl['id']] = cl['name']

        image_bbox_dict = {}
        annotations = data.get('annotations', [])
        for anno in annotations:
            image_id = anno['image_id']
            bbox = anno['bbox']
            class_id = anno['category_id']

            if image_id not in image_bbox_dict:
                image_bbox_dict[image_id] = []

            image_bbox_dict[image_id].append({'bbox': bbox,
                                             'class_id': class_id})

        image_id = anno['image_id']
        bbox = anno['bbox']

        image_ids = list(image_dict.keys())
        image_id = image_ids[0]
        image_fpath = data_dirpath / image_dict[image_id].lstrip('/')
        assert image_fpath.exists()
        img = cv2.imread(str(image_fpath))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bbox_annotations = image_bbox_dict[image_id]
        print(f"Displaying image sample {image_fpath}")
        for i, bbox_anno in enumerate(bbox_annotations):
            bbox = bbox_anno['bbox']
            class_id = bbox_anno['class_id']
            class_name = class_dict[class_id]
            print(f"\tbbox {i} | Class id: {class_id} | Class name: {class_name}")
            draw_bbox(img, bbox, color=(0, 255, 0), thickness=3)

        fig, ax = plt.subplots()
        ax.imshow(img)

        return fig

    def plot_annotated_image_samples(raw_dataset_dir: Path,
                                     annotation_dir: Path,
                                     annotation_filenames: List[str]) -> List[Any]:
        figs = []
        for anno_fname in annotation_filenames:
            figs.append(plot_image_bounding_boxes(annotation_dir / anno_fname, raw_dataset_dir))

        return figs

    plot_annotated_image_samples(raw_dataset_dir, annotation_dir, annotation_filenames)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
