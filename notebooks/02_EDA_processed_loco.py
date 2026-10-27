"""Exploratory data analysis of processed LOCO dataset in YOLO format.

Analyses processed YOLO splits, raw subset class distributions, split balance,
object densities, bounding box geometries, and visual labels to document
modelling decisions for Phase 1 training.
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    from collections import Counter
    from pathlib import Path
    from typing import Any

    import cv2
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from pydantic import ValidationError

    from src import paths
    from src.datatypes import DatasetConfigModel
    from src.utils import file_ops

    return (
        Any,
        Counter,
        DatasetConfigModel,
        Image,
        Path,
        ValidationError,
        cv2,
        file_ops,
        mo,
        np,
        paths,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LOCO Processed Dataset EDA & Modelling Decisions

    This notebook analyses the processed LOCO dataset in YOLO format
    (`data/processed/loco/`) and establishes quantitative justification for
    modelling decisions in the training phase.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get dataset configuration
    """)
    return


@app.cell
def _(DatasetConfigModel, ValidationError, file_ops, paths):
    dataset_name = "loco"

    config_path = paths.CONFIG_DIR / "data_config.yaml"
    full_config = file_ops.load_yaml_config(config_path=config_path)
    dataset_config = full_config["datasets"][dataset_name]

    try:
        dataset_config_model = DatasetConfigModel.model_validate(dataset_config)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration for {dataset_name}") from e

    anno_dir_name = (dataset_config_model.model_extra or {}).get("annotations_path", "annotations")
    anno_dir = paths.RAW_DATA_DIR / dataset_name / anno_dir_name

    processed_dir = paths.PROCESSED_DATA_DIR / dataset_name
    processed_config = processed_dir / "dataset.yaml"
    return anno_dir, dataset_config, processed_config, processed_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load and verify processed YOLO splits
    """)
    return


@app.cell
def _(file_ops, processed_config):
    dataset_yaml = file_ops.load_yaml_config(processed_config)
    class_names = dataset_yaml["names"]

    print(f"Dataset configuration: {processed_config}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train split path: {dataset_yaml.get('train')}")
    print(f"Val split path:   {dataset_yaml.get('val')}")
    return (class_names,)


@app.cell
def _(Path):
    def parse_yolo_label_file(
        label_path: Path,
    ) -> list[tuple[int, float, float, float, float]]:
        """Parses a YOLO format label file and validates coordinate bounds.

        Args:
            label_path (Path): Path to the YOLO label file.

        Returns:
            list[tuple[int, float, float, float, float]]: Bounding boxes as
                (class_id, cx, cy, w, h).

        Raises:
            FileNotFoundError: If label_path does not exist or is not a file.
            ValueError: If a line does not contain 5 values or coordinates
                fall outside (0, 1].
        """
        if not label_path.is_file():
            raise FileNotFoundError(f"Label file does not exist: {label_path}")

        boxes: list[tuple[int, float, float, float, float]] = []
        with open(label_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) != 5:
                    raise ValueError(
                        f"Invalid line format in {label_path}:{line_num}: expected 5 elements, got {len(parts)}"
                    )
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                for coord, name in ((cx, "cx"), (cy, "cy"), (w, "w"), (h, "h")):
                    if not (0.0 < coord <= 1.0):
                        raise ValueError(
                            f"Coordinate {name}={coord} outside (0, 1] in {label_path}:{line_num}"
                        )
                boxes.append((class_id, cx, cy, w, h))

        return boxes

    def load_processed_split(
        processed_dir: Path, split_name: str
    ) -> dict[Path, list[tuple[int, float, float, float, float]]]:
        """Loads and verifies YOLO annotations for a dataset split.

        Args:
            processed_dir (Path): Root directory of the processed dataset.
            split_name (str): Name of the split to load.

        Returns:
            dict[Path, list[tuple[int, float, float, float, float]]]: Mapping
                of image paths to their parsed bounding boxes.

        Raises:
            FileNotFoundError: If the split image directory or label directory does not exist.
            ValueError: If image stems and label stems do not match 1:1.
        """
        img_dir = processed_dir / "images" / split_name
        label_dir = processed_dir / "labels" / split_name
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {img_dir}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

        image_paths = sorted(
            p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        label_paths = sorted(p for p in label_dir.iterdir() if p.suffix == ".txt")

        img_stems = {p.stem: p for p in image_paths}
        label_stems = {p.stem: p for p in label_paths}

        if img_stems.keys() != label_stems.keys():
            missing_labels = set(img_stems.keys()) - set(label_stems.keys())
            missing_images = set(label_stems.keys()) - set(img_stems.keys())
            raise ValueError(
                f"Split {split_name} stem mismatch: {len(missing_labels)} missing labels, "
                f"{len(missing_images)} missing images"
            )

        split_data: dict[Path, list[tuple[int, float, float, float, float]]] = {}
        for stem, img_path in img_stems.items():
            split_data[img_path] = parse_yolo_label_file(label_stems[stem])

        return split_data

    return (load_processed_split,)


@app.cell
def _(load_processed_split, processed_dir):
    train_records = load_processed_split(processed_dir, "train")
    val_records = load_processed_split(processed_dir, "val")

    assert len(train_records) == 2820, f"Expected 2820 train images, got {len(train_records)}"
    assert len(val_records) == 2277, f"Expected 2277 val images, got {len(val_records)}"

    print("Verified 1:1 image/label parity and coordinate bounds in (0, 1]:")
    print(f"  Train: {len(train_records)} images, {sum(len(b) for b in train_records.values())} boxes")
    print(f"  Val:   {len(val_records)} images, {sum(len(b) for b in val_records.values())} boxes")
    return train_records, val_records


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Raw subset class distribution
    """)
    return


@app.cell
def _(Any, Counter, anno_dir, dataset_config, file_ops, np, plt):
    raw_annotation_files = dataset_config["annotations"]
    raw_subset_counts = {}

    for fname in raw_annotation_files:
        fpath = anno_dir / fname
        coco_data = file_ops.load_json_config(fpath)
        cat_id_to_name = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
        counter = Counter(
            cat_id_to_name.get(anno["category_id"], str(anno["category_id"]))
            for anno in coco_data.get("annotations", [])
        )
        raw_subset_counts[fname] = counter

    # Display distribution table
    target_classes = ["small_load_carrier", "forklift", "pallet", "stillage", "pallet_truck"]
    header = f"{'Raw Annotation File':<26} | " + " | ".join(f"{c:>18}" for c in target_classes) + " | Total"
    print(header)
    print("-" * len(header))
    for fname, counts in raw_subset_counts.items():
        row_vals = " | ".join(f"{counts.get(c, 0):>18d}" for c in target_classes)
        total_boxes = sum(counts.values())
        print(f"{fname:<26} | {row_vals} | {total_boxes:>6d}")

    def plot_raw_subset_distribution(
        counts_by_annotation_file: dict[str, Counter],
        file_order: list[str],
        classes: list[str],
    ) -> Any:
        """Plots bounding box counts per class across raw annotation files.

        Args:
            counts_by_annotation_file (dict[str, Counter]): Mapping of annotation filename to class counters.
            file_order (list[str]): Sequence of annotation filenames.
            classes (list[str]): Target class names to plot.

        Returns:
            Any: Matplotlib figure containing the bar plot.
        """
        fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
        x_indices = np.arange(len(file_order))
        bar_width = 0.15

        for i, class_name in enumerate(classes):
            values = [counts_by_annotation_file[f].get(class_name, 0) for f in file_order]
            ax.bar(x_indices + i * bar_width, values, width=bar_width, label=class_name)

        short_labels = [f.replace("-v1-", "\n").replace(".json", "") for f in file_order]
        ax.set_xticks(x_indices + bar_width * 2)
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_ylabel("Bounding Box Count")
        ax.set_title("Class Instance Count Across Raw LOCO Annotation Files")
        ax.set_yscale("log")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
        ax.grid(True, linestyle="--", alpha=0.5, which="both")

        return fig

    plot_raw_subset_distribution(raw_subset_counts, raw_annotation_files, target_classes)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Modelling Decision: Split Strategy & Cross-Validation

    Inspecting individual raw subsets reveals severe class scarcity and absence across subsets:
    - `loco-sub4-v1-val.json`: 0 `small_load_carrier` instances and only 29 `stillage` instances.
    - `loco-sub1-v1-val.json`: only 5 `forklift` instances.
    - `loco-sub5-v1-train.json`: only 1 `stillage` instance.

    Attempting 5-fold cross-validation by subset or splitting validation by subset directory
    induces severe fold-level class scarcity. The official 2-split benchmark partitions
    subsets 2, 3, and 5 into training and subsets 1 and 4 into validation.

    The subsets contain images captured in sequences within warehouse environments. A random
    image-level split could place closely related views of the same scenes or objects in both
    training and validation, making validation results overly optimistic.

    Consequently, the official `train` split will be used for training and `val` for checkpoint
    selection, evaluation, and subsequent model development. Validation performance will be
    reported because an independent test set is not available.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Processed split class distribution & image-level frequency
    """)
    return


@app.cell
def _(Any, Counter, Path, class_names, np, plt, train_records, val_records):
    def compute_class_statistics(
        records: dict[Path, list[tuple[int, float, float, float, float]]],
        class_names: dict[int, str],
    ) -> dict[int, dict[str, float]]:
        """Computes per-class box count, box share, and image-level presence.

        Args:
            records (dict[Path, list[tuple[int, float, float, float, float]]]):
                Mapping of image paths to parsed YOLO bounding boxes.
            class_names (dict[int, str]): Mapping from class IDs to class names.

        Returns:
            dict[int, dict[str, float]]: Per-class dictionary with 'count',
                'share', and 'presence' metrics.
        """
        total_images = len(records)
        box_counts = Counter()
        image_presence_counts = Counter()

        for boxes in records.values():
            classes_in_image = set()
            for class_id, _, _, _, _ in boxes:
                box_counts[class_id] += 1
                classes_in_image.add(class_id)
            for class_id in classes_in_image:
                image_presence_counts[class_id] += 1

        total_boxes = sum(box_counts.values())
        stats = {}
        for class_id in sorted(class_names.keys()):
            count = box_counts[class_id]
            share = (count / total_boxes * 100.0) if total_boxes > 0 else 0.0
            presence = (image_presence_counts[class_id] / total_images * 100.0) if total_images > 0 else 0.0
            stats[class_id] = {
                "count": count,
                "share": share,
                "presence": presence,
            }
        return stats

    train_class_stats = compute_class_statistics(train_records, class_names)
    val_class_stats = compute_class_statistics(val_records, class_names)

    print(f"{'Class Name':<20} | {'Train Boxes':>11} | {'Train %':>7} | {'Train Img %':>11} | "
          f"{'Val Boxes':>9} | {'Val %':>7} | {'Val Img %':>10}")
    print("-" * 93)
    for class_id in sorted(class_names.keys()):
        class_name = class_names[class_id]
        train_stat = train_class_stats[class_id]
        val_stat = val_class_stats[class_id]
        print(
            f"{class_name:<20} | {train_stat['count']:>11d} | {train_stat['share']:>6.2f}% | "
            f"{train_stat['presence']:>10.2f}% | {val_stat['count']:>9d} | "
            f"{val_stat['share']:>6.2f}% | {val_stat['presence']:>9.2f}%"
        )

    def plot_processed_class_distribution(
        train_stats: dict[int, dict[str, float]],
        val_stats: dict[int, dict[str, float]],
        class_names: dict[int, str],
    ) -> Any:
        """Plots comparison of class share and image presence between splits.

        Args:
            train_stats (dict[int, dict[str, float]]): Class statistics for train split.
            val_stats (dict[int, dict[str, float]]): Class statistics for val split.
            class_names (dict[int, str]): Mapping from class IDs to class names.

        Returns:
            Any: Matplotlib figure containing the comparison subplots.
        """
        fig, (ax_share, ax_presence) = plt.subplots(1, 2, figsize=(12, 5))
        x_pos = np.arange(len(class_names))
        width = 0.35
        labels = [class_names[cid] for cid in sorted(class_names.keys())]

        train_shares = [train_stats[cid]["share"] for cid in sorted(class_names.keys())]
        val_shares = [val_stats[cid]["share"] for cid in sorted(class_names.keys())]
        ax_share.bar(x_pos - width / 2, train_shares, width, label="Train", color="#2980b9")
        ax_share.bar(x_pos + width / 2, val_shares, width, label="Val", color="#27ae60")
        ax_share.set_xticks(x_pos)
        ax_share.set_xticklabels(labels, rotation=30, ha="right")
        ax_share.set_ylabel("Class Share (% of all boxes)")
        ax_share.set_title("Bounding Box Share per Class")
        ax_share.legend()
        ax_share.grid(True, linestyle="--", alpha=0.5)

        train_presence = [train_stats[cid]["presence"] for cid in sorted(class_names.keys())]
        val_presence = [val_stats[cid]["presence"] for cid in sorted(class_names.keys())]
        ax_presence.bar(x_pos - width / 2, train_presence, width, label="Train", color="#2980b9")
        ax_presence.bar(x_pos + width / 2, val_presence, width, label="Val", color="#27ae60")
        ax_presence.set_xticks(x_pos)
        ax_presence.set_xticklabels(labels, rotation=30, ha="right")
        ax_presence.set_ylabel("Image Presence (% of images)")
        ax_presence.set_title("Image-Level Frequency per Class")
        ax_presence.legend()
        ax_presence.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()

        return fig

    plot_processed_class_distribution(train_class_stats, val_class_stats, class_names)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Modelling Decision: Per-Class Evaluation Metrics

    The processed dataset exhibits extreme class imbalance:
    - `pallet`: accounts for 78.39% of train boxes (90.28% image presence) and 81.09% of val boxes (95.78% presence).
    - `forklift`: represents only 0.55% of train boxes (12.34% presence) and 0.19% of val boxes (4.44% presence).

    Because aggregate mean Average Precision (mAP) averages across classes, a model that performs well
    on `pallet` while completely failing to detect `forklift` can still achieve deceptive mAP scores.
    In warehouse environments, forklift detection is safety-critical. This necessitates logging per-class
    metrics (`metrics/mAP50(B)_forklift`, `metrics/mAP50-95(B)_forklift`, etc.) alongside aggregate mAP in MLflow.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Object density
    """)
    return


@app.cell
def _(Any, np, plt, train_records, val_records):
    train_counts = np.array([len(b) for b in train_records.values()])
    val_counts = np.array([len(b) for b in val_records.values()])

    def get_density_summary(counts: np.ndarray) -> dict[str, float]:
        """Computes summary statistics and tail counts for per-image box density.

        Args:
            counts (np.ndarray): Array of bounding box counts per image.

        Returns:
            dict[str, float]: Summary metrics including mean, median, p90, p99,
                max, gt_100, and gt_300.
        """
        return {
            "mean": float(np.mean(counts)),
            "median": float(np.median(counts)),
            "p90": float(np.percentile(counts, 90)),
            "p99": float(np.percentile(counts, 99)),
            "max": int(np.max(counts)),
            "gt_100": int(np.sum(counts > 100)),
            "gt_300": int(np.sum(counts > 300)),
        }

    train_density_stats = get_density_summary(train_counts)
    val_density_stats = get_density_summary(val_counts)

    print(f"{'Metric':<18} | {'Train':>10} | {'Val':>10}")
    print("-" * 44)
    print(f"{'Mean':<18} | {train_density_stats['mean']:>10.2f} | {val_density_stats['mean']:>10.2f}")
    print(f"{'Median':<18} | {train_density_stats['median']:>10.1f} | {val_density_stats['median']:>10.1f}")
    print(f"{'p90':<18} | {train_density_stats['p90']:>10.1f} | {val_density_stats['p90']:>10.1f}")
    print(f"{'p99':<18} | {train_density_stats['p99']:>10.1f} | {val_density_stats['p99']:>10.1f}")
    print(f"{'Max':<18} | {train_density_stats['max']:>10d} | {val_density_stats['max']:>10d}")
    print(f"{'Count > 100':<18} | {train_density_stats['gt_100']:>10d} | {val_density_stats['gt_100']:>10d}")
    print(f"{'Count > 300':<18} | {train_density_stats['gt_300']:>10d} | {val_density_stats['gt_300']:>10d}")

    def plot_object_density(train_counts: np.ndarray, val_counts: np.ndarray, p99: float) -> Any:
        """Plots object density distribution histograms against max_det limit.

        Args:
            train_counts (np.ndarray): Box counts per image for train split.
            val_counts (np.ndarray): Box counts per image for val split.
            p99 (float): Bounding box density at 99th percentile.

        Returns:
            Any: Matplotlib figure containing the histogram plot.
        """
        fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
        bins = np.linspace(0, 360, 73)
        ax.hist(train_counts, bins=bins, alpha=0.6, label="Train", color="#2980b9", edgecolor="black")
        ax.hist(val_counts, bins=bins, alpha=0.5, label="Val", color="#27ae60", edgecolor="black")

        ax.axvline(p99, color="#e67e22", linestyle="--", linewidth=2, label=f"Train p99 = {p99:.0f}")
        ax.axvline(300, color="#c0392b", linestyle="-", linewidth=2.5, label="Default max_det = 300")

        ax.set_xlabel("Number of Objects per Image")
        ax.set_ylabel("Frequency (Image Count)")
        ax.set_title("Distribution of Object Density per Image vs max_det Limit")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
        ax.grid(True, linestyle="--", alpha=0.5)

        return fig

    plot_object_density(train_counts, val_counts, p99=train_density_stats["p99"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Modelling Decision: Evaluation & Inference `max_det` Limit

    Warehouse imagery contains high object clutter:
    - Median density is 19.0 (train) and 21.0 (val) objects per image.
    - 99% of images contain up to 162 objects.
    - High-density tails reach 333 objects in train and 345 objects in val.
    - 134 train images and 62 val images contain over 100 objects, while 2 train images and 6 val images exceed 300 objects.

    Images containing more than 300 annotated objects create a risk of a detection cap limiting
    recall. During model validation and deployment, maximum detections will be set explicitly
    to 500 to reduce this risk (`max_det=500`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Bounding box geometry & spatial distribution
    """)
    return


@app.cell
def _(
    Any,
    Counter,
    Image,
    Path,
    class_names,
    np,
    plt,
    train_records,
    val_records,
):
    def extract_geometry_metrics(
        records: dict[Path, list[tuple[int, float, float, float, float]]],
    ) -> tuple[dict[str, list[Any]], Counter[tuple[int, int]]]:
        """Measures image resolutions and extracts letterbox-640 box geometry.

        Args:
            records (dict[Path, list[tuple[int, float, float, float, float]]]):
                Mapping of image paths to parsed YOLO bounding boxes.

        Returns:
            tuple[dict[str, list[Any]], Counter[tuple[int, int]]]: Tuple of
                (geometry metrics dictionary, image resolution Counter as (width, height)).
        """
        data: dict[str, list[Any]] = {
            "class_id": [],
            "cx": [],
            "cy": [],
            "w": [],
            "h": [],
            "aspect_ratio": [],
            "area_640": [],
            "scale_bucket": [],
        }
        resolutions: Counter[tuple[int, int]] = Counter()

        for img_path, boxes in records.items():
            with Image.open(img_path) as im:
                img_w, img_h = im.size
            resolutions[(img_w, img_h)] += 1
            scale = 640.0 / max(img_w, img_h)

            for class_id, cx, cy, w, h in boxes:
                bw_px = w * img_w
                bh_px = h * img_h
                area_640 = (bw_px * scale) * (bh_px * scale)
                if area_640 < 32.0 ** 2:
                    bucket = "small"
                elif area_640 < 96.0 ** 2:
                    bucket = "medium"
                else:
                    bucket = "large"

                data["class_id"].append(class_id)
                data["cx"].append(cx)
                data["cy"].append(cy)
                data["w"].append(w)
                data["h"].append(h)
                data["aspect_ratio"].append(bw_px / bh_px if bh_px > 0 else 0.0)
                data["area_640"].append(area_640)
                data["scale_bucket"].append(bucket)

        return data, resolutions

    train_geom, train_resolutions = extract_geometry_metrics(train_records)
    val_geom, val_resolutions = extract_geometry_metrics(val_records)

    print("Measured Image Resolutions (width x height):")
    train_res_str = ", ".join(f"{w}x{h}: {cnt}" for (w, h), cnt in train_resolutions.most_common())
    val_res_str = ", ".join(f"{w}x{h}: {cnt}" for (w, h), cnt in val_resolutions.most_common())
    print(f"  Train: {train_res_str}")
    print(f"  Val:   {val_res_str}")

    def compute_scale_percentages(
        geom_data: dict[str, list[Any]],
    ) -> tuple[float, float, float]:
        """Computes percentage share of small, medium, and large bounding boxes.

        Args:
            geom_data (dict[str, list[Any]]): Extracted geometry metrics dictionary.

        Returns:
            tuple[float, float, float]: Percentages for (small, medium, large) buckets.
        """
        total = len(geom_data["scale_bucket"])
        small_pct = sum(1 for b in geom_data["scale_bucket"] if b == "small") / total * 100.0
        med_pct = sum(1 for b in geom_data["scale_bucket"] if b == "medium") / total * 100.0
        large_pct = sum(1 for b in geom_data["scale_bucket"] if b == "large") / total * 100.0
        return small_pct, med_pct, large_pct

    t_s, t_m, t_l = compute_scale_percentages(train_geom)
    v_s, v_m, v_l = compute_scale_percentages(val_geom)

    print("\nCOCO Scale Buckets at Letterboxed 640 Input Resolution:")
    print(f"  Train: Small (<32^2) = {t_s:.1f}%, Medium (32^2..96^2) = {t_m:.1f}%, Large (>=96^2) = {t_l:.1f}%")
    print(f"  Val:   Small (<32^2) = {v_s:.1f}%, Medium (32^2..96^2) = {v_m:.1f}%, Large (>=96^2) = {v_l:.1f}%")

    print("\nTrain Median Pixel Aspect Ratio (width / height) per Class:")
    for cid in sorted(class_names.keys()):
        class_aspect_ratio = [
            ar for c, ar in zip(train_geom["class_id"], train_geom["aspect_ratio"], strict=True) if c == cid
        ]
        print(f"  {class_names[cid]:<20}: {float(np.median(class_aspect_ratio)):.2f}")

    train_cx_median = float(np.median(train_geom["cx"]))
    train_cy_median = float(np.median(train_geom["cy"]))
    val_cx_median = float(np.median(val_geom["cx"]))
    val_cy_median = float(np.median(val_geom["cy"]))
    print(f"\nTrain Spatial Centre Medians: cx = {train_cx_median:.2f}, cy = {train_cy_median:.2f}")
    print(f"Val Spatial Centre Medians:   cx = {val_cx_median:.2f}, cy = {val_cy_median:.2f}")

    def plot_bounding_box_geometry(
        t_scales: tuple[float, float, float],
        v_scales: tuple[float, float, float],
        geom_data: dict[str, list[Any]],
        class_names: dict[int, str],
    ) -> Any:
        """Plots bounding box scale buckets, aspect ratios, and spatial distribution.

        Args:
            t_scales (tuple[float, float, float]): Small, medium, large percentages for train.
            v_scales (tuple[float, float, float]): Small, medium, large percentages for val.
            geom_data (dict[str, list[Any]]): Geometry metrics for train split.
            class_names (dict[int, str]): Mapping from class IDs to class names.

        Returns:
            Any: Matplotlib figure containing the three geometry subplots.
        """
        fig, (ax_scale, ax_aspect, ax_spatial) = plt.subplots(1, 3, figsize=(16, 5))

        # Scale buckets comparison
        x_b = np.arange(3)
        bw = 0.35
        ax_scale.bar(x_b - bw / 2, list(t_scales), bw, label="Train", color="#2980b9")
        ax_scale.bar(x_b + bw / 2, list(v_scales), bw, label="Val", color="#27ae60")
        ax_scale.set_xticks(x_b)
        ax_scale.set_xticklabels(["Small (<32^2)", "Medium (32^2..96^2)", "Large (>=96^2)"])
        ax_scale.set_ylabel("Share (%)")
        ax_scale.set_title("COCO Scale Buckets (Letterbox 640)")
        ax_scale.legend()
        ax_scale.grid(True, linestyle="--", alpha=0.5)

        # Aspect ratio boxplot per class
        train_ar_by_class = [
            [ar for cid, ar in zip(geom_data["class_id"], geom_data["aspect_ratio"], strict=True) if cid == c]
            for c in sorted(class_names.keys())
        ]
        ax_aspect.boxplot(train_ar_by_class, showfliers=False)
        ax_aspect.set_xticklabels([class_names[c] for c in sorted(class_names.keys())], rotation=30, ha="right")
        ax_aspect.set_ylabel("Pixel Aspect Ratio (width / height)")
        ax_aspect.set_title("Train Pixel Aspect Ratio per Class")
        ax_aspect.grid(True, linestyle="--", alpha=0.5)

        # Spatial center distribution (2D hexbin)
        hb = ax_spatial.hexbin(geom_data["cx"], geom_data["cy"], gridsize=40, cmap="viridis", mincnt=1)
        ax_spatial.set_xlim(0, 1)
        ax_spatial.set_ylim(1, 0)  # Inverted image coordinate system
        ax_spatial.set_xlabel("Normalised Centre X")
        ax_spatial.set_ylabel("Normalised Centre Y")
        ax_spatial.set_title("Train Spatial Centre Distribution (cx, cy)")
        plt.colorbar(hb, ax=ax_spatial, label="Box Count")

        plt.tight_layout()

        return fig

    plot_bounding_box_geometry((t_s, t_m, t_l), (v_s, v_m, v_l), train_geom, class_names)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Modelling Decision: Evaluation Metrics & Geometry Takeaways

    - Image Resolutions: All dataset images have landscape aspect ratios (`1280x720`, `1920x1080`, `1280x960`, and `1280x800` in train, plus `640x480` in val).
    - Scale Buckets (`<32^2` px at Letterbox 640): Under letterboxed scaling at `640` resolution, `67.7%` of train and `57.3%` of val objects fall into the COCO small bucket (`28.1%` train and `36.6%` val medium, `4.2%` train and `6.1%` val large). mAP50 evaluates detection performance using an IoU threshold of 0.50. mAP50-95 averages mAP across 10 IoU thresholds from 0.50 to 0.95 in 0.05 increments, giving an overall metric of detection performance. Because minor pixel shifts on small boxes reduce IoU sharply at stricter thresholds, both mAP50 and mAP50-95 will be tracked. Additionally, to evaluate performance across scales, the performance in each individual scale bucket will also be tracked.
    - Pixel Aspect Ratio per Class (`width / height`): In pixel coordinates, `pallet` (median `2.31`) and `small_load_carrier` (median `1.51`) are predominantly wide, whereas `pallet_truck` (`0.75`), `forklift` (`0.64`), and `stillage` (`0.58`) are tall and vertical. Anchor-free YOLO heads accommodate these varied aspect ratios natively without custom anchor clustering.
    - Spatial Centre Distribution (`cx, cy`): Object centres cluster horizontally across the frame (`cx` median `0.55` in train, `0.49` in val) in a vertical band slightly above the midpoint (`cy` median `0.38` in both splits), with minimal corner bias.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Visual verification of converted YOLO labels
    """)
    return


@app.cell
def _(cv2, np):
    def draw_yolo_bboxes(
        image: np.ndarray,
        boxes: list[tuple[int, float, float, float, float]],
    ) -> np.ndarray:
        """Draws class-coloured YOLO bounding boxes on an image.

        Args:
            image (np.ndarray): Input RGB image array.
            boxes (list[tuple[int, float, float, float, float]]): Bounding boxes
                as (class_id, cx, cy, w, h) in normalised coordinates.

        Returns:
            np.ndarray: Annotated RGB image copy.
        """
        canvas = image.copy()
        img_h, img_w = canvas.shape[:2]

        palette = [
            (46, 204, 113),   # small_load_carrier: green
            (231, 76, 60),    # forklift: red
            (52, 152, 219),   # pallet: blue
            (241, 196, 15),   # stillage: yellow
            (155, 89, 182),   # pallet_truck: purple
        ]

        for class_id, cx, cy, w, h in boxes:
            color = palette[class_id % len(palette)]
            x1 = max(0, min(img_w - 1, int((cx - w / 2.0) * img_w)))
            y1 = max(0, min(img_h - 1, int((cy - h / 2.0) * img_h)))
            x2 = max(0, min(img_w - 1, int((cx + w / 2.0) * img_w)))
            y2 = max(0, min(img_h - 1, int((cy + h / 2.0) * img_h)))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color=color, thickness=2)

        return canvas

    return (draw_yolo_bboxes,)


@app.cell
def _(Any, Path, cv2, draw_yolo_bboxes, plt, train_records, val_records):
    def plot_yolo_sample_image(
        img_path: Path,
        boxes: list[tuple[int, float, float, float, float]],
        split_name: str,
    ) -> Any:
        """Plots a single image with denormalised YOLO bounding boxes.

        Args:
            img_path (Path): Path to the image file.
            boxes (list[tuple[int, float, float, float, float]]): Bounding boxes as
                (class_id, cx, cy, w, h).
            split_name (str): Split label for the subplot title.

        Returns:
            Any: Matplotlib figure containing the annotated image.

        Raises:
            FileNotFoundError: If the image file cannot be read by OpenCV.
        """
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image file: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        annotated = draw_yolo_bboxes(rgb, boxes)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(annotated)
        ax.set_title(f"{split_name} Sample: {img_path.name} ({len(boxes)} boxes)")
        ax.axis("off")

        return fig

    def plot_yolo_annotated_samples(
        samples: list[tuple[str, Path, list[tuple[int, float, float, float, float]]]],
    ) -> list[Any]:
        """Plots annotated sample images across splits.

        Args:
            samples (list[tuple[str, Path, list[tuple[int, float, float, float, float]]]]):
                List of (split_name, img_path, boxes) tuples.

        Returns:
            list[Any]: List of matplotlib figures.
        """
        figs = []
        for split_name, img_path, boxes in samples:
            figs.append(plot_yolo_sample_image(img_path, boxes, split_name))

        return figs

    sample_items = [
        ("Train", *next(iter(train_records.items()))),
        ("Val", *next(iter(val_records.items()))),
    ]
    plot_yolo_annotated_samples(sample_items)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Verification Summary

    Visual inspection from samples from the `train` and `val` set show that:
    1. Normalised YOLO coordinates denormalise correctly to pixel dimensions.
    2. Bounding boxes align with the physical boundaries and positions of objects in the sample images.
    3. None of the bounding boxes displayed exhibit out-of-bounds clipping or inverted coordinates.
    """)
    return


if __name__ == "__main__":
    app.run()
