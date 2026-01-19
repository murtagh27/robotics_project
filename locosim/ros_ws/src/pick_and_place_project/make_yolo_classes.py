"""
Generate YOLO classes.txt file from BRICK_CLASSES dictionary.

This utility creates a classes.txt file required for YOLO training, where each
line number corresponds to a class ID and contains the class name.
"""

from pathlib import Path
from brick_classes import BRICK_CLASSES


def generate_yolo_classes(output_dir: str = "training_data") -> None:
    """
    Generate classes.txt file for YOLO training.

    Args:
        output_dir: Directory to save classes.txt (default: "training_data")

    The file format is one class name per line, where line N contains the name
    for class ID N. Empty lines are included for any missing class IDs.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create ordered list of class names by ID
    max_id = max(props['id'] for props in BRICK_CLASSES.values())
    names = [""] * (max_id + 1)

    for name, props in BRICK_CLASSES.items():
        idx = props['id']
        if names[idx]:  # Check for duplicate IDs
            raise ValueError(f"Duplicate class ID {idx} for '{names[idx]}' and '{name}'")
        names[idx] = name

    # Write to file
    classes_file = output_path / "classes.txt"
    with open(classes_file, "w") as f:
        for n in names:
            f.write(n + "\n")

    print(f"✓ Created {classes_file}")
    print(f"  Total classes: {sum(1 for n in names if n)}")


if __name__ == "__main__":
    generate_yolo_classes()
