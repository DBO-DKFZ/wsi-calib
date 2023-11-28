import os
from pathlib import Path


def read_txt(file_p: Path) -> list:
    with open(file_p, "r") as f:
        lines = f.readlines()
    cleaned_lines = [line.rstrip() for line in lines]  # Remove newline characters
    unique_elements = sorted(list(set(cleaned_lines)))
    print(unique_elements)
    return unique_elements


def write_txt(list: list, out_p: Path):
    with open(out_p, "w") as f:
        for item in list:
            f.write(item + "\n")


if __name__ == "__main__":
    in_p = Path("/mnt/ssd/Data/MCO-SCalib/broken_slides.txt")
    out_p = Path("/mnt/ssd/Data/MCO-SCalib/broken_slides_unique.txt")
    output = read_txt(in_p)
    write_txt(output, out_p)
