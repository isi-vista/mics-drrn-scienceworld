from argparse import ArgumentParser
from pathlib import Path
import csv


SLURM_FILE = "slurm"
TENSORBOARD_CSV = "tb_csv"
FILE_TYPES = [SLURM_FILE, TENSORBOARD_CSV]


def main() -> None:
    parser = ArgumentParser(__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input file")
    parser.add_argument("--type", choices=FILE_TYPES, type=str, required=True, help="Type of file being passed for reading")
    args = parser.parse_args()

    input_path: Path = args.input
    if input_path.is_file():
        input_files = [input_path]
    elif input_path.is_dir():
        input_files = [file for file in input_path.iterdir() if file.suffix == ".csv" or file.suffix == ".out"]
    else:
        raise FileNotFoundError(f"Could not find file at path: {input_path}")
    file_type: str = args.type

    scores = []
    for input_file in input_files:
        if file_type == SLURM_FILE:
            for line in input_file.read_text().splitlines():
                if line.startswith("EVAL EPISODE SCORE: ") and "STEPS" not in line:
                    score = int(line.split(":")[1].strip())
                    scores.append(score if score >= 0 else 0)
        elif file_type == TENSORBOARD_CSV:
            with input_file.open("r") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    scores.append(float(row["EvalScore"]) if float(row["EvalScore"]) >= 0 else 0)
        else:
            raise NotImplementedError("No other file types are supported")

    avg_score = sum(scores) / len(scores)
    last_ten_percent = len(scores) // 10 * 9
    scienceworld_avg_score = sum(scores[last_ten_percent:]) / (len(scores) - last_ten_percent)

    print(f"Full Average Score: {avg_score}")
    print(
        f"ScienceWorld (10%) Average Score (Useless if a directory is passed): "
        f"{scienceworld_avg_score}; {len(scores)=} {last_ten_percent=}"
    )


if __name__ == "__main__":
    main()
