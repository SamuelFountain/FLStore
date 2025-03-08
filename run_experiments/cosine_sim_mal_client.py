#!/usr/bin/env python3


import subprocess
import csv


def call_curl(url, round):
    try:
        result = subprocess.run(
            ["curl", url, "-d", f"round={round}"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Print the output of curl
        return result.stdout
    except subprocess.CalledProcessError as e:
        # Handle any errors that occur during the curl command
        print(f"Error executing curl command: {e}")


# Example usage
# Example usage
url = "http://0.0.0.0:8080"
model = "resne18"
field_names = [
    "round",
    "Total data fetch time",
    "Total processing time",
    "Total time",
    "Total cost",
    "Total communication cost",
    "Total processing cost",
    "top_k_models",
]


def parse_output(a_str):
    values = (
        [second_str.strip() for second_str in line.split(":")]
        for line in a_str.split("\n")
    )
    values = filter(lambda x: len(x) == 2, values)
    values = dict(values)
    values = {key: values[key] for key in values if key in field_names}
    return values


def main():
    with open(f"cosine_similarity_mal_{model}.csv", "w") as fptr:
        writer = csv.DictWriter(fptr, fieldnames=field_names)
        writer.writeheader()
        for round in range(1, 3):
            res = call_curl(url, round)
            row = parse_output(res)
            row["round"] = round
            writer.writerow(row)
            print(res)


if __name__ == "__main__":
    main()
    print("hi")