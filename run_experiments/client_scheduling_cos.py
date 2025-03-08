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
url = "http://0.0.0.0:8080"
model = "resnet18"
field_names = [
    "Total data fetch time",
    "Total processing time",
    "Total time",
    "Total cost",
    "Total communication cost",
    "Total processing cost",
    "top_k_models",
]


def parse_output(a_str):
    values = [
        [second_str.strip() for second_str in line.split(":")]
        for line in a_str.split("\n")
    ]
    print(values)
    values = dict(values)
    return values


def main():
    with open(f"client_scheduling_cos_{model}.csv", "w") as fptr:
        writer = csv.DictWriter(fptr, fieldnames=field_names)
        writer.writeheader()
        for round in range(1, 3):
            res = call_curl(url, round)
            writer.writerow(parse_output(res))


if __name__ == "__main__":
    main()
