import os

def calculate_average(file_paths):
    total_lines = 0
    line_sums = []

    # Initialize line_sums list
    with open(file_paths[0], 'r') as first_file:
        first_lines = first_file.readlines()
        total_lines = len(first_lines)
        line_sums = [0.0] * total_lines

    # Accumulate values from each file
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()

            if len(lines) != total_lines:
                print(f"Error: File {file_path} has a different number of lines than expected.")
                return None

            for idx, line in enumerate(lines):
                try:
                    value = float(line.strip().split(',')[-1])
                    line_sums[idx] += value
                except ValueError:
                    print(f"Skipping non-numeric line in file {file_path}: {line}")

    # Calculate averages
    averages = [line_sum / len(file_paths) for line_sum in line_sums]
    return averages

def process_folder(input_folder, output_file):
    txt_files = [file for file in os.listdir(input_folder) if file.endswith('.txt')]
    file_paths = [os.path.join(input_folder, file) for file in txt_files]

    averages = calculate_average(file_paths)

    if averages is not None:
        with open(output_file, 'w') as output:
            for idx, average in enumerate(averages):
                output.write(f"Line {idx + 1} Average: {average:.4f}\n")

if __name__ == "__main__":
    input_folder_path = '/media/admin123/T7/ceshi/output'
    output_file_path = '/media/admin123/T7/ceshi/output/summary.txt'

    process_folder(input_folder_path, output_file_path)
