import os


def move_decimal_point(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            columns = line.strip().split(',')
            if len(columns) >= 3:
                try:
                    value = float(columns[2])
                    new_value = int(value * 1e9)
                    columns[2] = str(new_value)
                except ValueError:
                    print(f"Skipping non-numeric value in line: {line}")

            new_line = ','.join(columns) + '\n'
            file.write(new_line)


def process_folder(input_folder):
    txt_files = [file for file in os.listdir(input_folder) if file.endswith('.txt')]
    file_paths = [os.path.join(input_folder, file) for file in txt_files]

    for file_path in file_paths:
        move_decimal_point(file_path)


if __name__ == "__main__":
    input_folder_path = '/media/admin123/T7/ceshi/output'

    process_folder(input_folder_path)
