import os
import re
import pandas as pd


def save_df_to_dir(results_df, base_dir, sub_dirs, file_name_format, add_context, model_name):
    # Get the root directory of the project
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the output directory path
    output_dir = os.path.join(root_dir, base_dir, *sub_dirs)
    os.makedirs(output_dir, exist_ok=True)

    # Construct the file name
    file_name = file_name_format.format(model_name=model_name,
                                        context="with_context" if add_context else "without_context")

    # Construct the full file path
    file_path = os.path.join(output_dir, file_name)

    # Save the DataFrame to CSV
    results_df.to_csv(file_path, index=False)


def merge_dfs(base_dir, exp_name, part_format="part_{i}_", output_dir=None,
                     filename="patchscopes_results.parquet", output_filename="patchscopes_results.parquet"):
    """
    Merges DataFrames from directories matching the part format into a single DataFrame,
    and optionally saves the result to a file.

    Args:
        base_dir (str): The base directory containing the data.
        exp_name (str): The experiment name to look for within part directories.
        part_format (str): The general format for identifying parts (e.g., "part_{i}_").
        output_dir (str, optional): Directory to save the merged DataFrame. Default is None.
        filename (str): The filename of the Parquet file to read in each part directory.
        output_filename (str): Name of the output file if saving is enabled.

    Returns:
        pd.DataFrame: A single DataFrame containing data from all parts.
    """
    dataframes = []
    part_regex = part_format.replace("{i}", r"\d+")

    # List all directories in base_dir
    for dir_name in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, dir_name)) and re.match(part_regex, dir_name) and (dir_name.endswith(exp_name)):
            part_dir = os.path.join(base_dir, dir_name)
            file_path = os.path.join(part_dir, filename)

            if os.path.exists(file_path):
                # Read the DataFrame and add it to the list
                df = pd.read_parquet(file_path)
                dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(dataframes, axis=1)

    # Save the result to file if output_dir is given
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        merged_df.to_parquet(output_path, index=False)

    return merged_df, dataframes


def parse_string_list_from_file(file_path, delimiter=None):
    """
    Parses a list of strings from a file, handling various list formats.

    Args:
        file_path (str): Path to the file containing the list.

    Returns:
        list: A list of parsed strings.
    """
    with open(file_path, 'r') as file:
        content = file.read()

    if delimiter is None:
        # Remove newlines and excess whitespace
        content = re.sub(r'\s+', ' ', content.strip())

        # Handle different delimiters and list formats
        # Removes common list notations like commas, brackets, quotes, etc.
        items = re.split(r'[,\[\]\(\)\{\}"\'\s]+', content)
    else:
        if delimiter == "newline":  # TODO fix this
            delimiter = "\n"
        items = [item.strip() for item in content.split(delimiter)]

    # Filter out any empty strings from the list
    return [item for item in items if item]