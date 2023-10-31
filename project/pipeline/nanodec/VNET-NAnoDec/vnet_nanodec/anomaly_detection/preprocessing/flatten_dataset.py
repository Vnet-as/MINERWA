"""Flatens the dataset structure to make it preprocessable in a single directory.
The script recursively travels through the directory structure and creating symlinks
to flow files with a specified file extension contained within that structure.

Usage:
python source_folder destination_folder
"""

import os
import pathlib
import sys


# File extension for
FILE_EXT = 'flows.gz'


def main(args: list) -> None:
    # Argument check
    if len(args) != 2:
        print("Invalid number of argumnets", file=sys.stderr)
        sys.exit(1)

    # Save arguments to variables for better readability
    dir_src = args[0]
    dir_out = args[1]

    flowfiles = []

    # Create a folder to save the preprocessed flows to if it does not exist
    pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)

    # Obtain filenames of all files within the directory and filter those with specified extension
    for walk_item in os.walk(dir_src):
        # os.walk item is a 3-tuple - (dirpath, dirnames, filenames)
        # Check if a filename ends with determined file extension - if yes, add to output list
        for filename in walk_item[2]:
            if filename.endswith(FILE_EXT):
                flowfiles.append(os.path.join(walk_item[0], filename))

    flowfiles.sort()

    # Create a symbolic link in the output directory for each flowfile
    for flowfile in flowfiles:
        final_symlink_path = os.path.join(dir_out, os.path.basename(flowfile))

        os.symlink(flowfile, final_symlink_path)


if __name__ == '__main__':
    main(sys.argv[1:])
