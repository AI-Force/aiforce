import logging
import logging.handlers
import argparse
import sys

from dataclasses import dataclass
from os import listdir, remove
from os.path import join, isfile, isdir
from typing import List
from ..io.core import scan_files


logger = logging.getLogger(__name__)

@dataclass
class FileScanner:
    """
    Scans a given folder and subfolders for files.
    `path`: The file extensions to handle
    `exclude`: if True, exclude files in file extenstions
    """
    path: str
    file_extensions: List[str] = None
    exclude: bool = False

    def scan(self):
        """
        The main scan logic.
        """
        files = scan_files(self.path, self.file_extensions, self.exclude)
        return files


# Cell
def configure_logging(logging_level=logging.INFO):
    """
    Configures logging for the system.

    :param logging_level: The logging level to use.
    """
    logger.setLevel(logging_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging_level)

    logger.addHandler(handler)


# Cell
if __name__ == '__main__' and '__file__' in globals():
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="The folder to scan.")
    parser.add_argument("--file_extensions", nargs="+", 
                        default=None, help='The file extensions to list.')
    parser.add_argument("--exclude",
                        help="List all files not matching the file extensions.",
                        default=False,
                        action="store_true")
    args = parser.parse_args()

    scanner = FileScanner(args.folder, args.file_extensions, args.exclude)
    files = scanner.scan()

    all_files = len(files)

    for index, file in enumerate(files):
        logger.info(f"{index + 1} / {all_files} - {file}")

    logger.info('FINISHED!!!')
