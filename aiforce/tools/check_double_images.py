# AUTOGENERATED! DO NOT EDIT! File to edit: tools-check_double_images.ipynb (unless otherwise specified).

__all__ = ['logger', 'BUF_SIZE', 'FILE_FILTER', 'IMAGE_EXTENSION', 'DoubleFileChecker', 'configure_logging',
           'get_file_sha', 'check_double_entries', 'remove_entries', 'scan_folder']


# Cell
import hashlib
import logging
import logging.handlers
import argparse
import sys
from os import listdir, remove
from os.path import join, isfile, isdir


# Cell
logger = logging.getLogger(__name__)


# Cell
BUF_SIZE = 65536  # lets read stuff in 64kb chunks!
FILE_FILTER = ['.DS_Store']
IMAGE_EXTENSION = '.jpg'


# Cell
class DoubleFileChecker:
    """
    Checks a given folder and subfolders for double files by calculating the corresponding SHA1.
    `path`: the folder to process
    `reverse`: if True, order the file reverse
    """

    def __init__(self, path, reverse=False):
        self.path = path
        self.reverse = reverse

    def check(self):
        """
        The main validation logic.
        """

        images = scan_folder(self.path)
        images.sort(key=len, reverse=self.reverse)
        all_images = len(images)
        contents = []
        delete_entries = []

        for index, image in enumerate(images):
            logger.info("{} / {} - Handle Image {}".format(index + 1, all_images, image))
            content = (image, get_file_sha(image))
            contents.append(content)

        # Check double entries
        logger.info('Checking Double Entries:')
        double_entries = check_double_entries(contents)

        logger.info("Found {} Entries:".format(len(double_entries)))
        for (key, entrylist) in double_entries.items():
            logger.info("{}:".format(key))
            for entry in list(entrylist):
                logger.info("-> {}".format(entry[0]))

        if len(double_entries) > 0:
            logger.info('Using only the first of each double entries.')
            delete_files = input('Will you delete the ignored double files from source? (y/n) ')
            delete_files = delete_files == 'y'
            for (key, entry) in double_entries.items():
                entry = list(entry)
                delete_entries += entry[1:]
            remove_entries(contents, delete_entries, delete_files)
        return contents, delete_entries


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
def get_file_sha(fname):
    """
    Calculates the SHA1 of a given file.
    `fname`: the file path
    return: the calculated SHA1 as hex
    """

    result = ''
    if isfile(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                sha1.update(data)
            result = sha1.hexdigest()
    return result


# Cell
def check_double_entries(entries):
    """
    Process a list of tuples of filenames and corresponding hash for double hashes.
    `entries`: the list of entries with their hashes
    returns: a dictionary containing double entries by hash
    """

    hashes = dict()
    for entry in entries:
        h = entry[1]
        if h not in hashes:
            hashes[h] = []
        hashes[h].append(entry)

    double_hashes = dict()
    for (key, entrylist) in hashes.items():
        if len(entrylist) > 1:
            double_hashes[key] = entrylist
    return double_hashes


# Cell
def remove_entries(entries, to_remove, delete_source=False):
    """
    Removes entries from list and optionally remove the source file as well.
    `entries`: the list of entries to remove from
    `to_remove`: the list of entries to remove
    `delete_source`: werether or not to delete the source file as well
    returns: a list of resulting entries
    """

    for entry in to_remove:
        if entry in entries:
            index = entries.index(entry)
            logger.info("Remove Entry: ({}) {}".format(index, entry[0]))
            del entries[index]
            if delete_source:
                logger.info("Delete Source File: ".format(entry[0]))
                remove(entry[0])

    return entries


# Cell
def scan_folder(folder):
    """
    Scans a folder and subfolders for image content.
    `folder`: the folder to scan
    returns: a list of paths to images found
    """
    images = []

    contents = listdir(folder)
    for content in contents:
        if content in FILE_FILTER:
            continue
        path = join(folder, content)
        if isdir(path):
            result = scan_folder(path)
            for entry in result:
                images.append(entry)
        elif isfile(path):
            images.append(path)
    return images


# Cell
if __name__ == '__main__' and '__file__' in globals():
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="The folder to scan.")
    parser.add_argument("--reverse-sort",
                        help="If the double entries should be sorted reverse by length.",
                        default=False,
                        action="store_true")
    args = parser.parse_args()

    checker = DoubleFileChecker(args.folder, args.reverse_sort)
    checker.check()

    logger.info('FINISHED!!!')