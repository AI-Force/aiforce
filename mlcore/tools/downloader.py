# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/tools-downloader.ipynb (unless otherwise specified).

__all__ = ['Downloader', 'SaveToDirectory']


# Cell

import errno
import logging
import binascii
import os
import os.path
import io
import requests
from multiprocessing.dummy import Pool as ThreadPool


# Cell


class Downloader:
    """Multi-threaded downloader that can retry a download on failure."""

    _DEFAULT_THREAD_COUNT = 10
    """The default number of threads to use when downloading images."""
    _DEFAULT_RETRIES = 3
    """The number of times to attempt to download the file before giving up."""
    _DEFAULT_TIMEOUT = 30
    """The number of seconds to wait when attempting to create a connection."""

    _SERVER_ERROR_CODE = 500
    """Represents a server error."""
    _FILE_NOT_FOUND = 404
    """Represents that a file could not be found on the remote server."""

    _RANDOM_FILE_NAME_LENGTH = 15
    """Length of the randomly generated filename."""

    def __init__(self, retries=_DEFAULT_RETRIES, thread_count=_DEFAULT_THREAD_COUNT,
                 timeout=_DEFAULT_TIMEOUT):
        """Creates a new Downloaded object with an empty list of downloads."""
        self.tasks = []
        self.retries = retries
        self.thread_count = thread_count
        self.output_paths = []
        self.timeout = timeout
        self.failed_downloads = []
        self.callbacks = []

    def add_callback(self, callback):
        """
        Adds a callback that will be run after the download of the file.

        :param callback: The callback to add.
        """
        self.callbacks.append(callback)

    def _random_file_name(self):
        """Generates a random file name. This is used in cases where the file
        cannot be determined from the URL alone."""
        return binascii.b2a_hex(os.urandom(self._RANDOM_FILE_NAME_LENGTH))

    def _guess_file_name(self, url):
        """Guesses the file name by first attempting to extract the last part
        of the URL, after the slash."""
        file_name = url.rpartition('/')[-1]
        if not file_name:
            file_name = self._random_file_name()
        return file_name

    @staticmethod
    def _get_file_extension(url_part):
        """Returns the file extension of the specified URL."""
        return os.path.splitext(url_part)[-1]

    def add_task(self, url, index):
        """Adds a task to the list of tasks to download."""
        logging.debug("Added Task, URL: %s, INDEX: %d.", (url, index))
        self.tasks.append((url, index))

    def add_tasks(self, urls, indexes):
        """Adds a list of tasks to the list of tasks to download."""
        self.tasks.extend(list(zip(urls, indexes)))

    def _do_download(self, task):
        """Performs the actual download of the file."""
        retry_count = 0
        downloaded = False
        url, index = task
        while not downloaded and retry_count < self.retries:
            try:
                logging.info("Downloading %s.", url)
                r = requests.get(url, stream=True, timeout=self.timeout)
                r.raise_for_status()
                raw_data = io.BytesIO(r.content)
                for callback in self.callbacks:
                    callback.perform(index, raw_data)
                downloaded = True
            except requests.exceptions.HTTPError as e:
                logging.error("Communication occurred during download.")
                logging.exception(e)
                if e.response.status_code == self._FILE_NOT_FOUND:
                    logging.error("Remote file does not exist; skipping.")
                    break
                if e.response.status_code == self._SERVER_ERROR_CODE:
                    retry_count += 1
                    logging.warning("Retry attempt %d", retry_count)
                    continue
                else:
                    logging.error("Unhandled HTTP error occurred during download.")
                    logging.exception(e)
                    break
            except Exception as e:
                logging.error("Unknown error occurred during download.")
                logging.exception(e)
                break
        if not downloaded:
            self.failed_downloads.append(task)

    def download(self):
        """Downloads the files to the specified location."""
        pool = ThreadPool(self.thread_count)
        pool.map(self._do_download, self.tasks)
        pool.close()
        pool.join()

    def clear(self):
        """Removes all of the downloads from the list of downloads."""
        del self.tasks[:]
        del self.failed_downloads[:]

    def get_failed_downloads(self):
        """Returns a list of downloads that failed to complete."""
        return self.failed_downloads


# Cell


class SaveToDirectory:
    """Callback to download file to a specific directory."""
    def __init__(self, file_path):
        """
        Initializes the object to save to a specific directory.

        :param file_path:  The base path to save the file to.
        """
        self.path = file_path
        self._make_path()

    def _make_path(self):
        """
        Ensures a destination path exists by creating it.
        """
        try:
            os.makedirs(self.path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(self.path):
                pass
            else:
                raise

    def perform(self, filename, data):
        """
        Saves the data to the specified file.

        :param filename: The filename to write the file to.
        :param data: The data to save as a BytesIO object.
        """
        save_path = os.path.join(self.path, filename)
        with open(save_path, "wb") as f:
            f.write(data.getvalue())