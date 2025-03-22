import hashlib
from pathlib import Path


class UtilFilehandling:
    """
    Various small helpers to deal with files
    """

    @staticmethod
    def generate_filename_hash_from_parameters(**kwargs) -> str:
        """
        Create a filename from the request parameters.

        We will read all parametervalues, convert to string and concatenate them.
        Then we will hash the resulting string and return the hash.

        :param kwargs: any number of named arguments
        :return: filename
        """
        filename = ""
        for key, value in kwargs.items():
            filename += str(value)

        return UtilFilehandling.create_hash_from_string(filename)

    @staticmethod
    def create_hash_from_string(string: str, length: int = 50) -> str:
        """
        Create a hash from a string. For a given parameter string the hash will always have the same value.
        :param string: string to hash
        :param length: length of the hash
        :return: hash
        """
        sha256_hash = hashlib.sha256(string.encode('utf-8')).hexdigest()
        return sha256_hash[0:length]

    @staticmethod
    def get_latest_file_in_folder(path: str, extension: str, mask: str = "*.") -> str:
        """
        Get the latest file in a folder respecting filetype/extension
        :param path: path to the folder
        :param extension: file extension, e.g. "pkl"
        :param mask: mask for the file, e.g. "confluence_pages_*.
        :return: latest file
        """
        if not mask[-1] == ".":
            mask += "."

        files = [f for f in Path(path).glob(f"{mask}{extension}")]
        if not files:
            return ""
        return str(max(files, key=lambda f: f.stat().st_mtime))
