import os
import pathlib
import random
import string


def generate_random_string(length: int = 16):
    """
    Generate a random string of fixed length using alphanumeric characters.

    Args:
        length: length of the generated string

    Returns: random string of given length

    """
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_random_filename(instance, filename: str, subdir: str = "other") -> str:
    """
    Generate a short random string for the uploaded file name while keeping the extension.

    Use like this:

    image = models.ImageField(
        upload_to=partial(generate_random_filename, subdir="avatars")
    )

    Args:
        instance: instance of the corresponding model
        filename: name of the file to be uploaded
        subdir: subdirectory to store the file

    Returns: generated path for the file

    """
    ext: str = pathlib.Path(filename).suffix
    random_name = f"{generate_random_string()}{ext}"

    try:
        user_id = f"user_{instance.user.pk}"
    except AttributeError:
        raise ValueError(
            "The instance must have a user attribute to generate a filename."
        )

    return os.path.join("uploads", user_id, subdir, random_name)
