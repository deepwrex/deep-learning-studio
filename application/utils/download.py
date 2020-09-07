import os
import tensorflow as tf


class DatasetDownloader:

    @staticmethod
    def download_from_url(url: str):
        file_extention = url.split('.')[-1]
        file_name = url.split('/')[-1].split('.')[0]
        download_path = tf.keras.utils.get_file(
            file_name + '.' + file_extention, origin=url,
            extract=True if file_extention == 'zip' else False, archive_format='auto',
            untar=True if file_extention == 'tar' or file_extention == 'tgz' else False
        )
        download_path = os.path.join(os.path.dirname(download_path), file_name)
        return download_path
