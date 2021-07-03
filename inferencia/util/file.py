import os
import os.path as osp
import urllib
from google_drive_downloader import GoogleDriveDownloader as gdd


def download(url, save_path):
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlretrieve(url, save_path)


def download_from_google_drive(url, save_path):
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    gdd.download_file_from_google_drive(file_id=url,
                                        dest_path=save_path)


def get_model_path(task_major_name,
                   task_minor_name,
                   model_name,
                   model_detail_name,
                   model_precision):
    model_dir = osp.join(osp.expanduser("~"),
                         '.inferencia',
                         task_major_name,
                         task_minor_name,
                         model_name,
                         model_detail_name,
                         model_precision)
    model_save_name = '{model_name}-{model_detail_name}-{model_precision}.onnx'.format(model_name=model_name,
                                                                                       model_detail_name=model_detail_name,
                                                                                       model_precision=model_precision)
    model_path = osp.join(model_dir, model_save_name)
    return model_path
