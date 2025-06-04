"""Model store which provides pretrained models."""
from __future__ import print_function

import os
# import zipfile

from ..utils import check_sha1

__all__ = ['get_model_file', 'purge']

_custom_model_map = {
    'resnet50': 'resnet50-ebb6acbb.pth'  # 直接映射你的本地文件名
}

_model_sha1 = {name: checksum for checksum, name in [
    ('ebb6acbbd1d1c90b7f446ae59d30bf70c74febc1', 'resnet50'),
    ('2a57e44de9c853fa015b172309a1ee7e2d0e4e2a', 'resnet101'),
    ('0d43d698c66aceaa2bc0309f55efdd7ff4b143af', 'resnet152'),
    ('662e979de25a389f11c65e9f1df7e06c2c356381', 'fcn_resnet50_ade'),
    ('eeed8e582f0fdccdba8579e7490570adc6d85c7c', 'fcn_resnet50_pcontext'),
    ('54f70c772505064e30efd1ddd3a14e1759faa363', 'psp_resnet50_ade'),
    ('075195c5237b778c718fd73ceddfa1376c18dfd0', 'deeplab_resnet50_ade'),
    ('5ee47ee28b480cc781a195d13b5806d5bbc616bf', 'encnet_resnet101_coco'),
    ('4de91d5922d4d3264f678b663f874da72e82db00', 'encnet_resnet50_pcontext'),
    ('9f27ea13d514d7010e59988341bcbd4140fcc33d', 'encnet_resnet101_pcontext'),
    ('07ac287cd77e53ea583f37454e17d30ce1509a4a', 'encnet_resnet50_ade'),
    ('3f54fa3b67bac7619cd9b3673f5c8227cf8f4718', 'encnet_resnet101_ade'),
    ]}


# def short_hash(name):
#     if name not in _model_sha1:
#         raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
#     return _model_sha1[name][:8]

def get_model_file(name, root=os.path.join('~', '.encoding', 'models')):
    r"""强制使用本地模型文件，禁止下载"""
    root = os.path.expanduser(root)
    # local_model_path = os.path.join(root, 'resnet50-19c8e357.pth')
    
    if name == 'resnet50':
        custom_file = _custom_model_map[name]
        custom_path = os.path.join(root, custom_file)
    #     # 如果本地文件存在且名称匹配，则直接返回本地文件路径
    #     print(f"Using local pretrained weights: {local_model_path}")
    #     return local_model_path
    
    # file_name = f'{name}-{short_hash(name)}.pth'
    # root = os.path.expanduser(root)
    # file_path = os.path.join(root, file_name)
    # sha1_hash = _model_sha1[name]
    # if os.path.exists(custom_path):
    #     if check_sha1(file_path, sha1_hash):
    #         return file_path
    #     else:
    #             raise ValueError(f'Local model {file_path} hash mismatch. '
    #                             'Please manually download the correct model.')
    # else:
    #         raise FileNotFoundError(
    #             f'Model {file_name} not found in {root}. '
    #             'You must manually place the pretrained model in this directory.')

        if os.path.exists(custom_path):
            print(f"Using custom pretrained weights: {custom_path}")
            return custom_path  # 直接返回，跳过SHA1检查
        else:
            raise FileNotFoundError(
                    f"Required model file '{custom_file}' not found in:\n{root}\n"
                    "请将您的本地文件 resnet50-19c8e357.pth 放置到上述目录"
                )
        
        # 其他模型保持原校验逻辑
    expected_hash = _model_sha1[name][:8]
    file_name = f'{name}-{expected_hash}.pth'
    file_path = os.path.join(root, file_name)
        
    if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Model {file_name} not found in {root}\n"
                "请手动下载并放置模型文件到该目录"
            )
        
    if not check_sha1(file_path, _model_sha1[name]):
            raise ValueError(
                f"SHA1 mismatch for {file_name}\n"
                "文件可能已损坏，请重新下载正确版本"
            )
        
    return file_path
    
    #     else:
    #         print('Mismatch in the content of model file {} detected.' +
    #               ' Downloading again.'.format(file_path))
    # else:
    #     print('Model file {} is not found. Downloading.'.format(file_path))

    # if not os.path.exists(root):
    #     os.makedirs(root)

    # zip_file_path = os.path.join(root, file_name+'.zip')
    # repo_url = os.environ.get('ENCODING_REPO', encoding_repo_url)
    # if repo_url[-1] != '/':
    #     repo_url = repo_url + '/'
    # download(_url_format.format(repo_url=repo_url, file_name=file_name),
    #          path=zip_file_path,
    #          overwrite=True)
    # with zipfile.ZipFile(zip_file_path) as zf:
    #     zf.extractall(root)
    # os.remove(zip_file_path)

    # if check_sha1(file_path, sha1_hash):
    #     return file_path
    # else:
    #     raise ValueError('Downloaded file has different hash. Please try again.')

def purge(root=os.path.join('~', '.encoding', 'models')):
    r"""Purge all pretrained model files in local file store.

    Parameters
    ----------
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    """
    root = os.path.expanduser(root)
    files = os.listdir(root)
    for f in files:
        if f.endswith(".pth"):
            os.remove(os.path.join(root, f))

def pretrained_model_list():
    return list(_model_sha1.keys())
