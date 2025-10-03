import sys
import torch
import numpy as np
import flask
import tqdm
import platform
from packaging import version
import PIL

if __name__ == '__main__':
    # 检查各库版本
    print("python 版本：",sys.version)
    print("pytorch 版本：", torch.__version__)
    print("NumPy 版本：", np.__version__)
    print("Flask 版本：", flask.__version__)
    print("tqdm版本：", tqdm.__version__)
    print("PIL版本：", PIL.__version__)

    # 以下检查不强行需求
    if version.parse(platform.python_version()) < version.parse("3.9"):
        raise ValueError("python 版本不匹配")
    else:
        print("python 版本符合要求")

    # 检查 pytorch 版本
    if version.parse(torch.__version__) < version.parse("2.7.1+cu118"):
        raise ValueError("torch 版本不匹配")
    else:
        print("torch 版本符合要求")

    # 检查 numpy 版本
    if version.parse(np.__version__) < version.parse("1.23.5"):
        raise ValueError("numpy 版本不匹配")
    else:
        print("numpy 版本符合要求")

    # 检查 flask 版本
    if version.parse(flask.__version__) < version.parse("3.1.2"):
        raise ValueError("flask 版本不匹配")
    else:
        print("flask 版本符合要求")

    # 检查 tqdm 版本
    if version.parse(tqdm.__version__) < version.parse("4.67.1"):
        raise ValueError("tqdm 版本不匹配")
    else:
        print("tqdm 版本符合要求")

    # 检查 PIL 版本
    if version.parse(PIL.__version__) < version.parse("9.4.0"):
        raise ValueError("PIL 版本不匹配")
    else:
        print("PIL 版本符合要求")