import os
import sys


DEFAULT_GPU_ID = "2"


def cli_gpu_id(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    for index, arg in enumerate(argv):
        if arg == "--gpu_id" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--gpu_id="):
            return arg.split("=", 1)[1]
    return None


def configure_cuda_visible_devices(argv=None):
    gpu_id = cli_gpu_id(argv)
    if gpu_id is None:
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", DEFAULT_GPU_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return os.environ["CUDA_VISIBLE_DEVICES"]
