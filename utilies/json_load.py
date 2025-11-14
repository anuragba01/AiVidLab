import json
import os


def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    fd = os.open(config_path, os.O_RDONLY)
    try:
        raw = os.read(fd, os.path.getsize(config_path)).decode("utf-8")
        return json.loads(raw)
    finally:
        os.close(fd)




def load_input(input_path="input.json"):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Config file not found: {input_path}")

    fd = os.open(input_path, os.O_RDONLY)
    try:
        raw = os.read(fd, os.path.getsize(input_path)).decode("utf-8")
        return json.loads(raw)
    finally:
        os.close(fd)


