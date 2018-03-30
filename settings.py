# -*- encoding: utf-8 -*-
import os

# Constants
# =========
class BaseConfig(object):
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))

    UPLOAD_DIR = 'uploads'
    RESULT_DIR = 'results'

    TOMITA_BIN_PATH = '\\'.join([APP_ROOT, 'tomita', 'tomitaparser.exe'])
    TOMITA_CONFIG_PATH = '\\'.join([APP_ROOT, 'tomita', 'config', 'config.proto'])

    PUBLIC_MODEL_URL = 'http://127.0.0.1:5001/ml-models'