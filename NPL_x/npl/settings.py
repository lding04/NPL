import configparser
from pathlib import Path
import sys

OUTPUT_PATH = None
INPUT_FILE = None
COLUMN_FILE = None
DICT_FILE = None
conf = configparser.ConfigParser()


def init_vars():
    global OUTPUT_PATH
    global INPUT_FILE
    global COLUMN_FILE
    global DICT_FILE
    if sys.executable.endswith('python.exe'):
        npl_home = Path('.')
    else:
        npl_home = Path(sys.executable).parent.parent
    config_path = npl_home/'npl'/'config'
    conf.read(config_path/'npl.cfg')
    INPUT_FILE = conf.get('core', 'input_file')
    OUTPUT_PATH = conf.get('core', 'output_path')
    COLUMN_FILE = config_path/'columns.csv'
    DICT_FILE = config_path/'dictionary.conf'
