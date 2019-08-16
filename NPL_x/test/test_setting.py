from npl import settings


def test_vars():
    print('default input file: {}'.format(settings.INPUT_FILE))
    print('default output path: {}'.format(settings.OUTPUT_PATH))
