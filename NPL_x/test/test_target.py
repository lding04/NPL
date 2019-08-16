import pytest
from npl import crawl


def test_target():
    target = crawl.Target('test', '定海区解放西路135号内街商场130号')
    target.output('test')