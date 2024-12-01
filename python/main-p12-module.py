import sys
import test_import
from test_import import test_import_1
from test_import import *
from test_import_file.app import app
from test_import_file import module_name
import pprint

print(test_import)
print(sys.path)
print(dir(test_import))
print(test_import_1)
print(globals())

print(__name__)



app()

print(module_name)
pprint.pprint(sys.modules)
