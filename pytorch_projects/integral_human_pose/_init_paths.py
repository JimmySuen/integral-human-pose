import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)
add_path(os.path.join(this_dir, '..', '..', 'common'))
add_path(os.path.join(this_dir, '..', 'common_pytorch'))
add_path(os.path.join(this_dir, '..', '..'))
add_path(os.path.join(this_dir, '..'))
add_path(os.path.join(this_dir))

print("=================SYS PATH================\n")
for path in sys.path:
    print(path)
print("\n=================SYS PATH================")
