import os
import glob
import zipfile


DEFAULT_EXT_LST = ('py', 'json', 'yaml', 'cu', 'cpp', 'h', 'sh', 'md', 'txt')
DEFAULT_EXCLUDE = ('outputs',)

def snapshot_files_list(out_path, ext_lst=DEFAULT_EXT_LST, exclude=DEFAULT_EXCLUDE):
    total_lst = sum(
        [glob.glob('**/*.' + ext, recursive=True) for ext in ext_lst], start=[])

    excluded = tuple(os.path.normpath(d) + os.sep for d in exclude)
    total_lst = [
        f for f in total_lst if not os.path.normpath(f).startswith(excluded)
    ]

    with zipfile.ZipFile(out_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for f in total_lst:
            zf.write(f)
