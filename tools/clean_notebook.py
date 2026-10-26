"""Git clean filter: omit execution outputs, preserving local executed notebooks."""
import json
import sys


def clean(notebook):
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
        for key in ('execution', 'ExecuteTime', 'trusted', 'collapsed', 'scrolled'):
            cell.get('metadata', {}).pop(key, None)
    metadata = notebook.setdefault('metadata', {})
    for key in ('local_execution', 'widgets', 'signature'):
        metadata.pop(key, None)
    if 'language_info' in metadata:
        metadata['language_info'] = {'name': metadata['language_info'].get('name', 'python')}
    return notebook


if __name__ == '__main__':
    json.dump(clean(json.load(sys.stdin)), sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write('\n')
