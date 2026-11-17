"""Small, safe YAML project manifest; paths are always relative to the input root."""
import yaml

ALLOWED = {'title', 'credit', 'source', 'page', 'margin', 'gap', 'rows', 'split', 'order', 'skip'}


def read_manifest(text):
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError('tabsheet.yaml must be a mapping')
    unknown = set(data) - ALLOWED
    if unknown:
        raise ValueError(f'unsupported manifest fields: {sorted(unknown)}')
    for field in ('order', 'skip'):
        if field in data and (not isinstance(data[field], list) or
                              not all(isinstance(x, str) for x in data[field])):
            raise ValueError(f'manifest {field} must be a list of image names')
    return data


def apply_order(names, metadata, order):
    skip = set(metadata.get('skip', []))
    unknown = skip - set(names)
    if unknown:
        raise ValueError(f'manifest skip names missing: {sorted(unknown)}')
    if order == 'manifest' or (metadata.get('order') and order is None):
        listed = metadata.get('order')
        if not listed:
            raise ValueError('--order manifest requires an order list in tabsheet.yaml')
        if len(set(listed)) != len(listed):
            raise ValueError('duplicate names in manifest order')
        unknown = set(listed) - set(names)
        omitted = set(names) - set(listed) - skip
        if unknown or omitted:
            raise ValueError(f'manifest order mismatch; missing files={sorted(unknown)}, '
                             f'unlisted files={sorted(omitted)}')
        names = listed
    return [name for name in names if name not in skip]
