"""Visual Expert training and evaluation components."""

def __getattr__(name):
    # Metadata tools do not require importing PyTorch or OpenCLIP.
    if name == 'CLIPBinaryDetector':
        from .model import CLIPBinaryDetector
        return CLIPBinaryDetector
    if name in ('DOMAIN_TO_ID', 'ManifestDataset', 'load_manifest'):
        from . import data
        return getattr(data, name)
    raise AttributeError(name)

__all__ = [
    "CLIPBinaryDetector",
    "DOMAIN_TO_ID",
    "ManifestDataset",
    "load_manifest",
]
