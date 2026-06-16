
from huggingface_hub.inference._providers import provider_mapping
from huggingface_hub import HfApi
import os
print("HUGGINGFACEHUB_API_TOKEN set:", bool(os.getenv("HUGGINGFACEHUB_API_TOKEN")))
print("provider_mapping keys:", list(provider_mapping.keys()))
print("provider_mapping reprs:", [repr(v) for v in provider_mapping.values()])
try:
    print("whoami:", HfApi().whoami())
except Exception as e:
    print("whoami failed:", repr(e))