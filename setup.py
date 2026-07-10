from setuptools import setup, find_packages

# Backward-compatibility shims kept at the repo root so that pickled genomes /
# checkpoints (which store the original top-level module paths) still load, and
# so that ``import rendering`` keeps working for external notebooks. See each
# shim's docstring; the real code lives under picbreeder_vlm/.
_COMPAT_SHIMS = [
    "neat_components",
    "config",
    "picbreeder_reproduction",
    "archive_manager",
    "rendering",
]

setup(
    name="picbreeder-vlm",
    version="0.1.0",
    description=(
        "In Search of the Ingredients of Open-Endedness: "
        "Replicating Picbreeder with Large Vision-Language Models"
    ),
    packages=find_packages(include=["picbreeder_vlm", "picbreeder_vlm.*"]),
    py_modules=_COMPAT_SHIMS,
    # The NEAT preset lives beside the picture2d renderer that reads it. It has no
    # suffix, so no wildcard would pick it up; name it or a wheel install ships a
    # package whose only CPPN config is missing.
    package_data={"picbreeder_vlm.core": ["interactive_config_color"]},
    extras_require={"test": ["pytest"]},
)
