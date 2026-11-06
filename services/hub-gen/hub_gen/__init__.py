"""Rust-Hub's local 3D generation service.

Runs open-weight models (Hunyuan3D 2, TripoSG) on your own GPU behind a small
HTTP API that the hub's ``Generate`` command calls, so no model is billed.
"""

__version__ = "0.1.0"
