"""Model backends. Heavy libraries (torch, diffusers, hy3dgen) are imported only
when a model is first used, so the service starts, and its tests run, without them."""
