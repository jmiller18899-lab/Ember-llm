"""Opt-in v4 routing plus the unchanged v3 argument parser."""
from pathlib import Path
import torch

from .routing_data_v4 import LABELS
from .routing_v4 import predict, validate_request
from .runtime import hidden, prompt
from .runtime_v3 import ParserRuntime


class RoutingRuntime(ParserRuntime):
    def __init__(self, bundle, precision="int4"):
        super().__init__(bundle, precision)
        self.routing = torch.load(Path(bundle) / f"routing-v4-{precision}.pt",
                                  map_location="cpu", weights_only=True)
        if self.routing.get("labels") != list(LABELS):
            raise ValueError("Unknown routing labels")

    def route(self, user):
        validate_request(user)
        x = hidden(self.model, self.tokenizer, prompt(user)).unsqueeze(0)
        p, _, margin = predict(self.routing, [user], x)
        return LABELS[int(p[0])], float(margin[0])
