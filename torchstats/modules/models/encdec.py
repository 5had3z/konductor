from typing import Any, Dict
from dataclasses import dataclass

from torch import nn

from . import MODEL_REGISTRY, ENCODER_REGISTRY, DECODER_REGISTRY, POSTPROCESSOR_REGISTRY


@dataclass
class EncoderDecoderConfig:
    encoder: Any
    decoder: Any
    postproc: Any | None = None


@MODEL_REGISTRY.register_module()
class EncoderDecoder(nn.Module):
    def __init__(self, config: EncoderDecoderConfig) -> None:
        super().__init__()
        self.encoder = ENCODER_REGISTRY[config.encoder.name](config.encoder)
        self.decoder = DECODER_REGISTRY[config.decoder.name](config.decoder)
        if config.postproc is not None:
            self.postproc = POSTPROCESSOR_REGISTRY[config.postproc.name](
                config.postproc
            )
        else:
            self.postproc = None

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """"""
        x = self.encoder(inputs)
        out = self.decoder(x)
        if self.postproc is not None:
            out = self.postproc(out)

        return out
