from typing import Any, Dict
from dataclasses import dataclass

from torch import nn

from . import MODEL_REGISTRY, ENCODER_REGISTRY, DECODER_REGISTRY, POSTPROCESSOR_REGISTRY


@dataclass
@MODEL_REGISTRY.register_module("EncoderDecoder")
class EncoderDecoderConfig:
    encoder: Any
    decoder: Any
    postproc: Any | None = None

    @classmethod
    def from_config(cls, config):
        """"""
        model_args = config["model"]["args"]
        encoder = ENCODER_REGISTRY[model_args["encoder"]["name"]].from_config(config)
        decoder = DECODER_REGISTRY[model_args["decoder"]["name"]].from_config(config)

        if "postproc" in model_args:
            postproc = POSTPROCESSOR_REGISTRY[
                model_args["postproc"]["name"]
            ].from_config(config)
        else:
            postproc = None

        return cls(encoder, decoder, postproc)

    def get_instance(self) -> nn.Module:
        return EncoderDecoder(
            self.encoder.get_instance(),
            self.decoder.get_instance(),
            self.postproc.get_instance() if self.postproc is not None else None,
        )


class EncoderDecoder(nn.Module):
    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, postproc: nn.Module | None
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.postproc = postproc

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """"""
        x = self.encoder(inputs)
        out = self.decoder(x)
        if self.postproc is not None:
            out = self.postproc(out)

        return out
