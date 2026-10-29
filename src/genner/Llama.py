from .OAI import OAIGenner
from .config import ServerConfig


class LlamaGenner(OAIGenner):
    def __init__(self, client, config: ServerConfig):
        super().__init__(client, config, identifier="llama")
