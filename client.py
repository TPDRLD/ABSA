import json
from jsonrpc import ServerProxy, JsonRpc20, TransportTcpIp


class StanfordNLP:
    def __init__(self):
        self.server = ServerProxy(JsonRpc20(),
                                  TransportTcpIp(addr=("127.0.0.1", 8080),timeout=200.0))
    
    def parse(self, text):
        return json.loads(self.server.parse(text))
