from ..graph.edge import Edge

class StochasticTransit(Edge):
    _available = True
    _stochastic_group = None

    def set_stochastic_group(self, transit_group):
        self._stochastic_group = transit_group

    def set_availability(self, val):
        self._available = val

    def is_available(self):
        return self._available

    def get_stochastic_group(self):
        return self._stochastic_group