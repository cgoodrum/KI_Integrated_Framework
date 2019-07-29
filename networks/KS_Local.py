import networkx as nx
import matplotlib.pyplot as plt


class Local_KS(object):

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def __init__(self, name = None, filename = None):
        self.name = name
        self.network = self.init_network(filename)

    def init_network(self, filename):
        G = nx.read_pajek("../data_sources/{}".format(filename))
        G1 = nx.DiGraph(G)
        return G1

    def save_network(self, filename = 'untitled_network'):
        plt.figure()
        nx.draw_networkx(self.network, pos = {k[0]:(k[1]['x'],k[1]['y']) for k in self.network.nodes(data=True)})
        plt.savefig('../figures\\{}.png'.format(filename))


def main():

    # Import Gephi Networks into NetworkX format, in Local_KS class structure.
    ops = Local_KS(name='OPS',filename = 'KS_operations.net')
    dist = Local_KS(name='DIST',filename = 'KS_distribution.net')
    navarch = Local_KS(name='NAVARCH',filename = 'KS_navarch.net')

    return {"OPS": ops, "DIST": dist, "NAVARCH": navarch}

if __name__ == '__main__':
    main()
