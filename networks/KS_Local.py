import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy


class Local_KS(object):

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def __init__(self, name = None, filename = None):
        self.name = name
        self.network = self.init_network(name, filename)

    def init_network(self, name, filename):
        G = nx.read_pajek("../data_sources/{}".format(filename))
        G1 = nx.DiGraph(G)
        G2 = deepcopy(G1)

        for node in G2.nodes(data=True):
            x = node[1]['x']
            y = node[1]['y']
            z = float()
            pos = (x,y,z)

            del_keys = [k for k in node[1].keys()]

            for key in del_keys:
                del node[1][key]

            node[1]['pos'] = pos
            node[1]['node_name'] = node[0]
            node[1]['layer'] = name
            ################# Below may need to be changed!!!
            node[1]['val'] = None
            node[1]['data_status'] = None

        G3 = nx.convert_node_labels_to_integers(G2)
        return G3

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
