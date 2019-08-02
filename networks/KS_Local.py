import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
import yaml

class Local_KS(object):

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def __init__(self, name = None, filename = None, data = None):
        self.name = name
        self.data = data
        self.network = self.init_network(self.name, filename, self.data)

    def init_network(self, name, filename, data):
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

            if self.data:
                if node[0] in self.data.keys():
                    node[1]['val'] = self.data[node[0]]
                    node[1]['data_status'] = 1.0
                else:
                    node[1]['val'] = None
                    node[1]['data_status'] = 0.0
            else:
                node[1]['val'] = None
                node[1]['data_status'] = 0.0
            node[1]['time'] = 0

        G3 = nx.convert_node_labels_to_integers(G2)
        
        for e in G3.edges(data=False):
            G3.edges[e]['layer'] = name
            G3.edges[e]['type'] = 'INTERNAL'
            G3.edges[e]['time'] = 0

        G4 = nx.MultiDiGraph(G3)
        return G4

    def save_network(self, filename = 'untitled_network'):
        plt.figure()
        nx.draw_networkx(self.network, pos = {k[0]:(k[1]['x'],k[1]['y']) for k in self.network.nodes(data=True)})
        plt.savefig('../figures\\{}.png'.format(filename))

def import_yaml(filename = None):
    if filename:
        with open("../inputs\\{}.yaml".format(filename), 'r') as stream:
            data_loaded = yaml.safe_load(stream)
        return data_loaded
    else:
        print('Please input yaml filename.')

def main():

    yaml_data = import_yaml('initial_values')
    layer_names = {
        "NAVARCH": "KS_navarch.net",
        "OPS": "KS_operations.net",
        "DIST":"KS_distribution.net"
    }

    # Import Gephi Networks into NetworkX format, in Local_KS class structure.
    output = {}
    for layer, filename in layer_names.items():
        output[layer] = Local_KS(name=layer, filename=filename, data = yaml_data[layer])

    return output


if __name__ == '__main__':
    main()
