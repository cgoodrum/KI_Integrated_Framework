import sys
sys.path.append('../')
import networkx as nx
import graphviz as gv
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import KS_Local as KS_Local
import math
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import itertools
import matplotlib.pyplot as plt
import dit
from scipy.stats import entropy
import time
from collections import defaultdict
import logging
import seaborn as sns
import random as rd
import numpy as np
import pandas as pd
from copy import deepcopy

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, format = '%(levelname)s: %(message)s', level=logging.INFO)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)



class Knowledge_Network(object):
    # This class defines the knowledge network for a single layer

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def __init__(self, name = None, time_step = 0):
        self.name = name
        self.target_node = []
        self.time_step = time_step
        self.rework_time = float()
        self.network = self.init_network()
        self.topological_entropy_time_series = {}
        self.simple_entropy_time_series = {}
        self.binary_entropy_time_series = {}
        self.target_value_shannon_entropy_time_series = {}
        self.target_value_CRE_entropy_time_series = {}
        self.graphviz_path = 'C:/Python37/Lib/site-packages/graphviz-2.38/release/bin'

    def init_network(self):
        G = nx.MultiDiGraph()
        return G

    def calculate_data_status(self, node):
        # Determines the data status of a node based on if it has a value or
        # not.
        if self.network.node[node]['val']:
            status = 1.0
        else:
            status = 0.0
        return status

    def add_attributes(self, name = None, layer = None, value = None, pos = (float(),float(),float())):
        if value:
            status = 1.0
        else:
            status = 0.0
        out_dict = {
            'pos': pos,
            'node_name': name,
            'layer': layer,
            'val': value,
            'data_status': status
        }
        return out_dict

    def calculate_pagerank(self, alpha = 0.85):
        pagerank = nx.pagerank(self.network, alpha = alpha)
        return pagerank

    def calculate_outcome_probs(self, node):
        n = len(self.calculate_binary_outcomes(node)) # Find the connected nodes to the target node
        return [1.0/float(n)]*n

    def calculate_binary_entropy(self, node):
        temp = dit.Distribution(self.calculate_binary_outcomes(node), self.calculate_outcome_probs(node))
        entropy = dit.shannon.entropy(temp)
        return entropy

    def calculate_simple_entropy(self):
        # A simple calculation using the data status of each data element, using the
        # P(1) and P(0). This ignores structure of the network.
        # Has a min value of 0, and a max value of 1

        num_nodes = len(self.network.nodes())

        if num_nodes <= 1:
            return 0

        sum_1 = 0.0
        sum_0 = 0.0
        for node in self.network.nodes(data=True):
            data_status = node[1]['data_status']
            if data_status == 1.0:
                sum_1 += 1.0
            else:
                sum_0 += 1.0

        p_1 = sum_1/float(num_nodes)
        p_0 = sum_0/float(num_nodes)
        p = [p_0, p_1]
        H = entropy(p,base=2)
        return H

    def calculate_topological_entropy(self):
        # Calculates the entropy of the network using the pagerank of each node
        # ignoring any of the data statuses or values.
        pagerank = self.calculate_pagerank()
        d = dit.ScalarDistribution(pagerank)
        entropy = dit.shannon.entropy(d)
        return entropy

    def calculate_target_value_entropy(self, method = "shannon", **kwargs):
        # Calculates the entropy of the values of the target node based on its
        # distribution

        def value_shannon_entropy(node, **kwargs):
            histogram_data, values = self.get_histogram_data(node, **kwargs)
            probs = histogram_data[0]
            H = entropy(probs, base = 2)
            return H

        def value_CRE_entropy(node, **kwargs):
            histogram_data, values = self.get_histogram_data(node, **kwargs)
            bins = histogram_data[1][1:]
            probs = histogram_data[0]*(bins[1]-bins[0])

            d = dit.ScalarDistribution(bins,probs)
            H = dit.other.cumulative_residual_entropy(d)
            return H

        node = self.target_node
        value_entropy = {
            "shannon": value_shannon_entropy(node, **kwargs),
            "CRE": value_CRE_entropy(node, **kwargs)
            # Could add more entropy measures if required!
        }
        return value_entropy[method]

    def get_histogram_data(self, node, **kwargs):
        pred = list(self.network.predecessors(node)) # Find the connected nodes
        pred = sorted([k for k in pred], reverse=True) # Sort the calculated nodes in descending order.
        values = []
        for n in pred:
            values.append(self.network.node[n]['val'])
        histogram_data = plt.hist(values, **kwargs)
        plt.close()
        return histogram_data, values

    def build_entropy_time_series(self, time_series_name, entropy_function):
        time_series_name[self.time_step] = entropy_function

    def visualize_network_growth(self):
        self.get_network()
        pylab.draw()
        plt.pause(0.2)

    def get_network(self, prog = 'twopi',**kwargs):
        prog = '{}/{}'.format(self.graphviz_path, prog)
        pos = graphviz_layout(self.network, prog = prog)
        labels = {k: self.network.nodes(data=True)[k]['node_name'] for k in self.network.nodes()}
        nx.draw(self.network, pos, labels = labels, **kwargs)

    def save_entropy_time_series_plot(self, time_series_dict, filename = 'untitled', xlabel = 'Time', ylabel = 'Entropy', title = '', x_min = 0, x_max = 15, y_min = 0, y_max=5, grid = True,**kwargs):
        plt.figure()
        sns.set()
        sns.set_style('white')
        plt.plot(time_series_dict.keys(), time_series_dict.values())
        sns.despine()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        plt.grid(grid)
        plt.title(title)
        plt.savefig('../figures\\' + filename +'.png')

    def save_final_network_plot(self, filename = 'untitled_network', **kwargs):
        plt.figure()
        self.get_network(**kwargs)
        plt.savefig('../figures\\{}.png'.format(filename))

    def save_target_value_histogram_plot(self, filename = None, label = '', **kwargs):

        data, values = self.get_histogram_data(
            node = self.target_node,
            **kwargs['hist_kwargs']
        )

        plt.figure()
        sns.set()
        sns.set_style('white')
        sns.distplot(values, bins = kwargs['hist_kwargs']['bins'], label = label, **kwargs['plot_kwargs'])
        sns.despine()
        plt.legend(loc='upper right')
        plt.xlabel('Volume')
        plt.ylabel('Probability Density')
        plt.title('Histogram of Target Values')
        axes = plt.gca()
        axes.set_xlim([kwargs['hist_kwargs']['range'][0],kwargs['hist_kwargs']['range'][1]])
        axes.set_ylim([None,None])
        if filename:
            plt.savefig('../figures\\{}.png'.format(filename))
        else:
            plt.show()

    def grow(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def apply_layout(self, layout = "spring", **kwargs):

        def spring(network, **kwargs):
            pos = nx.spring_layout(network, **kwargs)

            for n in network.nodes():
                network.node[n]['pos'] = list(network.node[n]['pos'])
                network.node[n]['pos'][0] = pos[n][0]
                network.node[n]['pos'][1] = pos[n][1]
                network.node[n]['pos'] = tuple(network.node[n]['pos'])
            return pos

        method = {
            "spring": spring(self.network, **kwargs)
        }

        return method[layout]

class Integrated_Framework(Knowledge_Network):

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def __init__(self, data = {}, **kwargs):
        Knowledge_Network.__init__(self, **kwargs)
        self.data = data
        self.local_K = KS_Local.main()
        self.global_K = K_Global()
        self.global_I = I_Global()
        self.network = self.init_network()
        self.time_step = 0

    def init_network(self):
        G = nx.MultiDiGraph()
        return G

    def build_layered_network(self):
        local_networks = [local.network for local in self.local_K.values()]
        all_networks = local_networks + [self.global_K.network] + [self.global_I.network]
        self.network = nx.disjoint_union_all(all_networks)
        return self.network

    def add_KG_target_node(self, name = None, **kwargs):
        if name:
            # Get new node ID for added target node
            new_node_id = self.get_new_node_id()
            # Add to integrated framework
            self.network.add_node(new_node_id, **self.add_attributes(name = name, layer = 'K_GLOBAL', **kwargs))
            # Add node to global knowledge network and list of target nodes
            self.global_K.network.add_node(new_node_id, **self.add_attributes(name = name, layer = 'K_GLOBAL', **kwargs))
            self.global_K.target_node.append((new_node_id, self.network.node[new_node_id]))
        else:
            print('Please specify name of Global Knowledge Target Node')
        return self.network

    def grow_IG_from_KG(self, name = None, **kwargs):
        if name:
            # Get new node ID for added node
            new_node_id = self.get_new_node_id()
            # Get KG node id from which the IG node is created
            old_node_id = None
            for (KG_node_id, KG_node_data) in self.global_K.target_node:
                if KG_node_data['node_name'] == name:
                    old_node_id = KG_node_id
            if old_node_id == None:
                print('Error: No KG target node named {}'.format(name))
            # Add to integrated framework
            self.network.add_node(new_node_id, **self.add_attributes(name = name, layer = 'I_GLOBAL', **kwargs))
            # Add node to global information network
            self.global_I.network.add_node(new_node_id, **self.add_attributes(name = name, layer = 'I_GLOBAL', **kwargs))
            # Add edge from KG node to new IG node
            self.network.add_edge(old_node_id, new_node_id, layer = "BETWEEN", type = "CREATE", weight = 1.0)
        else:
            print("No node added to Global Information from Global Knowledge. Specify node name to add.")
        return self.network

    def grow_IG_from_KL(self, name = None, local_layer = None, **kwargs):
        if name:
            # Get new node ID for added node
            new_node_id = self.get_new_node_id()
            # Get KL node id from which the IG node is created
            old_node_id = None
            local_nodes = [(n,data) for n, data in self.network.nodes(data=True) if data['layer'] == local_layer]
            for (KL_node_id, KL_node_data) in local_nodes:
                if KL_node_data['node_name'] == name:
                    old_node_id = KL_node_id
            if old_node_id == None:
                print('Error: No node named <{}> in local layer <{}> '.format(name, layer))
            # Add to integrated framework
            self.network.add_node(new_node_id, **self.add_attributes(name = name, layer = 'I_GLOBAL', **kwargs))
            # Add node to global information network
            self.global_I.network.add_node(new_node_id, **self.add_attributes(name = name, layer = 'I_GLOBAL', **kwargs))
            # Add edge from KL node to new IG node
            self.network.add_edge(old_node_id, new_node_id, layer = "BETWEEN", type = "CREATE", weight = 1.0)
        else:
            print("No node added to Global Information from Local Knowledge. Specify local node name to add.")
        return self.network

    def select_KL_node_from_IG(self, IG_node = None, local_layer = None, **kwargs):
        if IG_node:
            global_node =[(n,data) for n, data in self.network.nodes(data=True) if (data['layer'] == "I_GLOBAL" and data['node_name'] == IG_node)]
            local_node = [(n,data) for n, data in self.network.nodes(data=True) if (data['layer'] == local_layer and data['node_name'] == IG_node)]
            if len(local_node) != 1:
                print("Multiple nodes named {} in layer {}".format(IG_node, local_layer))
            else:
                global_node_id = self.get_node_id(node_name=IG_node, layer= "I_GLOBAL")
                local_node_id = self.get_node_id(node_name=IG_node, layer= local_layer)
                self.network.add_edge(global_node_id, local_node_id, layer = "BETWEEN", type = "SELECT", weight = 1.0)
        return local_node_id

    def create_update_edge(self, start_node_name = None, start_layer = None, end_node_name = None, end_layer = None, **kwargs):
        start_node_id = self.get_node_id(node_name = start_node_name, layer = start_layer)
        end_node_id = self.get_node_id(node_name = end_node_name, layer = end_layer)
        if start_layer == end_layer:
            self.network.add_edge(start_node_id, end_node_id, layer = "INTERNAL", type = "UPDATE", weight = 1.0)
        else:
            self.network.add_edge(start_node_id, end_node_id, layer = "BETWEEN", type = "UPDATE", weight = 1.0)
        return self.network

    def get_node_id(self, node_name = None, layer = None, **kwargs):
        if node_name:
            node = [n for n, data in self.network.nodes(data=True) if (data['layer'] == layer and data['node_name'] == node_name)]
            node_id = node[0]
        return node_id

    def update_data_status(self, node):
        # Need to write code to update the data status of a node based on nodes it is connected to.
        pass

    def get_new_node_id(self):
        new_node_id = max(self.network.nodes()) + 1
        return new_node_id

    def update_node_positions(self):

        self.global_K.apply_layout(layout='spring', scale = 500, center = (1500,0), dim = 2 )
        self.global_I.apply_layout(layout='spring', scale = 500, center = (1500,0), dim = 2 )

        for (n, data) in self.global_I.network.nodes(data=True):
            self.network.node[n]['pos'] = data['pos']

        for (n, data) in self.global_K.network.nodes(data=True):
            self.network.node[n]['pos'] = data['pos']

        return self.network

    def draw_framework(self, hoz_offset = 1500, vert_offset = 100, labels = True, color = {"OPS": "green", "DIST": "blue", "NAVARCH": "red", "K_GLOBAL": "grey", "I_GLOBAL": "yellow"}, label_kwargs = {'color': 'black','size': '6'}, edge_kwargs = {'arrowstyle': "-|>", 'lw': 1, 'mutation_scale' : 8, 'color' : 'black',}):

        # update the node positions from the GlobalK and GlobalI networks

        self.update_node_positions()

        fig = plt.figure()
        ax = Axes3D(fig)

        lat_offset = float(0)
        plotted_node_pos = {}

        # Find unique layers to plot nodes
        layers = list(set(tup[1]['layer'] for tup in self.network.nodes(data=True)))

        for layer in layers:

            nodes = [(n,data) for n, data in self.network.nodes(data=True) if data['layer'] == layer]
            in_layer_edges = [(u,v,data) for (u,v, data) in self.network.edges(data=True) if data['layer'] == layer]

            if "GLOBAL" not in layer:

                # PLOT LOCAL NODES
                for (node_id, node_data) in nodes:
                    label = node_data['node_name']
                    xi = node_data['pos'][0] + lat_offset
                    yi = node_data['pos'][1]
                    zi = node_data['pos'][2] + vert_offset*2
                    plotted_node_pos[node_id] = {'x':xi, 'y':yi, 'z':zi}

                    ax.scatter(xi,yi,zi, c=color[layer])

                    if labels == True:
                        ax.text(xi,yi,zi,label, **label_kwargs)

                lat_offset += hoz_offset

                # PLOT LOCAL EDGES
                for (n1, n2, edge_data) in in_layer_edges:
                    x = np.array((plotted_node_pos[n1]['x'], plotted_node_pos[n2]['x']))
                    y = np.array((plotted_node_pos[n1]['y'], plotted_node_pos[n2]['y']))
                    z = np.array((plotted_node_pos[n1]['z'], plotted_node_pos[n2]['z']))

                    a = Arrow3D(x,y,z,**edge_kwargs)
                    ax.add_artist(a)

            if layer == "K_GLOBAL":

                # PLOT GLOBAL K NODES
                for (node_id, node_data) in nodes:
                    label = node_data['node_name']
                    xi = node_data['pos'][0]
                    yi = node_data['pos'][1]
                    zi = node_data['pos'][2] + vert_offset*0.0
                    plotted_node_pos[node_id] = {'x':xi, 'y':yi, 'z':zi}

                    ax.scatter(xi, yi, zi, c=color['K_GLOBAL'])

                    if labels == True:
                        ax.text(xi,yi,zi,label, **label_kwargs)

                # PLOT GLOBAL K EDGES
                for (n1, n2, edge_data) in in_layer_edges:
                    x = np.array((plotted_node_pos[n1]['x'], plotted_node_pos[n2]['x']))
                    y = np.array((plotted_node_pos[n1]['y'], plotted_node_pos[n2]['y']))
                    z = np.array((plotted_node_pos[n1]['z'], plotted_node_pos[n2]['z']))

                    a = Arrow3D(x,y,z,**edge_kwargs)
                    ax.add_artist(a)

            if layer == "I_GLOBAL":

                # PLOT GLOBAL I NODES
                for (node_id, node_data) in nodes:
                    label = node_data['node_name']
                    xi = node_data['pos'][0]
                    yi = node_data['pos'][1]
                    zi = node_data['pos'][2] + vert_offset*1.0
                    plotted_node_pos[node_id] = {'x':xi, 'y':yi, 'z':zi}

                    ax.scatter(xi, yi, zi, c=color['I_GLOBAL'])

                    if labels == True:
                        ax.text(xi,yi,zi,label, **label_kwargs)

                # PLOT GLOBAL I EDGES
                for (n1, n2, edge_data) in in_layer_edges:
                    x = np.array((plotted_node_pos[n1]['x'], plotted_node_pos[n2]['x']))
                    y = np.array((plotted_node_pos[n1]['y'], plotted_node_pos[n2]['y']))
                    z = np.array((plotted_node_pos[n1]['z'], plotted_node_pos[n2]['z']))

                    a = Arrow3D(x,y,z,**edge_kwargs)
                    ax.add_artist(a)

        # Plot inter-layer edges

        between_layer_edges = [(u,v,data) for (u,v, data) in self.network.edges(data=True) if data['layer'] == "BETWEEN"]

        for (n1, n2, edge_data) in between_layer_edges:
            x = np.array((plotted_node_pos[n1]['x'], plotted_node_pos[n2]['x']))
            y = np.array((plotted_node_pos[n1]['y'], plotted_node_pos[n2]['y']))
            z = np.array((plotted_node_pos[n1]['z'], plotted_node_pos[n2]['z']))

            a = Arrow3D(x,y,z,**edge_kwargs)
            ax.add_artist(a)

        ax.set_axis_off()
        plt.show()

class K_Global(Knowledge_Network):

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def __init__(self, data = {}, **kwargs):
        Knowledge_Network.__init__(self, **kwargs)
        self.data = data

    def add_target_node(self, target_node_label=None):
        self.network.add_node(target_node_label, **self.add_attributes(name=target_node_label, layer = "K_GLOBAL"))
        self.target_node.append((target_node_label, self.network.node[target_node_label]))
        return self.network



class I_Global(Knowledge_Network):

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def __init__(self, data = {}, **kwargs):
        Knowledge_Network.__init__(self, **kwargs)
        self.data = data


def main():
    KIF = Integrated_Framework()

    KIF.build_layered_network()
    KIF.add_KG_target_node("GMT")
    KIF.grow_IG_from_KG("GMT")
    KIF.grow_IG_from_KL("W_fuel","NAVARCH")
    KIF.select_KL_node_from_IG("GMT","NAVARCH")
    KIF.create_update_edge("GMT", "NAVARCH", "GMT", "I_GLOBAL")

    #KIF.update_node_positions()
    #KIF.global_K.apply_layout(layout='spring', scale = 500, center = (1500,0), dim = 2 )
    #KIF.global_I.apply_layout(layout='spring', scale = 500, center = (1500,0), dim = 2 )

    #KIF.network.add_edge(77,78, layer="BETWEEN")
    #KIF.network.add_edge(78,72, layer="BETWEEN")

    #for (n1, n2, data) in KIF.network.edges(data=True):
    # print(KIF.network.node[n1]['node_name'],(KIF.network.node[n1]['layer']), KIF.network.node[n2]['node_name'], (KIF.network.node[n2]['layer']), data)

    df = nx.to_pandas_edgelist(KIF.network)
    temp = df[df['layer'] == "BETWEEN"]
    print(df)

    #KIF.global_I.add_node_from_KG(KIF.global_K.target_node[0])
    #print(KIF.global_K.target_node)
    #KIF.global_I.test_network()
    KIF.draw_framework()



if __name__ == '__main__':
    main()
