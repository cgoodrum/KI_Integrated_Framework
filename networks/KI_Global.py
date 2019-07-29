import sys
sys.path.append('../')
import networkx as nx
import graphviz as gv
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import KS_Local as KS_Local
import math
from mpl_toolkits.mplot3d import Axes3D
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

class Knowledge_Network(object):
    # This class defines the knowledge network for a single agent, used to
    # demonstrate cases 1, 2, and 3.

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def __init__(self, name = None, target_node = 0, time_step = 0):
        self.name = name
        self.target_node = target_node
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
        G = nx.DiGraph()
        return G

    def calculate_data_status(self, node):
        # Determines the data status of a node based on if it has a value or
        # not.
        if self.network.node[node]['val']:
            status = 1.0
        else:
            status = 0.0
        return status

    def add_attributes(self, name = '', value = None):
        if value:
            status = 1.0
        else:
            status = 0.0
        out_dict = {
            'node_name': name,
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

class Integrated_Framework(object):

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def __init__(self):
        self.local_K = KS_Local.main()
        self.global_K = KS_Global()
        self.global_I = IS_Global()
        self.time_step = 0

    def draw_framework(self,
        hoz_offset = 1500,
        vert_offset = 100,
        color = {
            "OPS": "green",
            "DIST": "blue",
            "NAVARCH": "red"
            }
        ):

        # Draw local Knowledge layers
        name_pos = {}
        edges = {}
        lat_offset = float(0)
        for name, local_class in self.local_K.items():
            edges[name] = local_class.network.edges()
            pos = {}
            for node_data in local_class.network.nodes(data=True):
                pos[node_data[0]] = (node_data[1]['x']+lat_offset, node_data[1]['y'], vert_offset*2)
            name_pos[name] = pos
            lat_offset += hoz_offset

        fig = plt.figure()
        ax = Axes3D(fig)

        for name, local_class in self.local_K.items():
            # PLOT NODES
            for _name, _node in name_pos[name].items():
                label = _name
                xi = _node[0]
                yi = _node[1]
                zi = _node[2]

                ax.scatter(xi,yi,zi, c=color[name])
            # PLOT EDGES
            for edge in edges[name].keys():
                x = np.array((name_pos[name][edge[0]][0], name_pos[name][edge[1]][0]))
                y = np.array((name_pos[name][edge[0]][1], name_pos[name][edge[1]][1]))
                z = np.array((name_pos[name][edge[0]][2], name_pos[name][edge[1]][2]))

                ax.plot(x,y,z,c='black')

        # Draw Global Knowledge layer
        for node in self.global_K.network.nodes(data=True):
            ax.scatter(node[1]['pos'][0], node[1]['pos'][1], node[1]['pos'][2]+vert_offset*0, c='black')

        # TO DO: DRAW EDGES

        # Draw Global Information layer
        for node in self.global_I.network.nodes(data=True):
            ax.scatter(node[1]['pos'][0], node[1]['pos'][1], node[1]['pos'][2]+vert_offset*1, c='black')

        # TO DO: DRAW EDGES


        ax.set_axis_off()
        plt.show()

class KS_Global(Knowledge_Network):

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def __init__(self, data = {}, **kwargs):
        Knowledge_Network.__init__(self, **kwargs)
        self.data = data

    def test_network(self):
        self.network.add_node(0, **{'pos': (1500, 0, 0)})
        return self.network

class IS_Global(Knowledge_Network):

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def __init__(self, data = {}, **kwargs):
        Knowledge_Network.__init__(self, **kwargs)
        self.data = data

    def test_network(self):
        self.network.add_node(0, **{'pos': (1500, 0, 0)})
        return self.network


def main():
    KIF = Integrated_Framework()
    KIF.global_K.test_network()
    KIF.global_I.test_network()
    KIF.draw_framework()



if __name__ == '__main__':
    main()
