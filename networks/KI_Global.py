import sys
import os.path
sys.path.append('../')
import networkx as nx
import graphviz as gv
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.algorithms import bipartite
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
from openpyxl import load_workbook
import csv
import xlwings as xw
import yaml
import pickle as pkl


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, format = '%(levelname)s: %(message)s', level=logging.INFO)

###############################################################################
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

###############################################################################
class Excel(object):
    def __init__(self,excel_filename, references_filename, save_filename = None):
        self.app = xw.App(visible=False)
        self.wb = xw.Book("../excel\\{}".format(excel_filename))
        self.sheets = self.wb.sheets
        self.dest= "../excel\\{}".format(excel_filename)
        self.cell_references = self.import_cell_references("../excel\\{}".format(references_filename))
        self.save_filename = save_filename
        self.macro = None

    # Write value to cell
    def write_val(self,sheet,cell,value):
        self.sheets[sheet].range(cell).value = value

    # Get cell value
    def read_val(self, sheet, cell):
        val = self.sheets[sheet].range(cell).value
        return val

    # Save excel file
    def save_excel(self):
        if self.save_filename:
            #if os.path.exists("../results\\{}".format(self.save_filename)):

            #else:
            self.wb.save("../results\\{}".format(self.save_filename))
        else:
            self.wb.save(self.dest)

    def import_cell_references(self, references_filename):
        if references_filename:
            with open(references_filename, 'r') as stream:
                data_loaded = yaml.safe_load(stream)
            return data_loaded
        else:
            print('Please input yaml filename.')

    def run_macro(self, macro_name):
        macro = self.wb.macro(macro_name)
        return macro() ## Check this!!

    def run_function(self, macro_name):
        macro = self.wb.macro(macro_name)
        return macro()

    def close(self):
        self.wb.close()
        self.app.kill()

###############################################################################
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

    def add_attributes(self, name = None, layer = None, type = None, value = None, pos = (float(),float(),float())):
        if value != None:
            status = 1.0
        else:
            status = 0.0
        out_dict = {
            'pos': pos,
            'node_name': name,
            'layer': layer,
            'type': type,
            'val': value,
            'data_status': status,
            'time': self.time_step,
            'val_ts': {self.time_step: value},
            'data_status_ts': {self.time_step: status},
            'intermediate_vals': []
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

###############################################################################
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
        self.time_step = 1
        #self.excel_doc = Excel()

    def init_network(self):
        G = nx.MultiDiGraph()
        return G

    def build_layered_network(self):
        # Build integrated network
        local_networks = [local.network for local in self.local_K.values()]
        all_networks = local_networks + [self.global_K.network] + [self.global_I.network]
        self.network = nx.disjoint_union_all(all_networks)

        # Relabel_nodes in local layers with new labels from integrated network
        for local_network_name, local in self.local_K.items():
            mapping = {}
            for old_node_id, old_data in local.network.nodes(data=True):
                for new_node_id, new_data in self.network.nodes(data=True):
                    if old_data == new_data:
                        mapping.update({old_node_id:new_node_id})
            nx.relabel_nodes(self.local_K[local_network_name].network, mapping, copy=False)
        return self.network

    def calculate_data_status(self, node_id):
        # Determines the data status of a node based on if it has a value or
        # not.
        if self.network.node[node_id]['val'] != None:
            self.network.node[node_id]['data_status'] = 1.0
        else:
            self.network.node[node_id]['data_status'] = 0.0

    def add_KG_target_node(self, name = None, **kwargs):
        if name:
            # Get new node ID for added target node
            new_node_id = self.get_new_node_id()
            # Add to integrated framework
            self.network.add_node(new_node_id, **self.add_attributes(name = name, layer = 'K_GLOBAL', type = 'TARGET',**kwargs))
            # Add node to global knowledge network and list of target nodes
            self.global_K.network.add_node(new_node_id, **self.add_attributes(name = name, layer = 'K_GLOBAL', type = 'TARGET', **kwargs))
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
            self.network.add_node(new_node_id, **self.add_attributes(name = name, layer = 'I_GLOBAL', type = 'TARGET', value = self.network.nodes(data=True)[old_node_id]['val'], **kwargs))
            # Add node to global information network
            self.global_I.network.add_node(new_node_id, **self.add_attributes(name = name, layer = 'I_GLOBAL', type = 'TARGET', value = self.network.nodes(data=True)[old_node_id]['val'], **kwargs))
            # Add edge from KG node to new IG node
            self.network.add_edge(old_node_id, new_node_id, layer = "BETWEEN", type = "CREATE", weight = 1.0, time = self.time_step)
        else:
            print("No node added to Global Information from Global Knowledge. Specify node name to add.")
        return self.network

    def grow_IG_from_KL(self, name = None, local_layer = None, **kwargs):
        # NEED TO ADD LOGIC FOR IF A NODE WITH THAT NAME ALREADY EXISTS IN I_GLOBAL, THEN ADD AN UPDATE EDGE TO THAT NODE RATHER THAN CREATE A NEW ONE.
        if name:
            # Get KL node id from which the IG node is created
            old_node_id = None
            local_nodes = [(n,data) for n, data in self.network.nodes(data=True) if data['layer'] == local_layer]
            for (KL_node_id, KL_node_data) in local_nodes:
                if KL_node_data['node_name'] == name:
                    old_node_id = KL_node_id
                    if old_node_id == None:
                        print('Error: No node named <{}> in local layer <{}> '.format(name, layer))


            IG_nodes = [(n,data) for n, data in self.network.nodes(data=True) if data['layer'] == "I_GLOBAL"]
            IG_node_names = [data['node_name'] for n, data in self.network.nodes(data=True) if data['layer'] == "I_GLOBAL"]

            if name in IG_node_names:
                for (IG_node_id, IG_node_data) in IG_nodes:
                    if IG_node_data['node_name'] == name:
                        new_node_id = IG_node_id
            else:
                # Get new node ID for added node
                new_node_id = self.get_new_node_id()

                # Add to integrated framework
                self.network.add_node(new_node_id, **self.add_attributes(name = name, layer = 'I_GLOBAL', type = 'NEGOTIATED', **kwargs))
                # Add node to global information network
                self.global_I.network.add_node(new_node_id, **self.add_attributes(name = name, layer = 'I_GLOBAL', type = 'NEGOTIATED', **kwargs))

            # Add edge from KL node to new IG node
            self.network.add_edge(old_node_id, new_node_id, layer = "BETWEEN", type = "CREATE", weight = 1.0, time = self.time_step)
        else:
            print("No node added to Global Information from Local Knowledge. Specify local node name to add.")
        return (new_node_id, name)

    def select_KL_node_from_IG(self, IG_node_name = None, local_layer = None, **kwargs):
        if IG_node_name:
            global_node =[(n,data) for n, data in self.network.nodes(data=True) if (data['layer'] == "I_GLOBAL" and data['node_name'] == IG_node_name)]
            local_node = [(n,data) for n, data in self.network.nodes(data=True) if (data['layer'] == local_layer and data['node_name'] == IG_node_name)]
            if len(local_node) > 1:
                print("Multiple nodes named {} in layer {}!!".format(IG_node_name, local_layer))
            elif len(local_node) == 0:
                print("No nodes named {} in layer {}!!".format(IG_node_name, local_layer))
            else:
                global_node_id = self.get_node_id(node_name=IG_node_name, layer= "I_GLOBAL")
                local_node_id = self.get_node_id(node_name=IG_node_name, layer= local_layer)
                self.network.add_edge(global_node_id, local_node_id, layer = "BETWEEN", type = "SELECT", weight = 1.0, time = self.time_step)
        return local_node_id

    def select_IG_node_from_KL(self, IG_node_name = None, local_layer = None, **kwargs):
        if IG_node_name:
            global_node =[(n,data) for n, data in self.network.nodes(data=True) if (data['layer'] == "I_GLOBAL" and data['node_name'] == IG_node_name)]
            local_node = [(n,data) for n, data in self.network.nodes(data=True) if (data['layer'] == local_layer and data['node_name'] == IG_node_name)]
            if len(local_node) > 1:
                print("Multiple nodes named {} in layer {}!!".format(IG_node_name, local_layer))
            elif len(local_node) == 0:
                print("No nodes named {} in layer {}!!".format(IG_node_name, local_layer))
            else:
                global_node_id = self.get_node_id(node_name=IG_node_name, layer= "I_GLOBAL")
                local_node_id = self.get_node_id(node_name=IG_node_name, layer= local_layer)
                self.network.add_edge(local_node_id, global_node_id, layer = "BETWEEN", type = "SELECT", weight = 1.0, time = self.time_step)
        return local_node_id

    def create_update_edge(self, start_node_name = None, start_layer = None, end_node_name = None, end_layer = None, **kwargs):
        start_node_id = self.get_node_id(node_name = start_node_name, layer = start_layer)
        end_node_id = self.get_node_id(node_name = end_node_name, layer = end_layer)
        if start_layer == end_layer:
            self.network.add_edge(start_node_id, end_node_id, layer = "INTERNAL", type = "UPDATE", weight = 1.0, time = self.time_step)
        else:
            self.network.add_edge(start_node_id, end_node_id, layer = "BETWEEN", type = "UPDATE", weight = 1.0, time = self.time_step)
        return self.network

    def send_node_value(self, start_node_name = None, start_layer = None, end_node_name = None, end_layer = None, **kwargs):
        # copies node value from origin node to destination node
        start_node_id = self.get_node_id(node_name = start_node_name, layer = start_layer)
        end_node_id = self.get_node_id(node_name = end_node_name, layer = end_layer)
        self.network.node[end_node_id]['val'] = self.network.node[start_node_id]['val']
        self.network.node[end_node_id]['data_status'] = self.network.node[start_node_id]['data_status'] ## CHECK??

        self.network.node[end_node_id]['val_ts'][self.time_step] = self.network.node[start_node_id]['val']
        self.network.node[end_node_id]['data_status_ts'][self.time_step] = self.network.node[start_node_id]['data_status']

        if not (end_layer == "I_GLOBAL" or end_layer == "K_GLOBAL"):
            self.local_K[end_layer].network.node[end_node_id]['val'] = self.network.node[end_node_id]['val']
            self.local_K[end_layer].network.node[end_node_id]['data_status'] = self.network.node[end_node_id]['data_status']

            self.local_K[end_layer].network.node[end_node_id]['val_ts'][self.time_step] = self.network.node[start_node_id]['val']
            self.local_K[end_layer].network.node[end_node_id]['data_status_ts'][self.time_step] = self.network.node[start_node_id]['data_status']

        self.create_update_edge(start_node_name, start_layer, end_node_name, end_layer)

    def get_node_id(self, node_name = None, layer = None, **kwargs):
        if node_name:
            node = [n for n, data in self.network.nodes(data=True) if (data['layer'] == layer and data['node_name'] == node_name)]
            node_id = node[0]
        return node_id

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

    def draw_framework(self, hoz_offset = 3000, vert_offset = 100, labels = True, color = {"OPS": "green", "DIST": "blue", "NAVARCH": "red", "K_GLOBAL": "grey", "I_GLOBAL": "yellow"}, label_kwargs = {'color': 'black','size': '6'}, edge_kwargs = {'arrowstyle': "-|>", 'lw': 1, 'mutation_scale' : 8, 'color' : 'black',}):

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

    def draw_layer(self, layer_name):
        plt.figure()
        edges = {(u,v,k):d for u,v,k,d in self.network.edges(keys=True, data=True) if d['layer'] == layer_name}
        layer_network = nx.edge_subgraph(self.network, edges)
        labels = {k: layer_network.nodes(data=True)[k]['node_name'] for k in layer_network.nodes()}
        nx.draw_spring(layer_network, labels = labels)
        plt.show()

    def create_dataframe(self):

        df = nx.to_pandas_edgelist(self.network)
        mapping = {'source': {}, 'target': {}}
        for node_type, node_data in mapping.items():
            for k, v in self.network.nodes(data=True):
                for k2 in v.keys():
                    mapping[node_type][k2] = '{}_{}'.format(node_type, k2)

        tdfs = []
        for node_type in mapping.keys():
            dfs = []
            for i, row in df.iterrows():
                node_data = self.network.node[row[node_type]]
                new_data = deepcopy(node_data)
                #print(new_data['val_ts'])
                del new_data['val_ts']
                del new_data['data_status_ts']
                del new_data['intermediate_vals']
                new_data['pos'] = [new_data['pos']]
                new_data['val'] = node_data['val_ts'][row['time']]
                new_data['data_status'] = node_data['data_status_ts'][row['time']]
                temp_df = pd.DataFrame(new_data)
                temp_df.rename(columns=mapping[node_type], inplace=True)
                dfs.append(temp_df)
            tdf = pd.concat(dfs, ignore_index = True)
            tdfs.append(tdf)
        temp_df = pd.concat(tdfs, ignore_index = False, axis=1)
        df_out = pd.concat([df,temp_df],axis=1, sort=False)
        return df_out

    def create_node_dataframe(self):
        dfs = []
        for node, node_data in self.network.nodes(data=True):
            node_name = node_data['node_name']
            layer = node_data['layer']
            type = node_data['type']
            time = [t for t,ds in node_data['data_status_ts'].items()]
            ds = [ds for t,ds in node_data['data_status_ts'].items()]
            values = [val for t,val in node_data['val_ts'].items()]
            data_dict = {'data_status':ds, "val":values, 'time': time}
            tdf = pd.DataFrame(data_dict)
            tdf['node_name'] = node_name
            tdf['layer'] = layer
            tdf['type'] = type
            dfs.append(tdf)
        df = pd.concat(dfs).reset_index(drop=True)
        return df

    def save_network_pickle(self, filename):
        nx.write_gpickle(self.network,"../results\\{}.gpickle".format(filename))

    def do_local_calculations(self, local_layer_name = "", excel_filename = "", references_filename = "cell_references.yaml", macro_name = None ):
        # Find independent variables, place their values into the excel sheet, and place all new values into the network.
        # Determine in degrees of local network nodes
        in_degrees = self.local_K[local_layer_name].network.in_degree([n for n,v in self.network.nodes(data=True) if v['layer'] == local_layer_name])
        ind_nodes = {}
        dep_nodes = {}

        #Sort independent and dependent nodes
        for n, k_in in in_degrees:
            if k_in == 0:
                ind_nodes[n] = self.network.node[n]['node_name']
            else:
                dep_nodes[n] = self.network.node[n]['node_name']

        # Open instance of excel spreadsheet
        excel_doc = Excel(excel_filename, references_filename)#, save_filename = "local_calculations_enhanced_final_state.xlsm")

        # Write independent values to spreadsheet (inputs)
        for node_id, node_name in ind_nodes.items():
            excel_doc.write_val(sheet = local_layer_name, cell = excel_doc.cell_references[local_layer_name][node_name], value = self.network.node[node_id]['val'])
            #self.local_K[local_layer_name].network.node[node_id]['val'] = self.network.node[node_id]['val']

        # Run macros (if specified)
        intermediate_results = None
        if macro_name != None:
            excel_doc.run_macro(macro_name)
            # Store intermediate variables as well.
            print(macro_name)
            intermediate_results = self.get_intermediate_results(excel_doc)

        # Read independent variables back into network post-macro
        for node_id, node_name in ind_nodes.items():
            # Update values
            self.network.node[node_id]['val'] = excel_doc.read_val(sheet=local_layer_name, cell=excel_doc.cell_references[local_layer_name][node_name])
            self.local_K[local_layer_name].network.node[node_id]['val'] = self.network.node[node_id]['val']
            #self.network.node[node_id]['data_status'] = self.calculate_data_status(node_id)
            # Update Data status
            self.calculate_data_status(node_id)
            self.local_K[local_layer_name].network.node[node_id]['data_status'] = self.network.node[node_id]['data_status']

            # Update intermediate_vals if there are any
            if intermediate_results != None:
                self.network.node[node_id]['intermediate_vals'].extend(intermediate_results[node_name])
                self.local_K[local_layer_name].network.node[node_id]['intermediate_vals'] = self.network.node[node_id]['intermediate_vals']

        # Read dependent values to network (outputs)
        for node_id, node_name in dep_nodes.items():
            self.network.node[node_id]['val'] = excel_doc.read_val(sheet=local_layer_name, cell=excel_doc.cell_references[local_layer_name][node_name])
            self.local_K[local_layer_name].network.node[node_id]['val'] = self.network.node[node_id]['val']
            #self.network.node[node_id]['data_status'] = self.calculate_data_status(node_id)
            self.calculate_data_status(node_id)
            self.local_K[local_layer_name].network.node[node_id]['data_status'] = self.network.node[node_id]['data_status']

            # Update intermediate_vals if there are any
            if intermediate_results != None:
                self.network.node[node_id]['intermediate_vals'].extend(intermediate_results[node_name])
                self.local_K[local_layer_name].network.node[node_id]['intermediate_vals'] = self.network.node[node_id]['intermediate_vals']

        # Close and save excel document
        excel_doc.save_excel()
        excel_doc.close()

    def get_intermediate_results(self, excel_doc):

        step_results = excel_doc.run_function('Get_Step_Results')

        if step_results != None:
            step_results = list(step_results)
            output = {}
            for col in range(len(step_results[0])):
                output[step_results[0][col]] = []
                for row in range(1,len(step_results)):
                    output[step_results[0][col]].append(step_results[row][col])
            return output

        else:
            return None


    # def get_intermediate_results(self, excel_filename = "", references_filename = "cell_references.yaml" ):
    #     excel_doc = Excel(excel_filename, references_filename)#, save_filename = "local_calculations_enhanced_final_state.xlsm")
    #     step_results = excel_doc.run_function('Get_Step_Results')
    #     if step_results != None:
    #         step_results = list(step_results)
    #
    #         output = {}
    #         for col in range(len(step_results[0])):
    #             output[step_results[0][col]] = {}
    #             for row in range(1,len(step_results)):
    #                 output[step_results[0][col]].update({row:step_results[row][col]})
    #
    #         #print(macro_name)
    #         [print(k,v) for k,v in output.items()]
    #         print()
    #
    #     excel_doc.close()


    def find_negotiated_nodes(self, target_node_id, local_layer_name):
        out_nodes = []
        ancestors = list(nx.ancestors(self.local_K[local_layer_name].network, target_node_id))
        for node in ancestors:
            if self.local_K[local_layer_name].network.in_degree(node) == 0 and self.network.node[node]['data_status'] == 0:
                out_nodes.append(node)
        out_data = [(n,self.local_K[local_layer_name].network.node[n]['node_name']) for n in out_nodes]
        return out_data

    def find_all_unknown_nodes(self, local_layer_name):
        out_nodes = []
        for node, data in self.local_K[local_layer_name].network.nodes(data=True):
            if data['data_status'] == 0:
                out_nodes.append(node)
        out_data = [(n,self.local_K[local_layer_name].network.node[n]['node_name']) for n in out_nodes]
        return out_data

    def find_all_unknown_intermediate_nodes(self, local_layer_name):
        out_nodes = []
        for node, data in self.local_K[local_layer_name].network.nodes(data=True):
            if self.local_K[local_layer_name].network.in_degree(node) != 0 and data['data_status'] == 0:
                out_nodes.append(node)
        out_data = [(n,self.local_K[local_layer_name].network.node[n]['node_name']) for n in out_nodes]
        return out_data

    def create_IG_edge_projections(self, IG_target_node_id, local_layer_name):
        # Find all between edges from each node in I_GLOBAL. if they share the same local node, then draw an INTERNAL edge in IGLOBAL between the nodes.

        # Find BETWEEN edges from global information to local_layer_name
        edge_list = []
        for u,v,k,d in self.network.edges(keys=True, data=True):
            if d['layer'] == 'BETWEEN':
                if (self.network.node[u]['layer'] == local_layer_name and self.network.node[v]['layer'] == "I_GLOBAL") or (self.network.node[u]['layer'] == "I_GLOBAL" and self.network.node[v]['layer'] == local_layer_name):
                    edge_list.append((u,v,k))

        #print("Edgelist: ", edge_list)
        # Create new graph containing these edges
        B = nx.DiGraph(self.network.edge_subgraph(edge_list))

        # Need to add temporary edges from negotiated_nodes to target_node in IG
        edges_to_add = []
        for e in edge_list:
            if (self.network.node[e[0]]['layer'] == "I_GLOBAL" and self.network.node[e[1]]['layer'] == local_layer_name):
                edges_to_add.append((e[1], IG_target_node_id))

        B.add_edges_from(edges_to_add)
        #print("Edges to Add: ", edges_to_add)
        # Do bipartite projection
        IG_projection = bipartite.projected_graph(B, nodes = {n:d for n,d in B.nodes(data=True) if d['layer'] == "I_GLOBAL"})

        #print(IG_projection.nodes(data='node_name'))
        #print(IG_projection.edges())

        # Add projected edges to network
        for u,v,d in IG_projection.edges(data=True):
            self.network.add_edge(u,v, layer = "I_GLOBAL", type = "PROJECTION", weight = 1.0, time = self.time_step)

    def project_IG_layer_edges(self):

        G = deepcopy(self.network)
        IG_node_IDs = [n for n,d in G.nodes(data=True) if d['layer'] == "I_GLOBAL"]
        edges_to_remove = [(u,v) for u,v,d in G.edges(data=True) if d['type'] == "PROJ"]
        G.remove_edges_from(edges_to_remove)

        for node in IG_node_IDs:
            all_shortest_paths = nx.single_target_shortest_path(G,node)
            other_IG_nodes = [n for n in IG_node_IDs if n != node]

            for other_node in other_IG_nodes:
                if other_node in all_shortest_paths.keys():
                    found = False
                    for n in all_shortest_paths[other_node]:
                        if G.node[n]['layer'] == "I_GLOBAL":
                            if found == False:
                                start_node_id = n
                            else:
                                end_node_id = n
                                if self.network.has_edge(start_node_id, end_node_id) == False:
                                    self.network.add_edge(start_node_id,end_node_id, layer = "I_GLOBAL", type = "PROJ", weight = 1.0, time = self.time_step)
                                    #print("Edge Added: {} ({})".format((start_node_id, end_node_id), (G.node[start_node_id]['node_name'], G.node[end_node_id]['node_name'])))
                                break
                            found = True

    def project_KG_layer_edges(self):

        G = deepcopy(self.network)
        KG_node_IDs = [n for n,d in G.nodes(data=True) if d['layer'] == "K_GLOBAL"]
        nodes_to_remove = [n for n,d in G.nodes(data=True) if d['layer'] != "I_GLOBAL" and d['layer'] != "K_GLOBAL"]
        G.remove_nodes_from(nodes_to_remove)

        for node in KG_node_IDs:
            all_shortest_paths = nx.single_target_shortest_path(G,node)
            other_KG_nodes = [n for n in KG_node_IDs if n != node]

            for other_node in other_KG_nodes:
                if other_node in all_shortest_paths.keys():
                    found = False
                    for n in all_shortest_paths[other_node]:
                        if G.node[n]['layer'] == "K_GLOBAL":
                            if found == False:
                                start_node_id = n
                            else:
                                end_node_id = n
                                if self.network.has_edge(start_node_id, end_node_id) == False:
                                    self.network.add_edge(start_node_id,end_node_id, layer = "K_GLOBAL", type = "PROJ", weight = 1.0, time = self.time_step)
                                    #print("Edge Added: {} ({})".format((start_node_id, end_node_id), (G.node[start_node_id]['node_name'], G.node[end_node_id]['node_name'])))
                                break
                            found = True

    def update_time(self):
        # Increment time_step by 1, and update all node timeseries in all networks (local and integrated)

        for n in self.network.nodes():
            # Update data statuses
            self.calculate_data_status(n)

            #Update time series
            self.network.node[n]['val_ts'][self.time_step] = self.network.node[n]['val']
            self.network.node[n]['data_status_ts'][self.time_step] = self.network.node[n]['data_status']

            # Create projections in IG and KG layers
            self.project_IG_layer_edges()
            self.project_KG_layer_edges()

        self.time_step += 1

    def get_network_data_status_entropy(self, layer_name=None):
        # Will only work for local layers right now
        if layer_name:
            local_network = {n:d for n,d in self.network.nodes(data=True) if d['layer'] == layer_name}
        else:
            local_network = {n:d for n,d in self.network.nodes(data=True)}

        time_list = [t for t in range(self.time_step)]
        ds_dict = {}
        entropy_dict = {}
        new_entropy = {}
        for t in time_list:
            step_ds = []
            for n,d in local_network.items():
                if t in d['data_status_ts'].keys():
                    step_ds.append(d['data_status_ts'][t])
            ds_dict[t] = step_ds
            entropy_dict[t] = self.calculate_simple_entropy(step_ds)
            new_entropy[t] = 1-(entropy_dict[t]/entropy_dict[0])

        #return new_entropy
        return entropy_dict, new_entropy

    def calculate_simple_entropy(self, data_status_list):
        # A simple calculation using the data status of each data element, using the
        # P(1) and P(0). This ignores structure of the network.
        # Has a min value of 0, and a max value of 1

        num_nodes = len(data_status_list)

        if num_nodes <= 1:
            return 1

        sum_1 = 0.0
        sum_0 = 0.0
        for ds in data_status_list:
            if ds == 1.0:
                sum_1 += 1.0
            else:
                sum_0 += 1.0

        p_1 = sum_1/float(num_nodes)
        p_0 = sum_0/float(num_nodes)
        p = [p_0, p_1]
        H = entropy(p,base=2)
        return H

    def get_network_topological_entropy(self, layer_name, time):
        G = self.create_DiGraph_from_MultiDiGraph(self.network)
        if layer_name:
            nodes_to_remove = [n for n,d in G.nodes(data=True) if d['layer'] != layer_name or d['time'] > time]
        else:
            nodes_to_remove = [n for n,d in G.nodes(data=True) if d['time'] > time]

        G.remove_nodes_from(nodes_to_remove)
        edges_to_remove = [(u,v) for u,v,d in G.edges(data=True) if d['time'] > time]
        G.remove_edges_from(edges_to_remove)

        num_nodes = G.number_of_nodes()

        if num_nodes == 0:
            entropy = 0
        else:
            pagerank = self.calculate_pagerank(G)
            d = dit.ScalarDistribution(pagerank)
            entropy = dit.shannon.entropy(d)

        return entropy

    def build_topological_entropy_time_series(self, layer_name=None):
        time_list = range(self.time_step+1)
        entropy_dict = {}
        for t in time_list:
            entropy_dict[t] = self.get_network_topological_entropy(layer_name, t)

        return entropy_dict

    def calculate_pagerank(self, G, alpha = 0.85):
        pagerank = nx.pagerank(G, alpha = alpha)
        return pagerank

    def create_DiGraph_from_MultiDiGraph(self, G):
        # create weighted graph from self.network

        G1 = nx.create_empty_copy(G)
        G1 = nx.DiGraph(G1)

        for u,v,data in G.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if G1.has_edge(u,v):
                G1[u][v]['weight'] += w
            else:
                G1.add_edge(u, v, weight=w)
            G1[u][v]['layer'] = data['layer']
            G1[u][v]['type'] = data['type']
            G1[u][v]['time'] = data['time']

        return G1

###############################################################################
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


###############################################################################
class I_Global(Knowledge_Network):

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def __init__(self, data = {}, **kwargs):
        Knowledge_Network.__init__(self, **kwargs)
        self.data = data

###############################################################################
def save_pickle(dataframe, filename):
    dataframe.to_pickle('../results\\{}.pkl'.format(filename))

def get_pickle(filename):
    return pd.read_pickle('../results\\{}.pkl'.format(filename))

def get_network_pickle(filename):
    return nx.read_gpickle("../results\\{}.gpickle".format(filename))

def create_DiGraph_from_MultiDiGraph(G):
    # create weighted graph from self.network

    G1 = nx.create_empty_copy(G)
    G1 = nx.DiGraph(G1)

    for u,v,data in G.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if G1.has_edge(u,v):
            G1[u][v]['weight'] += w
        else:
            G1.add_edge(u, v, weight=w)
        G1[u][v]['layer'] = data['layer']
        G1[u][v]['type'] = data['type']
        G1[u][v]['time'] = data['time']

    return G1

def get_case_params(filename):
    if filename:
        filename = "../inputs\\{}".format(filename)
        with open(filename, 'r') as stream:
            data_loaded = yaml.safe_load(stream)
        return data_loaded
    else:
        print('Please input yaml filename.')

def export_pajek(G, filename):
    G = create_DiGraph_from_MultiDiGraph(G)
    num_nodes = G.number_of_nodes()
    node_header = "*vertices {} \n".format(num_nodes)
    edge_header = "*arcs \n"
    if filename:
        filename = "../networks\\{}".format(filename)
        file = open(filename, "w")
        file.write(node_header)

        for node_id, data in G.nodes(data = True):
            line_string = '{} "{}" \n'.format(node_id, data['node_name'])
            file.write(line_string)

        file.write(edge_header)

        for u,v, data in G.edges(data=True):
            line_string = "{} {} {} \n".format(u,v,data["weight"])
            file.write(line_string)

        file.close()
    else:
        print('Please input filename.')

def export_for_gephi(G,filename):

    #G = create_DiGraph_from_MultiDiGraph(G)
    if filename:
        node_filename = "../networks\\{}_node_sheet.csv".format(filename)
        edge_filename = "../networks\\{}_edge_sheet.csv".format(filename)
        # Write node sheet
        with open(node_filename, 'w', newline='') as csvFile:
            fieldnames = ['Id', "Label", "timeset", "Layer", "Layer[z]"]
            layer_num = {
                "K_GLOBAL": 0,
                "I_GLOBAL": 1,
                "NAVARCH": 2,
                "OPS": 2,
                "DIST": 2,
            }
            writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
            writer.writeheader()
            for node_id, data in G.nodes(data = True):
                min_time = min(data['val_ts'].keys())
                max_time = max(data['val_ts'].keys())
                line_dict = {
                    "Id": node_id,
                    "Label": data['node_name'],
                    "timeset": "\"<[{},{}]>\"".format(min_time, max_time),
                    "Layer": data['layer'],
                    "Layer[z]": layer_num[data['layer']],
                    }
                writer.writerow(line_dict)

        # Write edge sheet
        with open(edge_filename, 'w', newline='') as csvFile:
            fieldnames = ['Source', "Target", "Weight", "Type", "Layer", 'timeset', "Label"]
            writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
            writer.writeheader()
            for u,v, data in G.edges(data = True):
                min_time = data['time']
                line_dict = {
                    "Source": u,
                    "Target": v,
                    "Weight": data['weight'],
                    "Type": data['type'],
                    "Layer": data['layer'],
                    "timeset": "\"<[{},{}]>\"".format(min_time, max_time),
                    "Label": data['type']
                    }
                writer.writerow(line_dict)


    else:
        print('Please input filename.')

def run_case_study(case_study_filename):

    case_params = get_case_params(case_study_filename)

    case_name = case_params["name"]
    layer_sequence = case_params["layer_sequence"]
    KG_target_nodes = case_params["KG_target_nodes"]
    excel_filename = case_params["excel_filename"]
    cell_references_filename = case_params["references_filename"]

    KIF = Integrated_Framework()
    KIF.build_layered_network()

    for KG_target_node_name in KG_target_nodes:
        # Create KG target node and grow IG from KG
        KIF.add_KG_target_node(KG_target_node_name)
        KIF.grow_IG_from_KG(KG_target_node_name)
        # Add created IG node to the list of target nodes
        KIF.global_I.target_node.append(KIF.get_node_id(KG_target_node_name, layer = "I_GLOBAL"))
        KIF.update_time()

        for layer in layer_sequence:
            # Store a list of target nodes in current layer
            local_target_node_names = [d['node_name'] for n,d in KIF.local_K[layer].network.nodes(data=True) if d['type'] == 'TARGET']


            if KG_target_node_name in local_target_node_names:
                # Select a local node from created IG node
                local_node_id = KIF.select_KL_node_from_IG(KG_target_node_name, layer)
                KIF.update_time()
                #print(local_node_id, KIF.local_K[layer].network.node[local_node_id])
                # Find local nodes to transfer to IG
                negotiated_nodes = KIF.find_negotiated_nodes(local_node_id, layer)

                #### COULD ADD LOGIC TO BUILD LOCAL LAYER KNOWLEDGE STRUCTURE HERE RATHER THAN LOOKING TO OTHER LAYERS

                # Grow IG from KL
                IG_negotiated_nodes = []
                for node_id, node_name in negotiated_nodes:
                    new_node_id = KIF.grow_IG_from_KL(node_name, layer)
                    IG_negotiated_nodes.append(new_node_id)
                KIF.update_time()

                # Find other local layers to search (other than current local layer)
                layers_to_search = {lay:KIF.local_K[lay].network for lay in layer_sequence if lay != layer}

                # Find nodes in other local structures to populate IG node values
                found_nodes = {}
                for lay, network in layers_to_search.items():
                    local_nodes = {node_id:node_data for node_id, node_data in network.nodes(data=True)}
                    local_node_names = [node_data['node_name'] for node_id, node_data in local_nodes.items()]
                    for IG_node_id, IG_node_name in IG_negotiated_nodes:
                        if IG_node_name in local_node_names:
                            # get node data for local node
                            found_nodes.update({node_id:node_data for node_id, node_data in local_nodes.items() if node_data['node_name'] == IG_node_name})

                # If found_nodes is empty (meaning there are no other structures), then just communicate to global IG
                if len(found_nodes) == 0:
                    pass

                else:

                    # Prioritize what nodes to select based on node type (prioritize target nodes), and select them from local layer
                    new_found_nodes = {}
                    for found_node_id, found_node_data in found_nodes.items():
                        if found_node_data['type'] == "TARGET":
                            new_found_nodes.update({found_node_id: found_node_data})
                            KIF.select_KL_node_from_IG(found_node_data['node_name'], found_node_data['layer'])
                    KIF.update_time()

                    # Send the values from the selected nodes back to Global information
                    for found_node_id, found_node_data in new_found_nodes.items():
                        KIF.send_node_value(found_node_data['node_name'], found_node_data['layer'], found_node_data['node_name'], "I_GLOBAL") ## POTENTIAL ISSUE
                    KIF.update_time()

                    # Send the values from the global information layer back to the original layers
                    for node_id, node_name in negotiated_nodes:
                        KIF.send_node_value(node_name, "I_GLOBAL", node_name, layer)
                    KIF.update_time()

                # Do calculation in original layers
                KIF.do_local_calculations(layer)

                # Send new calculated value from layer target node to global information
                KIF.send_node_value(KG_target_node_name, layer, KG_target_node_name, "I_GLOBAL")
                # Can change this potentially wqith new node 'type' property
                #IG_target_node_id = KIF.get_node_id(KG_target_node_name, layer = "I_GLOBAL")
                #KIF.create_IG_edge_projections(IG_target_node_id, local_layer_name = layer)
                KIF.update_time()

                # Send global information value back to global knowledge
                KIF.send_node_value(KG_target_node_name, "I_GLOBAL", KG_target_node_name, "K_GLOBAL")
                KIF.update_time()

                #[print(k,v['node_name'], v['type']) for k,v in new_found_nodes.items()]


                #print(new_found_nodes)
                #temp_local_node_id = KIF.select_KL_node_from_IG(node_name, layer)




    df = KIF.create_dataframe()
    df.to_excel("../results\\{}_temp.xlsx".format(case_name))

    KIF.draw_framework()
    #KIF.draw_layer("I_GLOBAL")

def run_simple_case(case_study_filename):

    case_params = get_case_params(case_study_filename)

    case_name = case_params["name"]
    layer_sequence = case_params["layer_sequence"]
    KG_target_nodes = case_params["KG_target_nodes"]
    KG_unknown_target_node_sequence = case_params["KG_sequence"]
    excel_filename = case_params["excel_filename"]
    cell_references_filename = case_params["references_filename"]

    # Open instance of excel spreadsheet, and clear workbooks for processing
    excel_doc = Excel(excel_filename, cell_references_filename)
    excel_doc.run_macro("Reset_Workbook")
    excel_doc.save_excel()
    excel_doc.close()

    KIF = Integrated_Framework()
    KIF.build_layered_network()

    # Add all target nodes to Global K
    for KG_target_node_name, KG_target_node_val in KG_target_nodes.items():
        KIF.add_KG_target_node(KG_target_node_name, **{"value": KG_target_node_val})
    KIF.update_time()

    # Grow Global I from Global K
    for KG_target_node_name, KG_target_node_val in KG_target_nodes.items():
        KIF.grow_IG_from_KG(KG_target_node_name)
        KIF.global_I.target_node.append(KIF.get_node_id(KG_target_node_name, layer = "I_GLOBAL"))
    KIF.update_time()

    # Start with selecting GMT

    local_node_id = KIF.select_KL_node_from_IG("GMT", "NAVARCH")
    negotiated_nodes = KIF.find_negotiated_nodes(local_node_id, "NAVARCH")
    KIF.update_time()

    # Grow IG from KL
    IG_negotiated_nodes = []
    for node_id, node_name in negotiated_nodes:
        new_node_id = KIF.grow_IG_from_KL(node_name, "NAVARCH")
        IG_negotiated_nodes.append(new_node_id)
    KIF.update_time()

    # Select nodes in OPS from newly grown IG
    # Find nodes in other local structures to populate IG node values
    layers_to_search = {lay:KIF.local_K[lay].network for lay in layer_sequence if lay == "OPS"}
    found_nodes = {}
    for lay, network in layers_to_search.items():
        local_nodes = {node_id:node_data for node_id, node_data in network.nodes(data=True)}
        local_node_names = [node_data['node_name'] for node_id, node_data in local_nodes.items()]
        for IG_node_id, IG_node_name in IG_negotiated_nodes:
            if IG_node_name in local_node_names:
                # get node data for local node
                found_nodes.update({node_id:node_data for node_id, node_data in local_nodes.items() if node_data['node_name'] == IG_node_name})

    for IG_node, IG_node_data in found_nodes.items():
        if IG_node_data['node_name'] != "z_fuel":
            local_node_id = KIF.select_KL_node_from_IG(IG_node_data['node_name'], "OPS")
    KIF.update_time()

    # Select vehicle numbers in I_GLOBAL needed by OPS
    for KG_target_node_name, KG_target_node_val in KG_target_nodes.items():
        if KG_target_node_val != None:
            KIF.select_IG_node_from_KL(IG_node_name = KG_target_node_name, local_layer = "OPS")
    KIF.update_time()

    # Initiate Values in OPS LAYER
    for KG_target_node_name, KG_target_node_val in KG_target_nodes.items():
        if KG_target_node_val != None:
            local_node_id = KIF.select_KL_node_from_IG(KG_target_node_name, "OPS")
            KIF.send_node_value(start_node_name=KG_target_node_name, start_layer="I_GLOBAL",end_node_name=KG_target_node_name, end_layer="OPS")

    KIF.do_local_calculations(local_layer_name = "OPS", excel_filename = excel_filename, macro_name = "Calculate_OPS_z")
    KIF.update_time()

    # Send the values from the selected OPS nodes back to Global information
    for found_node_id, found_node_data in found_nodes.items():
        KIF.send_node_value(found_node_data['node_name'], found_node_data['layer'], found_node_data['node_name'], "I_GLOBAL") ## POTENTIAL ISSUE
    KIF.update_time()

    # Send the values from the global information layer back to the original layers
    for node_id, node_name in negotiated_nodes:
        KIF.send_node_value(node_name, "I_GLOBAL", node_name, "NAVARCH")
    KIF.do_local_calculations(local_layer_name = "NAVARCH", excel_filename = excel_filename, macro_name = "Calculate_NAVARCH_z_fuel")
    KIF.update_time()

    # Send GMT Value to IG
    KIF.send_node_value("GMT", "NAVARCH", "GMT", "I_GLOBAL")
    KIF.update_time()

    # NEXT DO %

    # Get Value from NAVARCH and communicate to IG.
    local_node_id = KIF.select_KL_node_from_IG("%", "NAVARCH")
    KIF.update_time()

    # Communicate the Trim value to the IG layer
    KIF.send_node_value("%", "NAVARCH", "%", "I_GLOBAL")
    KIF.update_time()

    # NEXT DO TRIM

    # Select Trim node in NAVARCH layer from IG layer
    local_node_id = KIF.select_KL_node_from_IG("Trim", "NAVARCH")
    KIF.update_time()

    # Do the calculations in the NAVARCH LAYER
    KIF.do_local_calculations(local_layer_name = "NAVARCH", excel_filename = excel_filename, macro_name = "Calculate_Trim")
    #KIF.update_time()

    # Communicate the Trim value to the IG layer
    KIF.send_node_value("Trim", "NAVARCH", "Trim", "I_GLOBAL")
    KIF.update_time()

    # Now have OPS decide where the vehicles need to go.
    OPS_unknown_nodes = KIF.find_all_unknown_nodes("OPS")
    OPS_unknown_intermediate_nodes = KIF.find_all_unknown_intermediate_nodes("OPS")

    # Grow IG from KL
    IG_negotiated_nodes = []
    for node_id, node_name in OPS_unknown_intermediate_nodes:
        new_node_id = KIF.grow_IG_from_KL(node_name, "OPS")
        IG_negotiated_nodes.append(new_node_id)
    KIF.update_time()

    # Select nodes in NAVARCH from newly grown IG
    # Find nodes in other local structures to populate IG node values
    layers_to_search = {lay:KIF.local_K[lay].network for lay in layer_sequence if lay == "NAVARCH"}
    found_nodes = {}
    for lay, network in layers_to_search.items():
        local_nodes = {node_id:node_data for node_id, node_data in network.nodes(data=True)}
        local_node_names = [node_data['node_name'] for node_id, node_data in local_nodes.items()]
        for IG_node_id, IG_node_name in IG_negotiated_nodes:
            if IG_node_name in local_node_names:
                # get node data for local node
                found_nodes.update({node_id:node_data for node_id, node_data in local_nodes.items() if node_data['node_name'] == IG_node_name})

    # Select the node in NAVARCH
    for IG_node, IG_node_data in found_nodes.items():
        local_node_id = KIF.select_KL_node_from_IG(IG_node_data['node_name'], "NAVARCH")
    KIF.update_time()

    # Communicate the selected node to the IG layer
    for found_node_id, found_node_data in found_nodes.items():
        KIF.send_node_value(found_node_data['node_name'], found_node_data['layer'], found_node_data['node_name'], "I_GLOBAL") ## POTENTIAL ISSUE
    KIF.update_time()

    # Send updated Values to OPS
    # Have OPS do their calculation to select the x,v for each vehicle. If exact solution found, done. Else, transmit new val to IG, then to NAVARCH
    # Communicate IG node back to OPS   ~~~~THIS IS WHERE THE RECALC CODE NEEDS TO BE ADDED FOR MULTIPLE VEHICLES. IF STATEMENT REQUIED!!!!!!!!!
    for node_id, node_name in OPS_unknown_intermediate_nodes:
        KIF.send_node_value(node_name, "I_GLOBAL", node_name, "OPS")
    KIF.do_local_calculations(local_layer_name = "OPS", excel_filename = excel_filename, macro_name = "Calculate_OPS_Vehicle_Locs")
    KIF.update_time()

    # NEXT DO REQUIRED POWER

    # Identify node in DIST
    local_node_id = KIF.select_KL_node_from_IG("required_power", "DIST")
    negotiated_nodes = KIF.find_negotiated_nodes(local_node_id, "DIST")
    KIF.update_time()


    IG_negotiated_nodes = []
    for node_id, node_name in negotiated_nodes:
        if node_name == "pipe_diameter":
            pass
        elif node_name == "vol_flow_rate":
            new_node_id = KIF.grow_IG_from_KL(node_name, "DIST")
            IG_negotiated_nodes.append(new_node_id)
        else:
            new_node_id = KIF.select_IG_node_from_KL(node_name,"DIST")

    KIF.update_time()

    # Transmit back z_veh, ask NAVARCH for z_fuel, and ask OPS for flow rate in single time step
    KIF.send_node_value("z_veh", "I_GLOBAL", "z_veh", "DIST")
    KIF.select_KL_node_from_IG("z_fuel", "NAVARCH")
    KIF.select_KL_node_from_IG("vol_flow_rate","OPS")
    KIF.update_time()

    # Send OPS and NAVARCH values back to IG
    KIF.send_node_value("vol_flow_rate", "OPS", "vol_flow_rate", "I_GLOBAL")
    KIF.send_node_value("z_fuel", "NAVARCH", "z_fuel", "I_GLOBAL")
    KIF.update_time()

    # Send IG value to DIST and do calculation
    KIF.send_node_value("vol_flow_rate", "I_GLOBAL", "vol_flow_rate", "DIST")
    KIF.send_node_value("z_fuel", "I_GLOBAL", "z_fuel", "DIST")
    KIF.do_local_calculations(local_layer_name = "DIST", excel_filename = excel_filename, macro_name = "Calculate_Power_Req")
    KIF.do_local_calculations(local_layer_name = "DIST", excel_filename = excel_filename, macro_name = "Calculate_Power_Req")
    KIF.update_time()

    # Send calculated power req to I_GLOBAL
    KIF.send_node_value("required_power", "DIST", "required_power", "I_GLOBAL")
    KIF.update_time()

    # NEXT DO PIPE_DIAMETER

    # Get Value from NAVARCH and communicate to IG.
    local_node_id = KIF.select_KL_node_from_IG("pipe_diameter", "DIST")
    KIF.update_time()

    # Communicate the Trim value to the IG layer
    KIF.send_node_value("pipe_diameter", "DIST", "pipe_diameter", "I_GLOBAL")
    KIF.update_time()

    # Communicate all values back to K_GLOBAL from I_GLOBAL
    #IG_unknown_nodes = {n:d for n,d in KIF.network.nodes(data=True) if d['layer'] == "I_GLOBAL" and not (d['node_name'] == 'n_F35' or d['node_name'] == 'n_AV8B' or d['node_name'] == 'n_SH60' or d['node_name'] == 'n_V22')}

    for n, d in KIF.global_K.network.nodes(data=True):
        if d['data_status'] == 0:
            KIF.send_node_value(d['node_name'],"I_GLOBAL", d['node_name'], "K_GLOBAL")

    KIF.update_time()

    # Save the results to a pickle and to excel doc
    #df = KIF.create_dataframe()
    #save_pickle(df, "{}".format(case_name))
    #df.to_excel("../results\\{}.xlsx".format(case_name))

    #KIF.create_node_dataframe()

    #KIF.save_network_pickle("{}_network".format(case_name))

    #filename = "KIF_pickle.p"
    #pkl.dump(KIF,"../results\\{}".format(filename))

    # temp_ts = {}
    # KIF.build_entropy_time_series(temp_ts, KIF.calculate_simple_entropy('OPS'))
    # print(temp_ts)



    # --------------------- PLOTTING FIGURES ----------------------

    # ------------------------ Topological Entropy ----------------------

    TE_dict = {
        "OPS" : KIF.build_topological_entropy_time_series("OPS"),
        "NAVARCH" : KIF.build_topological_entropy_time_series("NAVARCH"),
        "DIST" : KIF.build_topological_entropy_time_series("DIST"),
        "I_GLOBAL" : KIF.build_topological_entropy_time_series("I_GLOBAL"),
        "K_GLOBAL" : KIF.build_topological_entropy_time_series("K_GLOBAL"),
        "ALL": KIF.build_topological_entropy_time_series()
    }

    for name, vals in TE_dict.items():
        KIF.save_entropy_time_series_plot(vals,
            filename = '{}_TE_SIMPLE'.format(name),
            xlabel = 'Time',
            ylabel = 'Topological Entropy',
            title = '{}'.format(name),
            x_min = 0,
            x_max = KIF.time_step,
            y_min = -0.01,
            y_max=7,
            grid = True
            )

    #------------------------ Data Status Entropy ----------------------

    OPS_DSE, OPS_DSE_New = KIF.get_network_data_status_entropy("OPS")
    NAVARCH_DSE, NAVARCH_DSE_New = KIF.get_network_data_status_entropy("NAVARCH")
    DIST_DSE, DIST_DSE_New = KIF.get_network_data_status_entropy("DIST")
    I_GLOBAL_DSE, I_GLOBAL_DSE_New = KIF.get_network_data_status_entropy("I_GLOBAL")
    K_GLOBAL_DSE, K_GLOBAL_DSE_New = KIF.get_network_data_status_entropy("K_GLOBAL")
    ALL_DSE, ALL_DSE_New = KIF.get_network_data_status_entropy()
    DSE_dict = {
        "OPS" : OPS_DSE,
        "NAVARCH" : NAVARCH_DSE ,
        "DIST" : DIST_DSE,
        "I_GLOBAL" : I_GLOBAL_DSE,
        "K_GLOBAL" : K_GLOBAL_DSE,
        "ALL": ALL_DSE
    }
    DSE_new_dict = {
        "OPS" : OPS_DSE_New,
        "NAVARCH" : NAVARCH_DSE_New ,
        "DIST" : DIST_DSE_New,
        "I_GLOBAL" : I_GLOBAL_DSE_New,
        "K_GLOBAL" : K_GLOBAL_DSE_New,
        "ALL": ALL_DSE_New
    }

    for name, vals in DSE_dict.items():
        KIF.save_entropy_time_series_plot(vals,
            filename = '{}_DSE_SIMPLE'.format(name),
            xlabel = 'Time',
            ylabel = 'Data Status Entropy',
            title = '{}'.format(name),
            x_min = 0,
            x_max = KIF.time_step,
            y_min = -0.01,
            y_max=1.01,
            grid = True
            )
    for name, vals in DSE_new_dict.items():
        KIF.save_entropy_time_series_plot(vals,
            filename = '{}_DSE_New_SIMPLE'.format(name),
            xlabel = 'Time',
            ylabel = 'Data Status Entropy (NEW)',
            title = '{}'.format(name),
            x_min = 0,
            x_max = KIF.time_step,
            y_min = -0.01,
            y_max=1.01,
            grid = True
            )

    KIF.save_network_pickle("{}_network".format(case_name))
    edge_df = KIF.create_dataframe()
    node_df = KIF.create_node_dataframe()

    save_pickle(edge_df, "{}_edge_data".format(case_name))
    save_pickle(node_df, "{}_node_data".format(case_name))

    #KIF.draw_framework()
    #KIF.draw_layer("I_GLOBAL")


def run_hard_case(case_study_filename):

    case_params = get_case_params(case_study_filename)

    case_name = case_params["name"]
    layer_sequence = case_params["layer_sequence"]
    KG_target_nodes = case_params["KG_target_nodes"]
    KG_unknown_target_node_sequence = case_params["KG_sequence"]
    excel_filename = case_params["excel_filename"]
    cell_references_filename = case_params["references_filename"]

    # Open instance of excel spreadsheet, and clear workbooks for processing
    excel_doc = Excel(excel_filename, cell_references_filename)
    excel_doc.run_macro("Reset_Workbook")
    excel_doc.save_excel()
    excel_doc.close()

    KIF = Integrated_Framework()
    KIF.build_layered_network()

    # Add all target nodes to Global K
    for KG_target_node_name, KG_target_node_val in KG_target_nodes.items():
        KIF.add_KG_target_node(KG_target_node_name, **{"value": KG_target_node_val})
    KIF.update_time()

    # Grow Global I from Global K
    for KG_target_node_name, KG_target_node_val in KG_target_nodes.items():
        KIF.grow_IG_from_KG(KG_target_node_name)
        KIF.global_I.target_node.append(KIF.get_node_id(KG_target_node_name, layer = "I_GLOBAL"))
    KIF.update_time()

    # Start with selecting GMT

    local_node_id = KIF.select_KL_node_from_IG("GMT", "NAVARCH")
    negotiated_nodes = KIF.find_negotiated_nodes(local_node_id, "NAVARCH")
    KIF.update_time()

    # Grow IG from KL
    IG_negotiated_nodes = []
    for node_id, node_name in negotiated_nodes:
        new_node_id = KIF.grow_IG_from_KL(node_name, "NAVARCH")
        IG_negotiated_nodes.append(new_node_id)
    KIF.update_time()

    # Select nodes in OPS from newly grown IG
    # Find nodes in other local structures to populate IG node values
    layers_to_search = {lay:KIF.local_K[lay].network for lay in layer_sequence if lay == "OPS"}
    found_nodes = {}
    for lay, network in layers_to_search.items():
        local_nodes = {node_id:node_data for node_id, node_data in network.nodes(data=True)}
        local_node_names = [node_data['node_name'] for node_id, node_data in local_nodes.items()]
        for IG_node_id, IG_node_name in IG_negotiated_nodes:
            if IG_node_name in local_node_names:
                # get node data for local node
                found_nodes.update({node_id:node_data for node_id, node_data in local_nodes.items() if node_data['node_name'] == IG_node_name})

    for IG_node, IG_node_data in found_nodes.items():
        if IG_node_data['node_name'] != "z_fuel":
            local_node_id = KIF.select_KL_node_from_IG(IG_node_data['node_name'], "OPS")
    KIF.update_time()

    # Select vehicle numbers in I_GLOBAL needed by OPS
    for KG_target_node_name, KG_target_node_val in KG_target_nodes.items():
        if KG_target_node_val != None:
            KIF.select_IG_node_from_KL(IG_node_name = KG_target_node_name, local_layer = "OPS")
    KIF.update_time()


    # Initiate Values in OPS LAYER
    for KG_target_node_name, KG_target_node_val in KG_target_nodes.items():
        if KG_target_node_val != None:
            local_node_id = KIF.select_KL_node_from_IG(KG_target_node_name, "OPS")
            KIF.send_node_value(start_node_name=KG_target_node_name, start_layer="I_GLOBAL",end_node_name=KG_target_node_name, end_layer="OPS")

    KIF.do_local_calculations(local_layer_name = "OPS", excel_filename = excel_filename, macro_name = "Calculate_OPS_z")
    #KIF.get_intermediate_results(excel_filename = excel_filename, references_filename = "cell_references.yaml" )
    KIF.update_time()

    # Send the values from the selected OPS nodes back to Global information
    for found_node_id, found_node_data in found_nodes.items():
        KIF.send_node_value(found_node_data['node_name'], found_node_data['layer'], found_node_data['node_name'], "I_GLOBAL") ## POTENTIAL ISSUE
    KIF.update_time()

    # Send the values from the global information layer back to the original layers
    for node_id, node_name in negotiated_nodes:
        KIF.send_node_value(node_name, "I_GLOBAL", node_name, "NAVARCH")
    KIF.do_local_calculations(local_layer_name = "NAVARCH", excel_filename = excel_filename, macro_name = "Calculate_NAVARCH_z_fuel")
    #KIF.get_intermediate_results(excel_filename = excel_filename, references_filename = "cell_references.yaml" )
    KIF.update_time()

    # Send GMT Value to IG
    KIF.send_node_value("GMT", "NAVARCH", "GMT", "I_GLOBAL")
    KIF.update_time()

    # NEXT DO %

    # Get Value from NAVARCH and communicate to IG.
    local_node_id = KIF.select_KL_node_from_IG("%", "NAVARCH")
    KIF.update_time()

    # Communicate the Trim value to the IG layer
    KIF.send_node_value("%", "NAVARCH", "%", "I_GLOBAL")
    KIF.update_time()

    # NEXT DO TRIM

    # Select Trim node in NAVARCH layer from IG layer
    local_node_id = KIF.select_KL_node_from_IG("Trim", "NAVARCH")
    KIF.update_time()

    # Do the calculations in the NAVARCH LAYER
    KIF.do_local_calculations(local_layer_name = "NAVARCH", excel_filename = excel_filename, macro_name = "Calculate_Trim")
    #KIF.get_intermediate_results(excel_filename = excel_filename, references_filename = "cell_references.yaml" )
    #KIF.update_time()

    # Communicate the Trim value to the IG layer
    KIF.send_node_value("Trim", "NAVARCH", "Trim", "I_GLOBAL")
    KIF.update_time()

    # Now have OPS decide where the vehicles need to go.
    OPS_unknown_nodes = KIF.find_all_unknown_nodes("OPS")
    OPS_unknown_intermediate_nodes = KIF.find_all_unknown_intermediate_nodes("OPS")

    # Grow IG from KL
    IG_negotiated_nodes = []
    for node_id, node_name in OPS_unknown_intermediate_nodes:
        new_node_id = KIF.grow_IG_from_KL(node_name, "OPS")
        IG_negotiated_nodes.append(new_node_id)
    KIF.update_time()

    # Select nodes in NAVARCH from newly grown IG
    # Find nodes in other local structures to populate IG node values
    layers_to_search = {lay:KIF.local_K[lay].network for lay in layer_sequence if lay == "NAVARCH"}
    found_nodes = {}
    for lay, network in layers_to_search.items():
        local_nodes = {node_id:node_data for node_id, node_data in network.nodes(data=True)}
        local_node_names = [node_data['node_name'] for node_id, node_data in local_nodes.items()]
        for IG_node_id, IG_node_name in IG_negotiated_nodes:
            if IG_node_name in local_node_names:
                # get node data for local node
                found_nodes.update({node_id:node_data for node_id, node_data in local_nodes.items() if node_data['node_name'] == IG_node_name})

    # Select the node in NAVARCH
    for IG_node, IG_node_data in found_nodes.items():
        local_node_id = KIF.select_KL_node_from_IG(IG_node_data['node_name'], "NAVARCH")
    KIF.update_time()

    # Communicate the selected node to the IG layer
    for found_node_id, found_node_data in found_nodes.items():
        KIF.send_node_value(found_node_data['node_name'], found_node_data['layer'], found_node_data['node_name'], "I_GLOBAL") ## POTENTIAL ISSUE
    KIF.update_time()

    # Send updated Values to OPS
    for node_id, node_name in OPS_unknown_intermediate_nodes:
        KIF.send_node_value(node_name, "I_GLOBAL", node_name, "OPS")

    # Have OPS do their calculation to select the x,v for each vehicle. If exact solution found, done. Else, transmit new val to IG, then to NAVARCH
    KIF.do_local_calculations(local_layer_name = "OPS", excel_filename = excel_filename, macro_name = "Calculate_OPS_Vehicle_Locs") ######## USED TO BE "Recalculate_OPS"
    #KIF.get_intermediate_results(excel_filename = excel_filename, references_filename = "cell_references.yaml" )
    KIF.update_time()

    # Communicate OPS node back to IG
    KIF.send_node_value('x_veh', "OPS", 'x_veh', "I_GLOBAL")
    KIF.update_time()

    # Communicate IG node to NAVARCH and Have NAVARCH Recalculate their values
    KIF.send_node_value('x_veh', "I_GLOBAL", 'x_veh', "NAVARCH")
    KIF.do_local_calculations(local_layer_name = "NAVARCH", excel_filename = excel_filename)
    KIF.update_time()

    # Have NAVARCH recalculate trim (since its out of bounds)
    KIF.do_local_calculations(local_layer_name = "NAVARCH", excel_filename = excel_filename, macro_name = 'Recalculate_Trim')
    #KIF.get_intermediate_results(excel_filename = excel_filename, references_filename = "cell_references.yaml" )
    KIF.update_time()

    # Communicate the Trim value to the IG layer
    KIF.send_node_value("Trim", "NAVARCH", "Trim", "I_GLOBAL")
    KIF.update_time()

    # NEXT DO REQUIRED POWER

    # Identify node in DIST
    local_node_id = KIF.select_KL_node_from_IG("required_power", "DIST")
    negotiated_nodes = KIF.find_negotiated_nodes(local_node_id, "DIST")
    KIF.update_time()


    IG_negotiated_nodes = []
    for node_id, node_name in negotiated_nodes:
        if node_name == "pipe_diameter":
            pass
        elif node_name == "vol_flow_rate":
            new_node_id = KIF.grow_IG_from_KL(node_name, "DIST")
            IG_negotiated_nodes.append(new_node_id)
        else:
            new_node_id = KIF.select_IG_node_from_KL(node_name,"DIST")

    KIF.update_time()

    # Transmit back z_veh, ask NAVARCH for z_fuel, and ask OPS for flow rate in single time step
    KIF.send_node_value("z_veh", "I_GLOBAL", "z_veh", "DIST")
    KIF.select_KL_node_from_IG("z_fuel", "NAVARCH")
    KIF.select_KL_node_from_IG("vol_flow_rate","OPS")
    KIF.update_time()

    # Send OPS and NAVARCH values back to IG
    KIF.send_node_value("vol_flow_rate", "OPS", "vol_flow_rate", "I_GLOBAL")
    KIF.send_node_value("z_fuel", "NAVARCH", "z_fuel", "I_GLOBAL")
    KIF.update_time()

    # Send IG value to DIST and do calculation
    KIF.send_node_value("vol_flow_rate", "I_GLOBAL", "vol_flow_rate", "DIST")
    KIF.send_node_value("z_fuel", "I_GLOBAL", "z_fuel", "DIST")
    KIF.do_local_calculations(local_layer_name = "DIST", excel_filename = excel_filename, macro_name = "Calculate_Power_Req")
    KIF.do_local_calculations(local_layer_name = "DIST", excel_filename = excel_filename, macro_name = "Calculate_Power_Req")
    #KIF.get_intermediate_results(excel_filename = excel_filename, references_filename = "cell_references.yaml" )
    KIF.update_time()

    # Recalculate z_fuel since it no longer works
    KIF.do_local_calculations(local_layer_name = "DIST", excel_filename = excel_filename, macro_name = "Recalculate_DIST")
    #KIF.get_intermediate_results(excel_filename = excel_filename, references_filename = "cell_references.yaml" )
    KIF.update_time()

    # Communicate z_fuel to IG
    KIF.send_node_value("z_fuel","DIST","z_fuel","I_GLOBAL")
    KIF.update_time()

    # Communicte z_fuel in IG to NAVARCH and have them redo calc
    KIF.send_node_value("z_fuel", "I_GLOBAL", "z_fuel", "NAVARCH")
    KIF.do_local_calculations(local_layer_name = "NAVARCH", excel_filename = excel_filename)
    KIF.update_time()

    # Send new GMT to IG from NAVARCH
    KIF.send_node_value("GMT", "NAVARCH", "GMT", "I_GLOBAL")
    KIF.update_time()

    # Send calculated power req to I_GLOBAL
    KIF.send_node_value("required_power", "DIST", "required_power", "I_GLOBAL")
    KIF.update_time()

    # NEXT DO PIPE_DIAMETER

    # Get Value from NAVARCH and communicate to IG.
    local_node_id = KIF.select_KL_node_from_IG("pipe_diameter", "DIST")
    KIF.update_time()

    # Communicate the Trim value to the IG layer
    KIF.send_node_value("pipe_diameter", "DIST", "pipe_diameter", "I_GLOBAL")
    KIF.update_time()

    # Communicate all values back to K_GLOBAL from I_GLOBAL
    #IG_unknown_nodes = {n:d for n,d in KIF.network.nodes(data=True) if d['layer'] == "I_GLOBAL" and not (d['node_name'] == 'n_F35' or d['node_name'] == 'n_AV8B' or d['node_name'] == 'n_SH60' or d['node_name'] == 'n_V22')}

    for n, d in KIF.global_K.network.nodes(data=True):
        if d['data_status'] == 0:
            KIF.send_node_value(d['node_name'],"I_GLOBAL", d['node_name'], "K_GLOBAL")
    KIF.update_time()

    # --------------------- PLOTTING FIGURES ----------------------

    # ------------------------ Topological Entropy ----------------------

    TE_dict = {
        "OPS" : KIF.build_topological_entropy_time_series("OPS"),
        "NAVARCH" : KIF.build_topological_entropy_time_series("NAVARCH"),
        "DIST" : KIF.build_topological_entropy_time_series("DIST"),
        "I_GLOBAL" : KIF.build_topological_entropy_time_series("I_GLOBAL"),
        "K_GLOBAL" : KIF.build_topological_entropy_time_series("K_GLOBAL"),
        "ALL": KIF.build_topological_entropy_time_series()
    }

    for name, vals in TE_dict.items():
        KIF.save_entropy_time_series_plot(vals,
            filename = '{}_TE_HARD'.format(name),
            xlabel = 'Time',
            ylabel = 'Topological Entropy',
            title = '{}'.format(name),
            x_min = 0,
            x_max = KIF.time_step,
            y_min = -0.01,
            y_max=7,
            grid = True
            )

    #------------------------ Data Status Entropy ----------------------

    OPS_DSE, OPS_DSE_New = KIF.get_network_data_status_entropy("OPS")
    NAVARCH_DSE, NAVARCH_DSE_New = KIF.get_network_data_status_entropy("NAVARCH")
    DIST_DSE, DIST_DSE_New = KIF.get_network_data_status_entropy("DIST")
    I_GLOBAL_DSE, I_GLOBAL_DSE_New = KIF.get_network_data_status_entropy("I_GLOBAL")
    K_GLOBAL_DSE, K_GLOBAL_DSE_New = KIF.get_network_data_status_entropy("K_GLOBAL")
    ALL_DSE, ALL_DSE_New = KIF.get_network_data_status_entropy()

    DSE_dict = {
        "OPS" : OPS_DSE,
        "NAVARCH" : NAVARCH_DSE ,
        "DIST" : DIST_DSE,
        "I_GLOBAL" : I_GLOBAL_DSE,
        "K_GLOBAL" : K_GLOBAL_DSE,
        "ALL": ALL_DSE
    }
    DSE_new_dict = {
        "OPS" : OPS_DSE_New,
        "NAVARCH" : NAVARCH_DSE_New ,
        "DIST" : DIST_DSE_New,
        "I_GLOBAL" : I_GLOBAL_DSE_New,
        "K_GLOBAL" : K_GLOBAL_DSE_New,
        "ALL": ALL_DSE_New
    }

    for name, vals in DSE_dict.items():
        KIF.save_entropy_time_series_plot(vals,
            filename = '{}_DSE_HARD'.format(name),
            xlabel = 'Time',
            ylabel = 'Data Status Entropy',
            title = '{}'.format(name),
            x_min = 0,
            x_max = KIF.time_step,
            y_min = -0.01,
            y_max=1.01,
            grid = True
            )
    for name, vals in DSE_new_dict.items():
        KIF.save_entropy_time_series_plot(vals,
            filename = '{}_DSE_New_HARD'.format(name),
            xlabel = 'Time',
            ylabel = 'Data Status Entropy (NEW)',
            title = '{}'.format(name),
            x_min = 0,
            x_max = KIF.time_step,
            y_min = -0.01,
            y_max=1.01,
            grid = True
            )

    KIF.save_network_pickle("{}_network".format(case_name))
    edge_df = KIF.create_dataframe()
    node_df = KIF.create_node_dataframe()

    save_pickle(edge_df, "{}_edge_data".format(case_name))
    save_pickle(node_df, "{}_node_data".format(case_name))

    #KIF.draw_framework()

###############################################################################
def main():

    #GMT_case_study("case_study_parameters.yaml")
    #run_simple_case("simple_case_study_parameters.yaml")
    #run_hard_case("hard_case_study_parameters.yaml")

    # ------------------- SIMPLE ---------------------
    # G = get_network_pickle("SIMPLE_CASE_network")
    # node_df = get_pickle("SIMPLE_CASE_node_data")
    # edge_df = get_pickle("SIMPLE_CASE_edge_data")

    # ---------------------- HARD --------------------
    #G = get_network_pickle("HARD_CASE_network")
    #node_df = get_pickle("HARD_CASE_node_data")
    #edge_df = get_pickle("HARD_CASE_edge_data")

    # ---------------------- TEMP --------------------
    G = get_network_pickle("TEMP_CASE_network")
    node_df = get_pickle("TEMP_CASE_node_data")
    edge_df = get_pickle("TEMP_CASE_edge_data")

    def plot_intermediate_TVE(G, layer_name, node_name, bins = 100, show = True):

        int_vals = [v2 for k,v in G.nodes(data=True) if v['layer'] == layer_name and v['node_name'] == node_name for v2 in v['intermediate_vals'] ]
        int_vals = pd.DataFrame(int_vals)
        all_diff = int_vals.diff().dropna()

        H_diff = {}
        for row in range(2,len(int_vals.index)+1):
            diff = int_vals[:row].diff().dropna()
            hist, bin_edges = np.histogram(diff, bins = bins, range = (min(all_diff[0]), max(all_diff[0])), density = True)

            bins_diff = bin_edges[1:]
            if row == 2:
                probs_diff = hist*(bins_diff[1]-bins_diff[0])
            else:
                probs_diff = hist*(bins_diff[1]-bins_diff[0])

            d = dit.ScalarDistribution(bins_diff,probs_diff)
            H_diff[row] = dit.other.generalized_cumulative_residual_entropy(d)

        H_raw = {}
        for row in range(1,len(int_vals.index)+1):
            raw_vals = int_vals[:row]

            hist, bin_edges = np.histogram(raw_vals, bins = bins, range = (min(int_vals[0]), max(int_vals[0])), density = True)

            bins_raw = bin_edges[1:]
            if row == 2:
                probs = hist*(bins_raw[1]-bins_raw[0])
            else:
                probs = hist*(bins_raw[1]-bins_raw[0])

            d = dit.ScalarDistribution(bins_raw,probs)
            H_raw[row] = dit.other.generalized_cumulative_residual_entropy(d)


        fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize = (7,7))
        sns.set()
        sns.set_style('white')
        ax1.plot(H_diff.keys(),H_diff.values(),'-b')
        ax1.plot(H_raw.keys(),H_raw.values(),'-r')
        ax1.legend(["H(Diff)","H(Raw Data)"])
        ax1.title.set_text('Target Value Entropy')
        ax1.set_ylabel('TVE')
        ax1.set_xlabel('Iteration')
        sns.despine()

        #ax2 = ax1.twinx()
        ax2.plot(all_diff, '-b')
        ax2.plot(int_vals, '-r')
        ax2.legend(["Diff","Raw Data"])
        ax2.title.set_text('Value Plots')
        ax2.set_ylabel('Value')
        ax2.set_xlabel('Iteration')
        sns.despine()

        #ax3.hist(temp2, bins = 100, range = (min(all_diff[0]), max(all_diff[0])), density = True)
        ax3.bar(bins_diff, probs_diff, width = bins_diff[1]-bins_diff[0], color = 'b', alpha = 0.5)
        ax3.bar(bins_raw, probs, width = bins_raw[1]-bins_raw[0], color = 'r',alpha = 0.5)
        ax3.legend(["Diff","Raw Data"])
        ax3.title.set_text('Histogram of observed values')
        ax3.set_ylabel('Probability')
        ax3.set_xlabel('Value')
        sns.despine()

        fig.suptitle("Layer: {}, Node: {}".format(layer_name, node_name))
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)

        if show == True:
            plt.show()

        #temp.index = range(1,len(temp)+1)
        #print(temp.to_string())

    def test_TVE_sum(G, layer_name, bins = 100, show = True):

        int_vals = [v2 for k,v in G.nodes(data=True) if v['layer'] == layer_name and v['node_name'] == 'x_veh' for v2 in v['intermediate_vals'] ]
        int_vals1 = [v2 for k,v in G.nodes(data=True) if v['layer'] == layer_name and v['node_name'] == 'x_F35' for v2 in v['intermediate_vals'] ]
        int_vals2 = [v2 for k,v in G.nodes(data=True) if v['layer'] == layer_name and v['node_name'] == 'x_V22' for v2 in v['intermediate_vals'] ]
        int_vals3 = [v2 for k,v in G.nodes(data=True) if v['layer'] == layer_name and v['node_name'] == 'x_AV8B' for v2 in v['intermediate_vals'] ]
        int_vals4 = [v2 for k,v in G.nodes(data=True) if v['layer'] == layer_name and v['node_name'] == 'x_SH60' for v2 in v['intermediate_vals'] ]

        W_veh = [v2 for k,v in G.nodes(data=True) if v['layer'] == layer_name and v['node_name'] == 'W_veh' for v2 in v['intermediate_vals'] ]
        W_F35 = [v2 for k,v in G.nodes(data=True) if v['layer'] == layer_name and v['node_name'] == 'W_F35' for v2 in v['intermediate_vals'] ]
        W_V22 = [v2 for k,v in G.nodes(data=True) if v['layer'] == layer_name and v['node_name'] == 'W_V22' for v2 in v['intermediate_vals'] ]
        W_AV8B = [v2 for k,v in G.nodes(data=True) if v['layer'] == layer_name and v['node_name'] == 'W_AV8B' for v2 in v['intermediate_vals'] ]
        W_SH60 = [v2 for k,v in G.nodes(data=True) if v['layer'] == layer_name and v['node_name'] == 'W_SH60' for v2 in v['intermediate_vals'] ]

        int_vals = pd.DataFrame(int_vals)
        int_vals1 = pd.DataFrame(int_vals1)
        int_vals2 = pd.DataFrame(int_vals2)
        int_vals3 = pd.DataFrame(int_vals3)
        int_vals4 = pd.DataFrame(int_vals4)

        int_vals_dict = {
            'x_veh': int_vals,
            'x_F35': int_vals1,
            'x_V22': int_vals2,
            'x_AV8B': int_vals3,
            'x_SH60': int_vals4,
        }

        all_diff_dict = {
            'x_veh': int_vals.diff().dropna(),
            'x_F35': int_vals1.diff().dropna(),
            'x_V22': int_vals2.diff().dropna(),
            'x_AV8B': int_vals3.diff().dropna(),
            'x_SH60': int_vals4.diff().dropna(),
        }

        H_diff_dict = {
            'x_veh': {},
            'x_F35': {},
            'x_V22': {},
            'x_AV8B': {},
            'x_SH60': {}
        }

        H_dict = {
            'x_veh': {},
            'x_F35': {},
            'x_V22': {},
            'x_AV8B': {},
            'x_SH60': {}
        }


        for node_name, int_vals in int_vals_dict.items():
            for row in range(2,len(int_vals.index)+1):
                diff = int_vals[:row].diff().dropna()
                hist, bin_edges = np.histogram(diff, bins = bins, range = (min(all_diff_dict[node_name][0]), max(all_diff_dict[node_name][0])), density = True)

                bins_diff = bin_edges[1:]
                if row == 2:
                    probs_diff = hist*(bins_diff[1]-bins_diff[0])
                else:
                    probs_diff = hist*(bins_diff[1]-bins_diff[0])

                d = dit.ScalarDistribution(bins_diff,probs_diff)
                H_diff_dict[node_name][row] = dit.other.generalized_cumulative_residual_entropy(d)


            for row in range(1,len(int_vals.index)+1):
                raw_vals = int_vals[:row]

                hist, bin_edges = np.histogram(raw_vals, bins = bins, range = (min(int_vals[0]), max(int_vals[0])), density = True)

                bins_raw = bin_edges[1:]
                if row == 2:
                    probs = hist*(bins_raw[1]-bins_raw[0])
                else:
                    probs = hist*(bins_raw[1]-bins_raw[0])

                d = dit.ScalarDistribution(bins_raw,probs)
                H_dict[node_name][row] = dit.other.generalized_cumulative_residual_entropy(d)

        # sum answers
        H_sum = {}
        for t in range(1,220):
            H_sum[t] = (W_F35[t-1]*H_dict['x_F35'][t] + W_V22[t-1]*H_dict['x_V22'][t] + W_AV8B[t-1]*H_dict['x_AV8B'][t] + W_SH60[t-1]*H_dict['x_SH60'][t])/(W_F35[t-1] + W_V22[t-1] + W_AV8B[t-1]+ W_SH60[t-1])

        [print(H_sum[t], H_dict['x_veh'][t], H_sum[t] - H_dict['x_veh'][t] ) for t,v in H_sum.items()]
        #[print(t,v) for t,v in H_dict['x_veh'].items()]


        fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize = (7,7))
        sns.set()
        sns.set_style('white')
        ax1.plot(H_sum.keys(),H_sum.values(),'-b')
        ax1.plot(H_dict['x_veh'].keys(),H_dict['x_veh'].values(),'-r')
        ax1.legend(["H_sum","H(x_veh)"])
        ax1.title.set_text('Target Value Entropy')
        ax1.set_ylabel('TVE')
        ax1.set_xlabel('Iteration')
        sns.despine()
        plt.show()
        #
        # #ax2 = ax1.twinx()
        # ax2.plot(all_diff, '-b')
        # ax2.plot(int_vals, '-r')
        # ax2.legend(["Diff","Raw Data"])
        # ax2.title.set_text('Value Plots')
        # ax2.set_ylabel('Value')
        # ax2.set_xlabel('Iteration')
        # sns.despine()
        #
        # #ax3.hist(temp2, bins = 100, range = (min(all_diff[0]), max(all_diff[0])), density = True)
        # ax3.bar(bins_diff, probs_diff, width = bins_diff[1]-bins_diff[0], color = 'b', alpha = 0.5)
        # ax3.bar(bins_raw, probs, width = bins_raw[1]-bins_raw[0], color = 'r',alpha = 0.5)
        # ax3.legend(["Diff","Raw Data"])
        # ax3.title.set_text('Histogram of observed values')
        # ax3.set_ylabel('Probability')
        # ax3.set_xlabel('Value')
        # sns.despine()
        #
        # fig.suptitle("Layer: {}, Node: {}".format(layer_name, node_name))
        # fig.tight_layout()
        # fig.subplots_adjust(top=0.88)
        #
        # if show == True:
        #     plt.show()


    #node_df = get_pickle("TEMP_CASE_node_data")
    #midf = node_df.set_index(['layer','node_name', 'time'])
    #print(midf.to_string())


    #export_for_gephi(G, "SIMPLE_CASE_PROJ")


    # ------------------------------PLOTTING-----------------------------------

    # plot_intermediate_TVE(G,"OPS","x_veh",bins=100, show = False)
    # plot_intermediate_TVE(G,"OPS","x_F35",bins=100, show = False)
    # plot_intermediate_TVE(G,"OPS","x_AV8B",bins=100, show = False)
    # plot_intermediate_TVE(G,"OPS","x_V22",bins=100, show = False)
    # plot_intermediate_TVE(G,"OPS","x_SH60",bins=100, show = False)
    # plt.show()

    test_TVE_sum(G,"OPS")

    # node_name = 'Trim'
    # layer_name = 'NAVARCH'
    # midf = node_df.set_index(['layer','node_name', 'time'])
    # #print(midf.to_string())
    # temp = midf.loc[layer_name,node_name,:]['val'].dropna()
    # #print(temp)




    #
    #
    # G1 = create_DiGraph_from_MultiDiGraph(G)
    #
    #
    # print("----------------------------EDGES-------------------------")
    # [print(u,v,d) for u,v,d in G1.edges(data=True)]
    #print("----------------------------NODES-------------------------")
    #[print(n,d) for n,d in G1.nodes(data=True)]
    #print()
    #[print(u,v,d) for u,v,d in G.edges(data=True)]


    # ----------------------------- PLOTTING --------------------------------
    # plt.figure()
    # sns.set()
    # sns.set_style('white')
    # #plt.hold(True)
    # for node, node_data in G.nodes(data=True):
    #     if node_data['layer'] == "NAVARCH":
    #         node_name = node_data['node_name']
    #         temp = midf.loc['NAVARCH',node_name,:]['data_status']
    #         plt.plot(temp)
    # #plt.hold(False)
    # plt.grid(False)
    # plt.show()



    #time = [d for n,d in temp.keys()]
    #values = [d for d in temp.values()]
    #temp2 = {'time': temp['data_status_ts'].keys(), 'data_status': temp['data_status_ts'].values(), 'val': temp['val_ts'].values()}
    #print(temp)
    #print()
    #print(temp2)
    #df = pd.DataFrame(temp2)#, index=[self.network.nodes(data=True)[42]['node_name']])
    #df  = df.T
    #print(df)


    #df = get_pickle("SIMPLE_CASE")

    #print(df[])

if __name__ == '__main__':
    main()
