import sys
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
    def __init__(self,excel_filename, references_filename):
        self.app = xw.App(visible=False)
        self.wb = xw.Book("../excel\\{}".format(excel_filename))
        self.sheets = self.wb.sheets
        self.dest= "../excel\\{}".format(excel_filename)
        self.cell_references = self.import_cell_references("../excel\\{}".format(references_filename))

    # Write value to cell
    def write_val(self,sheet,cell,value):
        self.sheets[sheet].range(cell).value = value

    # Get cell value
    def read_val(self, sheet, cell):
        val = self.sheets[sheet].range(cell).value
        return val

    # Save excel file
    def save_excel(self) :
        self.wb.save(self.dest)

    def import_cell_references(self, references_filename):
        if references_filename:
            with open(references_filename, 'r') as stream:
                data_loaded = yaml.safe_load(stream)
            return data_loaded
        else:
            print('Please input yaml filename.')

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
            'data_status': status,
            'time': self.time_step,
            'val_ts': {self.time_step: value},
            'data_status_ts': {self.time_step: status}
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

    def init_network(self):
        G = nx.MultiDiGraph()
        return G

    def build_layered_network(self):
        local_networks = [local.network for local in self.local_K.values()]
        all_networks = local_networks + [self.global_K.network] + [self.global_I.network]
        self.network = nx.disjoint_union_all(all_networks)
        return self.network

    def calculate_data_status(self, node_id):
        # Determines the data status of a node based on if it has a value or
        # not.
        if self.network.node[node_id]['val']:
            self.network.node[node_id]['data_status'] = 1.0
        else:
            self.network.node[node_id]['data_status'] = 0.0

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
            self.network.add_edge(old_node_id, new_node_id, layer = "BETWEEN", type = "CREATE", weight = 1.0, time = self.time_step)
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
            self.network.add_edge(old_node_id, new_node_id, layer = "BETWEEN", type = "CREATE", weight = 1.0, time = self.time_step)
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
                self.network.add_edge(global_node_id, local_node_id, layer = "BETWEEN", type = "SELECT", weight = 1.0, time = self.time_step)
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

    def do_local_calculations(self, local_layer_name = "", excel_filename = "local_calculations.xlsx", references_filename = "cell_references.yaml" ):
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
        excel_doc = Excel(excel_filename, references_filename)

        # Write independent values to spreadsheet (inputs)
        for node_id, node_name in ind_nodes.items():
            excel_doc.write_val(sheet = local_layer_name, cell = excel_doc.cell_references[local_layer_name][node_name], value = self.network.node[node_id]['val'])

        # Read dependent values to network (outputs)
        for node_id, node_name in dep_nodes.items():
            self.network.node[node_id]['val'] = excel_doc.read_val(sheet=local_layer_name, cell=excel_doc.cell_references[local_layer_name][node_name])
            self.local_K[local_layer_name].network.node[node_id]['val'] = self.network.node[node_id]['val']
            self.calculate_data_status(node_id)

        ###NEED TO ADD UPDATES TO DATA STATUSES IN BOTH GLOBAL AND LOCAL LAYERS
        # Close and (save?) excel document
        #excel_doc.save_excel()
        excel_doc.close()

    def find_negotiated_nodes(self, target_node_id, local_layer_name):
        out_nodes = []
        ancestors = list(nx.ancestors(self.local_K[local_layer_name].network, target_node_id))

        for node in ancestors:
            if self.local_K[local_layer_name].network.in_degree(node) == 0 and self.network.node[node]['data_status'] == 0:
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

        # Create new graph containing these edges
        B = nx.DiGraph(self.network.edge_subgraph(edge_list))

        # Need to add temporary edges from negotiated_nodes to target_node in IG
        edges_to_add = []
        for e in edge_list:
            if (self.network.node[e[0]]['layer'] == "I_GLOBAL" and self.network.node[e[1]]['layer'] == local_layer_name):
                edges_to_add.append((e[1], IG_target_node_id))

        B.add_edges_from(edges_to_add)

        # Do bipartite projection
        IG_projection = bipartite.projected_graph(B, nodes = {n:d for n,d in B.nodes(data=True) if d['layer'] == "I_GLOBAL"})

        # Add projected edges to network
        for u,v,d in IG_projection.edges(data=True):
            self.network.add_edge(u,v, layer = "I_GLOBAL", type = "PROJECTION", weight = 1.0, time = self.time_step)


    def update_time(self):
        # Increment time_step by 1, and update all node timeseries in all networks (local and integrated)

        for n in self.network.nodes():
            self.network.node[n]['val_ts'][self.time_step] = self.network.node[n]['val']
            self.network.node[n]['data_status_ts'][self.time_step] = self.network.node[n]['data_status']
        self.time_step += 1

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

def get_case_params(filename):
    if filename:
        filename = "../inputs\\{}".format(filename)
        with open(filename, 'r') as stream:
            data_loaded = yaml.safe_load(stream)
        return data_loaded
    else:
        print('Please input yaml filename.')

def GMT_case_study(case_study_filename):

    case_params = get_case_params(case_study_filename)

    case_name = case_params["name"]
    layer_sequence = case_params["layer_sequence"]
    KG_target_nodes = case_params["KG_target_nodes"]
    excel_filename = case_params["excel_filename"]
    cell_references_filename = case_params["references_filename"]

    KIF = Integrated_Framework()
    KIF.build_layered_network()

    for KG_target_node in KG_target_nodes:
        #for layer in layer_sequence:
        KIF.add_KG_target_node(KG_target_node)
        KIF.grow_IG_from_KG(KG_target_node)
        KIF.update_time()
        local_node_id = KIF.select_KL_node_from_IG(KG_target_node, layer_sequence[0])
        KIF.update_time()
        negotiated_nodes = KIF.find_negotiated_nodes(local_node_id, layer_sequence[0])
        for node_id, node_name in negotiated_nodes:
            KIF.grow_IG_from_KL(node_name, layer_sequence[0])
        KIF.update_time()

        ops_node_ids = []
        for node_id, node_name in negotiated_nodes:
            temp_local_node_id = KIF.select_KL_node_from_IG(node_name, layer_sequence[1])
            ops_node_ids.append((temp_local_node_id, node_name))
        KIF.update_time()

        for ops_node_id, ops_node_name in ops_node_ids:
            KIF.send_node_value(ops_node_name, layer_sequence[1], ops_node_name, "I_GLOBAL")
        KIF.update_time()

        for ops_node_id, ops_node_name in ops_node_ids:
            KIF.send_node_value(ops_node_name, "I_GLOBAL", ops_node_name, layer_sequence[0])
        KIF.update_time()

        KIF.do_local_calculations(layer_sequence[0])

        KIF.send_node_value(KG_target_node, layer_sequence[0], KG_target_node, "I_GLOBAL")
        IG_target_node_id = KIF.get_node_id(KG_target_node, layer = "I_GLOBAL")
        KIF.create_IG_edge_projections(IG_target_node_id, local_layer_name = 'NAVARCH')
        KIF.update_time()
        KIF.send_node_value(KG_target_node, "I_GLOBAL", KG_target_node, "K_GLOBAL")
        KIF.update_time()



        df = KIF.create_dataframe()
        print(df)
        df.to_excel("../results\\{}.xlsx".format(case_name))


        KIF.draw_framework()

###############################################################################
def main():

    GMT_case_study("case_study_parameters.yaml")

    # excel_filename = "local_calculations.xlsx"
    # cell_references_filename = "cell_references.yaml"
    #
    # case_params = get_case_params("case_study_parameters.yaml")
    # print(case_params)
    #
    # KIF = Integrated_Framework()
    #
    # KIF.build_layered_network()
    # KIF.add_KG_target_node("GMT")
    # KIF.grow_IG_from_KG("GMT")
    # KIF.grow_IG_from_KL("W_fuel","NAVARCH")
    # KIF.select_KL_node_from_IG("GMT","NAVARCH")
    # KIF.create_update_edge("GMT", "NAVARCH", "GMT", "I_GLOBAL")
    # KIF.create_update_edge("GMT", "NAVARCH", "GMT", "I_GLOBAL")

    #KIF.do_local_calculations('NAVARCH')

    #df = KIF.create_dataframe()
    #print(df[df['source_data_status']==1.0])
    #KIF.draw_framework()



if __name__ == '__main__':
    main()
