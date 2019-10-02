import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
import yaml
import xlwings as xw

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

            if G2.out_degree(node[0]) == 0:
                node[1]['type'] = "TARGET"
            elif G2.in_degree(node[0]) == 0:
                node[1]['type'] = "NEGOTIATED"
            else:
                node[1]['type'] = "INTERMEDIATE"

            node[1]['val'] = self.data[node[0]]

            if node[1]['val'] is None:
                node[1]['data_status'] = 0.0
            else:
                node[1]['data_status'] = 1.0

            node[1]['time'] = 0

            node[1]['val_ts'] = {node[1]['time']:node[1]['val']}
            node[1]['data_status_ts'] = {node[1]['time']:node[1]['data_status']}

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
        with open(filename, 'r') as stream:
            data_loaded = yaml.safe_load(stream)
        return data_loaded
    else:
        print('Please input yaml filename.')

def get_data(excel_filename, references_filename):
    data_dict = {}
    excel_doc = Excel(excel_filename, references_filename)

    for layer, references in excel_doc.cell_references.items():
        data_dict[layer] = {}
        for node_name, cell_id in references.items():
            data_dict[layer][node_name] = excel_doc.read_val(layer,cell_id)

    excel_doc.close()
    return data_dict

def get_case_params(filename):
    if filename:
        filename = "../inputs\\{}".format(filename)
        with open(filename, 'r') as stream:
            data_loaded = yaml.safe_load(stream)
        return data_loaded
    else:
        print('Please input yaml filename.')

def main():

    # Import case study file

    #case_params = get_case_params("hard_case_study_parameters.yaml")
    case_params = get_case_params("simple_case_study_parameters.yaml")

    layer_names = case_params["layer_networks"]
    excel_filename = case_params["excel_filename"]
    cell_references_filename = case_params["references_filename"]

    # Import Data from spreadsheets
    yaml_data = get_data(excel_filename, cell_references_filename)

    # layer_names = {
    #     "NAVARCH": "KS_navarch.net",
    #     "OPS": "KS_OPS_enhanced_1_Veh.net",
    #     "DIST":"KS_DIST.net"
    # }

    # Import Gephi Networks into NetworkX format, in Local_KS class structure.
    output = {}
    for layer, filename in layer_names.items():
        output[layer] = Local_KS(name=layer, filename=filename, data = yaml_data[layer])
        #[print(n,d) for n,d in output[layer].network.nodes(data=True)]

    return output


if __name__ == '__main__':
    main()
