from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np


class Neuron():
    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index

    def draw(self, neuron_radius, layerType):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, facecolor = 'red', fill = True)
        pyplot.gca().add_patch(circle)
        # if (layerType == -1):
        label = pyplot.gca().annotate(str(self.index), xy=(self.x, self.y), fontsize=8, ha="center")


class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, pre_nodes_num, edge_s1, dims, edge_s2 = None, edge_s3 = None):
        self.vertical_distance_between_layers = 30
        self.horizontal_distance_between_neurons = 15
        self.neuron_radius = 3
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.dims = dims
        self.edge_s1 = edge_s1
        self.edge_s2 = edge_s2
        self.edge_s3 = edge_s3

        self.color_list = pyplot.cm.tab10(np.linspace(0, 1, 100))
        self.pre_nodes_num = pre_nodes_num
        
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.e1 = 0
        self.e2 = 0
        self.e3 = 0
        
    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y, iteration + self.pre_nodes_num)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2):

        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)

        if (neuron2.index, neuron1.index) in self.edge_s1:
            self.e1 += 1
            c = 'c'
            line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment), color=c)
            pyplot.gca().add_line(line)
        elif self.edge_s2 != None and (neuron2.index, neuron1.index) in self.edge_s2:
            self.e2 += 1
            c = 'r'
            line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment), color=c)
            pyplot.gca().add_line(line)
        elif self.edge_s3 != None and (neuron2.index, neuron1.index) in self.edge_s3:
            self.e3 += 1
            c = 'black'
            line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment), color=c)
            pyplot.gca().add_line(line)
        

    def draw(self, layerType=0):
        for neuron in self.neurons:
            neuron.draw(self.neuron_radius, layerType)
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron)
        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            pyplot.text(x_text, self.y, 'Hidden Layer 0', fontsize = 10)
        elif layerType == -1:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize = 10)
        else:
            pyplot.text(x_text, self.y, 'Hidden Layer ' + str(layerType), fontsize = 10)
        
        return self.e1, self.e2, self.e3

class NeuralNetwork():
    def __init__(self, widest_layer, edge_s1, dims, neural_list, edge_s2 = None, edge_s3 = None):
        self.number_of_neurons_in_widest_layer = widest_layer
        self.layers = []
        self.layertype = 0
        self.dims = dims
        self.edge_s1 = edge_s1
        self.neural_list = neural_list
        self.edge_s2 = edge_s2
        self.edge_s3 = edge_s3


    def add_layer(self, number_of_neurons, pre_nodes_num):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer, pre_nodes_num, self.edge_s1, self.dims, self.edge_s2, self.edge_s3)
        self.layers.append(layer)

    def draw(self, name, path):
        e1 = 0
        e2 = 0
        e3 = 0
        pyplot.figure(figsize=(20, 20))
        for i in range( len(self.layers) ):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            n1, n2, n3= layer.draw( i )
            e1 += n1
            e2 += n2
            e3 += n3
            
        print(f'cyan edges {e1}, red edges {e2}, black {e3}.')
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title(name, fontsize=12)
        # pyplot.savefig(path + name + ".png")
        pyplot.show()


class DrawNN():
    def __init__(self, dims, edge_set, neural_list, e2=None, e3=None):
        self.dims = dims
        self.edge_s1 = edge_set
        self.neural_list = neural_list
        self.edge_s2 = e2
        self.edge_s3 = e3

    def draw(self, name, path):
        cur_nodes_index = self.dims[0]
        widest_layer = max(self.neural_list)
        network = NeuralNetwork(widest_layer, self.edge_s1, self.dims, self.neural_list, self.edge_s2, self.edge_s3)
        i = 1
        for l in self.neural_list:
            network.add_layer(l, cur_nodes_index)
            cur_nodes_index +=self.dims[i]
            i += 1
        network.draw(name, path)