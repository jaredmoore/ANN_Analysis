import argparse

class NN_Node:
    """ Information about a node in the neural network. """
    def __init__(self,nnum,ntype):
        self.node_num   = nnum  # ID of the node.
        self.node_type  = ntype # Type of the node.
        self.node_level = 0     # Level for placement purposes.
        self.impact     = 0     # Flag for whether a node is connected to an output.
        self.ranked     = 0     # Flag for whether a node has been ranked or not.

    # Repr method.
    def __repr__(self):
        return str(self.node_num)

    # String method for the class.
    def __str__(self):
        return str(self.node_num)

    # Copy a node.
    def __copy__(self):
        return NN_Node(self.node_num,self.node_type)

class NN_Link:
    """ Information about a link in the neural network. """
    def __init__(self, lnum, lsrc, ldest, lweight, lrecur, linonum, lmutnum, lenable):
        self.link_num     = lnum        # Link ID
        self.link_src     = lsrc        # Link Source Node
        self.link_dest    = ldest       # Link Destination Node
        self.link_weight  = lweight     # Link Weight
        self.link_rec     = lrecur      # Link Reccurrence
        self.link_inonum  = linonum     # Link Innovation Number
        self.link_mutnum  = lmutnum     # Link Mutation Number
        self.link_enabled = lenable     # Link Enabled State

    # Print out a string.
    def __regr__(self):
        return str(self.link_num) +": "+str(self.link_src)+"->"+str(self.link_dest)

    # Print out a string.
    def __str__(self):
        return str(self.link_num) +": "+str(self.link_src)+"->"+str(self.link_dest)

class NN_Visualizer(object):
    """ Base class for visualizing a neural network. """

    def __init__(self,input_file,output_file):
        """ Initialize the visualizer. """
        self.input_file = input_file
        self.outfile = output_file
        
        # Keep track of the number and type of nodes
        self.nodes  = 0
        self.inputs = 0
        self.hidden = 0
        self.outputs= 0
        self.bias   = 0

        # Keep track of the number of links.
        self.links = 0

        # Keep a list of the nodes in the network.
        self.nn_nodes = {}

        # Keep a list of the connections in the network.
        self.nn_links = {}

        # Rank information for graphviz output
        self.ranks = {}

        # Process the given input file to get the necessary information.
        self.parse_ann_file()

        # Rank the nodes for placement in the graphviz file.
        self.rank_network()

        # Write the graphviz formatted file.
        self.write_file()

    def parse_ann_file(self):
        """ Read in the input file and section into neural network parameters. """
        with open(self.input_file,"r") as f:
            # Keep track of whether you're in a genome or not.
            genome = 0 

            for line in f:
                genome = self.parse_line(line,genome)

    def parse_line(self,line,genome):
        """ Parse a line from an ann input file. """
        parsed = line.split()

        # Line must be greater than 0 length.
        if len(parsed) == 0:
            return 0

        #Parse for genomestart
        if parsed[0] == 'genomestart':
            return 1
        elif parsed[0] == 'genomeend':
            return 0

        # Exit if not in a genome
        if genome == 0:
            return 0

        # Parse a node.
        if parsed[0] == 'node':
            self.nodes += 1
                        
            # Parse the node type
            if parsed[4] == '0':  # Hidden Layer Nodes
                self.hidden += 1
            elif parsed[4] == '1': # Inputs
                self.inputs += 1
            elif parsed[4] == '2': # Outputs
                self.outputs += 1
            elif parsed[4] == '3': # Bias
                self.bias += 1

            # Create a NN_Node object with parsed information.
            self.nn_nodes[int(parsed[1])] = NN_Node(int(parsed[1]),int(parsed[4]))

        # Parse a gene.
        elif parsed[0] == 'gene':
            self.links += 1
        
            # Create a NN_Link object with parsed information: Key is the destination for the link and then the source.

            # Check to see if key exists.
            if int(parsed[3]) not in self.nn_links:
                self.nn_links[int(parsed[3])] = {}

            self.nn_links[int(parsed[3])][int(parsed[2])] = NN_Link(int(parsed[1]),int(parsed[2]),int(parsed[3]),float(parsed[4]),int(parsed[5]),int(parsed[6]),float(parsed[7]),int(parsed[8]))

        return 1

    def rank_network(self):
        """ Rank the nodes in the network to determine what level they should be placed on. """

        # Keep track of the nodes that we have seen.
        prev_placed_nodes = []

        # Search for the input and bias nodes and add them to the first level.
        self.ranks[0] = []
        keys_to_remove = []
        for key, node in self.nn_nodes.iteritems():
            if node.node_type == 1 or node.node_type == 3:
                self.ranks[0].append(key)
                node.ranked = 1
                node.node_level = 0
                prev_placed_nodes.append(key)
                #keys_to_remove.append(key)

        # Keep track of the input nodes.
        input_nodes = list(prev_placed_nodes)

        #for key in keys_to_remove:
        #    del self.nn_nodes[key]

        # Find the maximum number of incoming links in non-output nodes.
        max_inc_links = 0
        for key, node in self.nn_nodes.iteritems():
            if node.node_type == 0:
                tmp = len(list(set(self.nn_links[node.node_num])-set(input_nodes)))
                max_inc_links = tmp if tmp > max_inc_links else max_inc_links

        rank = 1
        placed_at_rank = 0
        # Go through the rest of the nodes and place them at the according ranks.
        while len(prev_placed_nodes) < len(self.nn_nodes):
            self.ranks[rank] = []
            keys_to_remove = []
            for key, node in self.nn_nodes.iteritems():
                # Check to see if the links coming into the node come from previously ranked nodes.
                if node.node_type == 0 and node.ranked == 0 and set(self.nn_links[key]).issubset(set(prev_placed_nodes)):
                    self.ranks[rank].append(key)
                    node.ranked = 1
                    node.node_level = rank
                    prev_placed_nodes.append(key)
                    #keys_to_remove.append(key)
                    placed_at_rank += 1

            #for key in keys_to_remove:
            #    del self.nn_nodes[key]
            #keys_to_remove = []

            # Check to see if we didn't place a node at this rank.
            # Indicates a recursive property, in which case we place them all on one level.
            # TODO: Rank cycles better.
            if placed_at_rank == 0:
                i = 1
                while i <= max_inc_links and placed_at_rank == 0:
                    for key, node in self.nn_nodes.iteritems():
                        if node.node_type == 0 and node.ranked == 0 and len(set(self.nn_links[node.node_num])-set(input_nodes)) <= i:
                            self.ranks[rank].append(key)
                            node.ranked = 1
                            node.node_level = rank
                            prev_placed_nodes.append(key)
                            placed_at_rank += 1
                            #keys_to_remove.append(key)
                    i += 1
                #for key in keys_to_remove:
                #    del self.nn_nodes[key]
                #keys_to_remove = []
            
            # If we still didn't handle anything, we're left with outputs.
            # Handle Outputs
            if placed_at_rank == 0:
                self.ranks[rank] = [] # Add line for outputs.
                for key, node in self.nn_nodes.iteritems():
                    if node.node_type == 2 and node.ranked == 0:    
                        self.ranks[rank].append(key)
                        node.ranked = 1
                        node.node_level = rank
                        prev_placed_nodes.append(key)
                        #keys_to_remove.append(key)

                #for key in keys_to_remove:
                #    del self.nn_nodes[key]
            placed_at_rank = 0
            rank += 1

    # Write a graph to a file.
    def write_file(self):
        dest = open(self.outfile,'w')

        # Write graphviz header.
        dest.write('digraph G {\n')
#        dest.write('\tsize =\"16,16\"\n')
        # dest.write('\tsplines=false;\n') # Force straight line edges.
        dest.write('\toutputorder=edgesfirst;\n') # Draw edges under the nodes.
        dest.write('\tranksep=6.0;\n') # Set the distance between ranks.
        dest.write('\tnodesep=0.2;\n')

        max_rank = len(self.ranks)-1
        for i in xrange(len(self.ranks)):
            if i == 0: # Input
                dest.write('\tsubgraph cluster_'+str(i)+' {\n')
                dest.write('\t    style=invis;\n')
                dest.write('\t    node[shape=box,style=solid,color=blue4];\n')
                dest.write('\t    rank = min;\n')
                dest.write('\t    label = "Layer '+str(i)+' (Input Layer)";\n\t\t')
            elif i == max_rank: # Output
                dest.write('\tsubgraph cluster_'+str(i)+' {\n')
                dest.write('\t    style=invis;\n')
                dest.write('\t    node[shape=circle,style=solid,color=red2];\n')
                dest.write('\t    rank = max;\n')
                dest.write('\t    label = "Layer '+str(i)+' (Output Layer)";\n\t\t')
            else: # Hidden Nodes
                dest.write('\tsubgraph cluster_'+str(i)+' {\n')
                dest.write('\t    style=invis;\n')
                dest.write('\t    node[shape=diamond,style=solid,color=seagreen2];\n')
                dest.write('\t    rank = same;\n')
                dest.write('\t    label = "Layer '+str(i)+' (Hidden Layer)";\n\t\t')
            for node in self.ranks[i]:
                dest.write(str(node)+'; ')
            dest.write('\n\t}\n\n')

        # Have to hack the connections between clusters for vertical ordering.
        # Make an invisible link between links in two clusters to force this.
        for i in xrange(len(self.ranks)-1):
            dest.write('\t'+str(self.ranks[i][0])+' -> '+str(self.ranks[i+1][0]))
            dest.write('[style=invis];\n')

        # Write out each node connection to the file.
        for key, links in self.nn_links.iteritems():
            for key, link in links.iteritems():
                comma = 0 # Keep track of whether we need a prepended comment to style insertions or not.
                # Check to see if the node goes backwards up the ranking.
                # If so, flip src and dest and then reverse the arrow.
                # Allows for better layout of ANN
                if self.nn_nodes[link.link_src].node_level > self.nn_nodes[link.link_dest].node_level:
                    dest.write('\t'+str(link.link_dest)+' -> '+str(link.link_src)+'[dir=back')
                    comma = 1 
                # Check to see if the nodes are on the same level.
                # If so, set constraint property of link to false so nodes stay on same level.
                elif self.nn_nodes[link.link_src].node_level == self.nn_nodes[link.link_dest].node_level:
                    dest.write('\t'+str(link.link_src)+' -> '+str(link.link_dest)+'[constraint=false')
                    comma = 1
                else:
                    dest.write('\t'+str(link.link_src)+' -> '+str(link.link_dest)+'[')
                if self.nn_nodes[link.link_dest].node_type == 2: # Output Node as Destination
                    if comma == 1:
                        dest.write(',')
                    dest.write('color=green') # Set link color to green.
                elif self.nn_nodes[link.link_dest].node_type == 0 and self.nn_nodes[link.link_src].node_type == 1: # Input to Hidden
                    if comma == 1:
                        dest.write(',')
                    dest.write('color=blue') # Set link color to blue.
                dest.write('];\n')

        dest.write('}')
        dest.close()



###########################################################
# Main Execution for the program.

parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str, help="Input file containing the neural network.")
parser.add_argument("output_file", type=str, help="Output file for the graphviz representation.")
args = parser.parse_args()

NN_Visualizer(args.input_file,args.output_file)
