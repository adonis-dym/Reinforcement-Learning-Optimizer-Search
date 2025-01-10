from update_rule import *
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
import traceback
from copy import deepcopy

class ComputationalGraph(nx.DiGraph):
    def __init__(self, update_rule=None):
        super().__init__()

        # The flag indicates that if the update rule is valid
        # The invalid case includes:
        # 1. Usage of undefined variables as operands
        # 2. The update rule does not contain the 'update' variable (The declaration statement of 'update' variable may be removed by the agent)
        # When it is false, we should no longer use the graph again
        self.valid = True

        # If the update rule is None, build an empty graph
        # We set the empty graph to be valid, as the subgraph function requires that we have to accept the None update rule
        # See the usage of subgraph() function
        # Actually it is indeed valid, although trivial
        if not update_rule:
            return

        # Build the graph based on the update rule
        last_assignment = [0]*4 + [-1]*(MAX_VARIABLES - 4)

        # Initial input: w, g, v1, v2
        self.add_node((0, 0))
        self.add_node((1, 0))
        self.add_node((2, 0))
        self.add_node((3, 0))

        # Iterate over the update rule
        # Statements start from 1
        for i, (operation, operands, result_index) in enumerate(update_rule, start=1):
            # Add the result node with the operation name
            result_node_id = (result_index, i)
            self.add_node(result_node_id, operation=operation['func'].__name__)

            for operand_order, (operand_type, operand_index) in enumerate(operands):
                if operand_type == 'c':
                    # For constant nodes, we identify them with only the index information
                    operand_node_id = operand_index
                    if not self.has_node(operand_node_id):
                        self.add_node(operand_node_id)
                elif operand_type == 'v':
                    # Check if the variable node exists in the graph
                    if last_assignment[operand_index] == -1:
                        # If the variable node doesn't exist, it means that this statement uses an undefined variable
                        self.valid = False
                        return  # Stop processing the update rule. A partially constructed graph is returned. In this case we should not use the graph
                    operand_node_id = (operand_index, last_assignment[operand_index])

                # Add edges from operands to result with order as an attribute
                if self.has_edge(operand_node_id, result_node_id):
                    # If the edge already exists, append the order to the existing list
                    self[operand_node_id][result_node_id]['order'].append(operand_order)
                else:
                    # Otherwise, add the edge with the order as a one-element list
                    self.add_edge(operand_node_id, result_node_id, order=[operand_order])

            # Update the last assignment for the result_index
            last_assignment[result_index] = i

        # Check whether the 'update' variable has been defined
        if last_assignment[4] == -1:
            self.valid = False

    def refine(self):
        # This method modifies the current graph in-place to keep only the nodes that contribute to the output nodes
        # The method returns False if the graph is trivial (not determined by w and g), True otherwise

        # the refine function could only be called on a valid graph
        assert self.valid, "Cannot refine an invalid graph"

        # Define the input and output nodes
        input_nodes = set([(0, 0), (1, 0), (2, 0), (3, 0)])
        update_node = (4, self.last_assignment[4])

        # Perform DFS to find all nodes that can reach the update node
        reachable_nodes = nx.ancestors(self, source=update_node)

        # Find the intersection of input_nodes and reachable_nodes
        used_input_nodes = input_nodes & reachable_nodes

        # If both (0,0) and (1,0) are not in the used input nodes, return False
        if not ({(0, 0), (1, 0)} & used_input_nodes):
            return False

        # Define the output nodes to preserve, initialize with update_node
        output_nodes = {update_node}
        if (2, 0) in used_input_nodes:
            output_nodes.add((2, self.last_assignment[2]))
        if (3, 0) in used_input_nodes:
            output_nodes.add((3, self.last_assignment[3]))

        # Initialize contributive_nodes with output_nodes
        contributive_nodes = output_nodes.copy()

        # Find all nodes that contribute to the output
        for node in output_nodes:
            contributive_nodes = contributive_nodes.union(nx.ancestors(self, node))

        # Create a subgraph with the contributive nodes
        subgraph = self.subgraph(contributive_nodes)

        # Replace the current graph with the new graph
        self.__dict__.clear()
        self.__dict__.update(subgraph.__dict__)

        # Return True to indicate that the graph is not trivial
        return True

    def remove_orphan_nodes(self):
        """
        Remove nodes with no edges from the graph.
        It is a minor version of the refine() method, which rudiementarily cleans the graph data by removing the orphan nodes.
        It does not detect the logic relevance to the output node. In most cases you should use refine().
        """
        orphan_nodes = [node for node, degree in self.degree() if degree == 0]
        self.remove_nodes_from(orphan_nodes)
        return self  # Enable method chaining, such as graph.remove_orphan_nodes().draw('graph.png')

    def draw(self, filename):
        """
        Draw the graph using Matplotlib.
        """

        # Set a large figure size to ensure all nodes are within the image's scope
        plt.figure(figsize=(10, 10))

        pos = nx.planar_layout(self)

        # Calculate the ranges of x and y coordinates
        x_values, y_values = zip(*pos.values())
        x_range = max(x_values) - min(x_values)
        y_range = max(y_values) - min(y_values)

        # Set the offset to be a certain percentage of the ranges
        x_offset = x_range * 0
        y_offset = y_range * 0.033

        # Prepare labels, now including variable names and steps for variable nodes,
        # as well as the constant values for constant nodes. Split long labels into multiple lines.
        node_labels = {
            node: f"{var_names[node[0]]}_({node[1]})\n{self.nodes[node].get('operation', '')}"
            if isinstance(node, tuple) else f"Const: {constants[node]}"
            for node in self.nodes()
        }
        edge_labels = nx.get_edge_attributes(self, 'order')

        # Draw the nodes with larger size
        nx.draw_networkx_nodes(self, pos, node_color='blue', node_size=600, alpha=0.8)

        # Calculate label positions with the new offset
        label_pos = {k: [v[0] + x_offset, v[1] + y_offset] for k, v in pos.items()}

        # Draw the node labels with smaller, black font
        nx.draw_networkx_labels(self, label_pos, labels=node_labels, font_size=8, font_color='black')

        # Draw the edges
        nx.draw_networkx_edges(self, pos, edge_color='blue', alpha=0.8)

        # Draw the edge labels
        nx.draw_networkx_edge_labels(self, pos, edge_labels=edge_labels)

        # Save the figure
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    def __eq__(self, other: 'ComputationalGraph'):
        """
        Compares two computational graphs for equality.

        Args:
        other: Another instance of the ComputationalGraph class to compare with.

        Returns:
        True if the graphs are equal, False otherwise.
        """
        try:
            # 1. Quick checks: check whether the number of nodes and edges are equal
            if self.number_of_nodes() != other.number_of_nodes() or self.number_of_edges() != other.number_of_edges():
                return False

            # 2. Compare output nodes
            output_nodes_self = sorted(self.output_nodes)
            output_nodes_other = sorted(other.output_nodes)

            # quick way to determine if the lists could possibly be equal
            if len(output_nodes_self) != len(output_nodes_other):
                return False

            for node_self, node_other in zip(output_nodes_self, output_nodes_other):
                if node_self[0] != node_other[0] or self.nodes[node_self].get('operation') != other.nodes[node_other].get('operation'):
                    return False

            # 3. Compare input nodes
            # The inputs nodes consist of constant nodes and variable nodes
            # For variable nodes, we create lists, sort them and compare each feature following the sorted order
            variable_nodes_self, variable_nodes_other = [], []
            # For constant nodes, since we only identify them by one int, we can directly compare the sets
            constant_nodes_self, constant_nodes_other = set(), set()

            # Construct the sets and lists for input nodes
            for node in self.input_nodes:
                if isinstance(node, int):
                    constant_nodes_self.add(node)
                else:
                    variable_nodes_self.append(node)

            for node in other.input_nodes:
                if isinstance(node, int):
                    constant_nodes_other.add(node)
                else:
                    variable_nodes_other.append(node)

            # Sort the variable nodes by the variable index
            variable_nodes_self.sort(key=lambda x: x[0])
            variable_nodes_other.sort(key=lambda x: x[0])

            # Compare constant nodes
            if constant_nodes_self != constant_nodes_other:
                return False

            # Compare variable nodes
            if len(variable_nodes_self) != len(variable_nodes_other):
                return False

            for node_self, node_other in zip(variable_nodes_self, variable_nodes_other):
                if node_self[0] != node_other[0]:
                    return False

            # 4. Check on the graph structure: if the two graphs are isomorphic
            if not nx.is_isomorphic(self, other):
                return False

            # 5. Check the node features (function) and the edge features (operand order) by traversing the graph
            # Construct the set of visited nodes
            visited_self = set()
            visited_other = set()

            # The output nodes list are already sorted
            for output_node_self, output_node_other in zip(output_nodes_self, output_nodes_other):
                # DFS
                stack_self = [output_node_self]
                stack_other = [output_node_other]

                # Here we only check the condition on self, as we have checked the isomorphism
                while stack_self:
                    node_self = stack_self.pop()
                    node_other = stack_other.pop()

                    if node_self in visited_self and node_other in visited_other:
                        continue

                    # Checking
                    # Principles:
                    # For middle nodes, only check the operations
                    # For input nodes, check the constant values/variable names

                    # The operations should be the same, if the node have (i.e. middle nodes)
                    if self.nodes[node_self].get('operation') != other.nodes[node_other].get('operation'):
                        return False

                    # If both the nodes are input nodes
                    if self.nodes[node_self].get('operation') == None and other.nodes[node_other].get('operation') == None:
                        # For constant nodes, check the constant values
                        if isinstance(node_self, int) and isinstance(node_other, int):
                            if not node_self == node_other:
                                return False
                        # For variable nodes, check the variable names
                        elif isinstance(node_self, tuple) and isinstance(node_other, tuple):
                            # Assert they are indeed input nodes
                            assert node_self[1] == 0 and node_other[1] == 0, "The nodes without operation could only be the input nodes"
                            # Compare the variable names
                            if node_self[0] != node_other[0]:
                                return False
                        else:  # The two nodes are not of the same type
                            return False

                    # Get the predecessors of the current node, respectively
                    predecessors_self = list(self.predecessors(node_self))
                    predecessors_other = list(other.predecessors(node_other))

                    # Sort predecessors by their operand order of the connection edges
                    predecessors_self.sort(
                        key=lambda x: self.edges[(x, node_self)]['order'])
                    predecessors_other.sort(
                        key=lambda x: other.edges[(x, node_other)]['order'])

                    # Now the predecessors are ordered, we can compare the edge order list
                    # Because we use multiple elements in the 'order' attribute to represent multiple connections
                    # E.g. One node serves as the 0th operand, and another serves as the 1st and 2nd operand
                    # We must distinguish between [[0],[1,2]] and [[0,1],[2]], as it cannot be detected by graph isomorphism
                    edge_order_self = [self.edges[(pred, node_self)]['order']
                                       for pred in predecessors_self]
                    edge_order_other = [other.edges[(pred, node_other)]['order']
                                        for pred in predecessors_other]

                    if edge_order_self != edge_order_other:
                        return False

                    # Add the predecessors to the stack
                    stack_self.extend(predecessors_self)
                    stack_other.extend(predecessors_other)

                    # Add the current node to the visited set
                    visited_self.add(node_self)
                    visited_other.add(node_other)

            return True

        except Exception as e:
            # Error handling. this method often induce errors, here we log the info of two comparing graphs to help diagnose problems
            # You may copy these info into GPT to recover the nx.DiGraph object
            # For RLLib usage, when the info is too long to be shown, you may use `python main.py > output.log 2>&1`
            # This will print the full stack trace including where the exception occurred
            traceback.print_exc()

            # Printing detailed node information including data and type
            print("Error occurred while comparing graphs.")
            print("\nGraph 'self' node details:")
            for node, data in self.nodes(data=True):
                print(f"Node: {node}, Type: {type(node)}, Data: {data}")

            print("\nGraph 'self' edge details:")
            # Adjusted to properly unpack the tuple for edges with data
            for u, v, data in self.edges(data=True):
                print(f"Edge from {u} to {v}, Data: {data}")

            print("\nGraph 'other' node details:")
            for node, data in other.nodes(data=True):
                print(f"Node: {node}, Type: {type(node)}, Data: {data}")

            print("\nGraph 'other' edge details:")
            # Adjusted to properly unpack the tuple for edges with data
            for u, v, data in other.edges(data=True):
                print(f"Edge from {u} to {v}, Data: {data}")

            # Reraise the exception to ensure that it is not swallowed
            raise
    # TODO: Based on the equality rule, write a hash function, which allows fast lookup in the cache
    __hash__ = None

    @property
    def last_assignment(self):
        # Compute the last_assignment list based on the current graph structure
        last_assignment = [0]*4 + [-1]*(MAX_VARIABLES - 4)
        for node in self.nodes():
            if isinstance(node, tuple):
                last_assignment[node[0]] = max(last_assignment[node[0]], node[1])
        return last_assignment

    @property
    def input_nodes(self):
        # Compute the input nodes based on the current graph structure
        return [node for node, degree in self.in_degree() if degree == 0]

    @property
    def output_nodes(self):
        # Compute the output nodes based on the current graph structure
        return [node for node, degree in self.out_degree() if degree == 0]


def graph_to_update_rule(graph):
    # Check whether the graph is valid
    assert graph.valid, "Cannot convert an invalid graph to update rule"

    # Initialize update rule
    update_rule = []

    # Extract variable nodes except input nodes (node[1] > 0)
    variable_nodes = [node for node in graph.nodes() if isinstance(node, tuple) and node[1] > 0]

    # Sort variable nodes by their step
    variable_nodes.sort(key=lambda x: x[1])

    # Iterate over variable nodes and reconstruct each statement
    for node in variable_nodes:
        operation_name = graph.nodes[node]['operation']
        operation = next(
            op for op in operations if op['func'].__name__ == operation_name)

        # Get the edges for the current node
        edges = [(source, target, data)
                 for source, target, data in graph.in_edges(node, data=True)]

        # Initialize a list to hold the operands with their corresponding order
        operand_order_list = []

        for source, _, data in edges:
            # Check the type of the source node to determine the operand type
            if isinstance(source, tuple):
                for order in data['order']:
                    operand_order_list.append((order, ('v', source[0])))
            else:  # source is an int, indicating a constant node
                # Append the constant index directly
                for order in data['order']:
                    operand_order_list.append((order, ('c', source)))  # Use the index directly

        # Sort the operands by their order
        operand_order_list.sort(key=lambda x: x[0])

        # Extract the operands from the sorted list
        operands = [operand for _, operand in operand_order_list]

        # Append the reconstructed statement to the update rule
        update_rule.append([operation, operands, node[0]])

    return update_rule


class GraphFeatureEmbedding(nn.Module):
    def __init__(self, embedding_dim, device):
        super(GraphFeatureEmbedding, self).__init__()
        self.device = device

        # Embedding Layers

        # For variable nodes, the features consist of three parts:
        # 1. Variable name. Each variable node exactly belongs to 1 name class
        self.varname_embedding = nn.Embedding(MAX_VARIABLES, embedding_dim)
        # 2. Assignment statement number. Each variable node exactly belongs to 1 assignment class
        # [2024.12.16 bug fix] The assignment statement number starts from 1, 0 is reserved for the input nodes
        self.assignment_embedding = nn.Embedding(MAX_STATEMENTS + 1, embedding_dim)
        # 3. Operation type. For input nodes, no operation is specified; For other nodes, there are exactly 1 operation
        # The +1 is left for the 'no operation' placeholder
        self.operation_embedding = nn.Embedding(NUM_OPERATIONS + 1, embedding_dim)

        # For constant nodes, we simply map the scalar value to a 1D tensor
        self.const_feature_embedding = nn.Embedding(NUM_CONSTANTS, embedding_dim*3)  # *3 to match the variable node feature dimension

        # For edges, we design the embedding for each operand order
        # It is possible that one edge contains multiple orders, we take the sum or average to pool them. See get_edge_features function
        self.edge_embedding = nn.Embedding(NUM_OPERANDS, embedding_dim)

    def get_node_features(self, node_id, node_data):
        '''
        Given the node information, embed the node features.
        '''
        if isinstance(node_id, tuple):  # Variable node
            varname_index, assignment_statement = node_id
            # Embedding lookups instead of one-hot, create tensors on self.device
            varname_embedding = self.varname_embedding(torch.tensor(varname_index, dtype=torch.long, device=self.device))
            assignment_embedding = self.assignment_embedding(torch.tensor(assignment_statement, dtype=torch.long, device=self.device))
            
            # Check if the node has an operation attribute
            if 'operation' in node_data:
                operation_index = next(i for i, op in enumerate(operations) if op['func'].__name__ == node_data['operation'])
                operation_embedding = self.operation_embedding(torch.tensor(operation_index, dtype=torch.long, device=self.device))
            else:
                # Handle 'no operation' placeholder
                operation_embedding = self.operation_embedding(torch.tensor(NUM_OPERATIONS, dtype=torch.long, device=self.device))

            return torch.cat([varname_embedding, assignment_embedding, operation_embedding])

        elif isinstance(node_id, int):  # Constant node
            return self.const_feature_embedding(torch.tensor(node_id, dtype=torch.long, device=self.device))
        else:
            raise ValueError(f"Invalid node_id type: {type(node_id)}")

    def get_edge_features(self, edge_data):
        '''
        Given the edge information, embed the edge features.
        '''
        # Each edge must contain the order information, but there may be multiple orders
        # Get embeddings for all order indices and calculate the mean
        class_embeddings = [self.edge_embedding(torch.tensor(order, dtype=torch.long, device=self.device)) for order in edge_data['order']]
        return torch.stack(class_embeddings).mean(dim=0)

    def to_pyg_data(self, graph: ComputationalGraph):
        # Map each node to a unique index for edge indexing
        node_to_index = {node: idx for idx, node in enumerate(graph.nodes())}
        # Gather node features using the defined get_node_features method
        node_feature_list = [self.get_node_features(node_id, data) for node_id, data in graph.nodes(data=True)]
        node_features = torch.stack(node_feature_list)

        # Construct the edge index tensor for source and target node indices
        edge_index_list = [[node_to_index[src], node_to_index[dst]] for src, dst in graph.edges()]
        edge_indices = torch.tensor(edge_index_list, device=self.device).t().contiguous()

        # Gather edge features using the defined get_edge_features method
        edge_feature_list = [self.get_edge_features(edge_data) for _, _, edge_data in graph.edges(data=True)]
        edge_features = torch.stack(edge_feature_list)

        # Return a Data object containing all node features, edge index, and edge features
        pyg_data = Data(x=node_features, edge_index=edge_indices, edge_attr=edge_features)

        return pyg_data
