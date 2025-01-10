from tracemalloc import start
from globals import *
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
from ray.rllib.utils.annotations import override
from graph import ComputationalGraph
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.data import Batch
from graph import GraphFeatureEmbedding

class GATModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, embedding_dim=64):
        TorchModelV2.__init__(self, obs_space, action_space,
                              num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print('Policy network device:', self.device)

        self.num_outputs = num_outputs

        self.embeddings = GraphFeatureEmbedding(embedding_dim, self.device)

        # GAT Layers
        self.gat1 = GATConv(embedding_dim * 3, 64, heads=4, edge_dim=embedding_dim)
        self.gat2 = GATConv(256, 64, heads=4, edge_dim=embedding_dim)
        self.gat3 = GATConv(256, 128, heads=4,
                            edge_dim=embedding_dim, concat=False)

        self.action_head = nn.Linear(128, num_outputs)
        self.value_head = nn.Linear(128, 1)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Note: In the algo.build phase, the RLLib will pass in a dummy batch to perform some inspection
        # In this case the constructed graph will be invalid but we still need to return the action logits, the value and the state
        # As a workaround, we return the dummy value with the correct shape and device
        if torch.all(input_dict['obs_flat'].eq(0)):
            batch_size = input_dict["obs_flat"].shape[0]
            self._value = torch.zeros((batch_size, 1), device=self.device)
            return torch.zeros((batch_size, self.num_outputs), device=self.device), state

        # Build the graph from the observation
        # Here we must build the nx.graph object, as the node features in the PyG graph require the `last_assignment` attribute
        batch_update_rules = recover_update_rules(input_dict)
        nx_graphs = [ComputationalGraph(batch_update_rules[i])
                     for i in range(len(batch_update_rules))]

        # Check that the converted graph are all valid. for debugging
        valid_flags = [nx_graph.valid for nx_graph in nx_graphs]
        assert all(valid_flags)

        pyg_graphs = [self.embeddings.to_pyg_data(graph.remove_orphan_nodes())
                      for graph in nx_graphs]  # CPU, slow

        # Create a batch pf PyG graphs from the list of Data objects
        batch = Batch.from_data_list(pyg_graphs).to(self.device)

        # Apply GAT layers
        x = self.gat1(batch.x, batch.edge_index, batch.edge_attr)
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()
        x = F.relu(x)
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()
        x = self.gat2(x, batch.edge_index, batch.edge_attr)
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()
        x = F.relu(x)
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()
        x = self.gat3(x, batch.edge_index, batch.edge_attr)
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()

        # Pooling over the nodes to get a graph-level readout
        x = global_mean_pool(x, batch.batch)

        # Compute action logits and value
        action_logits = self.action_head(x)
        self._value = self.value_head(x)

        return action_logits, state

    def value_function(self):
        return self._value.squeeze(1)


def recover_update_rules(input_dict):
    '''
    Recover the update rules from the batched observation processed by RLLib.
    Returns:
        Converted batched update rules. update_rules[i] corresponds to the i-th converted observation in the batch.
    '''
    batched_obs = input_dict["obs"]
    batch_size = batched_obs[0]['func'].shape[0]

    def recover_update_rules_single(statement_index):
        # Phase 1: Reading
        operation_indices = torch.where(
            batched_obs[statement_index]['func'] == 1)[1]
        type_indices_stack = torch.stack(
            [operand['type'] for operand in batched_obs[statement_index]['operands']]).permute(1, 0, 2)
        variable_indices_stack = torch.stack(
            [operand['variable_index'] for operand in batched_obs[statement_index]['operands']]).permute(1, 0, 2)
        constant_indices_stack = torch.stack([operand['constant_index'] for operand in batched_obs[statement_index]['operands']]).permute(1, 0, 2)
        type_indices = torch.where(type_indices_stack == 1)
        variable_indices = torch.where(variable_indices_stack == 1)
        constant_indices = torch.where(constant_indices_stack == 1)
        result_indices = torch.where(
            batched_obs[statement_index]['result_index'] == 1)[1]
        # Phase 2: Building
        update_rules = []
        for batch_index in range(batch_size):
            op_index = operation_indices[batch_index]
            if op_index == NUM_OPERATIONS:  # 'nop' operation
                update_rules.append('nop')
                continue
            operation = operations[op_index]
            operands = []
            for i in range(2):
                type_idx = type_indices[2][batch_index * 2 + i]
                if type_idx == 0:  # variable
                    operands.append(
                        ('v', variable_indices[2][batch_index * 2 + i].item()))
                elif type_idx == 1:  # constant
                    operands.append(
                        ('c', constant_indices[2][batch_index * 2 + i].item()))
                elif type_idx == 2:  # nop placeholder
                    continue
            result = result_indices[batch_index].item()
            update_rules.append([operation, operands, result])
        return update_rules

    # Generate update rules for each statement
    all_update_rules = [recover_update_rules_single(
        statement_index) for statement_index in range(MAX_STATEMENTS)]
    # Transpose and truncate the list of lists to make it batch-first and remove nops
    truncated_update_rules = []
    for batch_index in range(batch_size):
        batch_update_rules = []
        for statement_index in range(MAX_STATEMENTS):
            rule = all_update_rules[statement_index][batch_index]
            if rule == 'nop':
                break
            batch_update_rules.append(rule)
        truncated_update_rules.append(batch_update_rules)
    return truncated_update_rules
