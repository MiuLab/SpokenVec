import math
import argparse
import re
import copy
import functools
import numbers
import ast
from typing import List, Optional, Sequence, Union

import numpy as np
import scipy.sparse

from tqdm import tqdm


LOG_ZERO = -1e10

class Sentence(object):
    def __init__(self, idx=None, score=None):
        self.idx = idx
        self.score = score

    def __getitem__(self, key):
        raise NotImplementedError("must be implemented by subclasses")

    def sent_len(self) -> int:
        raise NotImplementedError("must be implemented by subclasses")

    def len_unpadded(self) -> int:
        raise NotImplementedError("must be implemented by subclasses")

    def create_padded_sent(self, pad_len: numbers.Integral) -> 'Sentence':
        raise NotImplementedError("must be implemented by subclasses")

    def create_truncated_sent(self, trunc_len: numbers.Integral) -> 'Sentence':
        raise NotImplementedError("must be implemented by subclasses")

    def get_unpadded_sent(self) -> 'Sentence':
        if self.sent_len() == self.len_unpadded():
            return self
        else:
            return self[:self.len_unpadded()]


class ReadableSentence(Sentence):
    def __init__(self, idx: numbers.Integral, score: Optional[numbers.Real] = None) -> None:
        super().__init__(idx=idx, score=score)

    def str_tokens(self, **kwargs) -> List[str]:
        raise NotImplementedError("must be implemented by subclasses")

    def sent_str(self, custom_output_procs=None, **kwargs) -> str:
        out_str = " ".join(self.str_tokens(**kwargs))
        return out_str

    def __repr__(self):
        return f'"{self.sent_str()}"'

    def __str__(self):
        return self.sent_str()


class LatticeNode(object):
    """
    A lattice node, keeping track of neighboring nodes.

    Args:
    nodes_prev: A list indices of direct predecessors
    nodes_next: A list indices of direct successors
    value: Word id assigned to this node.
    fwd_log_prob: Lattice log probability normalized in forward-direction (successors sum to 1)
    marginal_log_prob: Lattice log probability globally normalized
    bwd_log_prob: Lattice log probability normalized in backward-direction (predecessors sum to 1)
    """
    def __init__(self,
                 nodes_prev: Sequence[numbers.Integral],
                 nodes_next: Sequence[numbers.Integral],
                 value: str,
                 fwd_log_prob: Optional[numbers.Real]=None,
                 marginal_log_prob: Optional[numbers.Real]=None,
                 bwd_log_prob: Optional[numbers.Real]=None) -> None:
        self.nodes_prev = nodes_prev
        self.nodes_next = nodes_next
        self.value = value
        self.fwd_log_prob = fwd_log_prob
        self.marginal_log_prob = marginal_log_prob
        self.bwd_log_prob = bwd_log_prob


class Lattice(ReadableSentence):
    """
    A lattice structure.

    The lattice is represented as a list of nodes, each of which keep track of the indices of predecessor and
    successor nodes.

    Args:
    idx: running sentence number (0-based; unique among sentences loaded from the same file, but not across files)
    nodes: list of lattice nodes
    num_padded: denoting that this many words are padded (without adding any physical nodes)
    unpadded_sent: reference to original, unpadded sentence if available
    """

    def __init__(self, idx: Optional[numbers.Integral], nodes: Sequence[LatticeNode],
                 num_padded: numbers.Integral = 0, unpadded_sent: 'Lattice' = None) -> None:
        self.idx = idx
        self.nodes = nodes
        assert len(nodes[0].nodes_prev) == 0
        assert len(nodes[-1].nodes_next) == 0
        for t in range(1, len(nodes) - 1):
            assert len(nodes[t].nodes_prev) > 0
            if len(nodes[t].nodes_next) == 0:
                nodes[t].nodes_next.append(len(nodes)-1)
        nodes[-1].fwd_log_prob = 0.0
        nodes[-1].marginal_log_prob = 0.0
        nodes[-1].bwd_log_prob = 0.0

        self.num_padded = num_padded
        self.unpadded_sent = unpadded_sent

    def sent_len(self) -> int:
        """Return number of nodes in the lattice, including padded words.

        Return:
          Number of nodes in lattice.
        """
        return len(self.nodes) + self.num_padded

    def len_unpadded(self) -> int:
        """Return number of nodes in the lattice, without counting padded words.

        Returns:
          Number of nodes in lattice.
        """
        return len(self.nodes)

    def __getitem__(self, key: numbers.Integral) -> Optional[int]:
        """
        Return the value of a particular lattice node. Padded nodes are virtually appended at the end.

        Args:
          key: Index of lattice node.

        Returns:
          Value of lattice node with given index, or ES if accessing a padded lattice node.
        """
        if self.len_unpadded() <= key < self.sent_len():
            return "</s>"
        node = self.nodes[key]
        if isinstance(node, list):
            # no guarantee that slice is still a consistent graph
            raise ValueError("Slicing not support for lattices.")
        return node.value

    def create_padded_sent(self, pad_len: numbers.Integral) -> 'Lattice':
        """
        Return padded lattice.

        Args:
          pad_len: Number of tokens to pad.

        Returns:
          New padded lattice, or self if pad_len==0.
        """
        if pad_len == 0:
            return self
        copied_nodes = copy.deepcopy(self.nodes)
        return Lattice(idx=self.idx, nodes=copied_nodes, num_padded=pad_len,
                       unpadded_sent=self.unpadded_sent or super().get_unpadded_sent())

    def create_truncated_sent(self, trunc_len: numbers.Integral) -> 'Lattice':
        """
        Return self, as truncation is not supported.

        Args:
          trunc_len: Number of tokens to truncate, must be 0.

        Returns:
          self.
        """
        if trunc_len != 0: raise ValueError("Lattices cannot be truncated.")
        return self

    def get_unpadded_sent(self) -> 'Lattice':
        return self.unpadded_sent or super().get_unpadded_sent()

    def reversed(self) -> 'Lattice':
        """
        Create a lattice with reversed direction.

        The new lattice will have lattice nodes in reversed order and switched successors/predecessors.
        It will have the same number of padded nodes (again at the end of the nodes!).

        Returns:
          Reversed lattice.
        """
        rev_nodes = []
        seq_len = len(self.nodes)
        for node in reversed(self.nodes[:self.len_unpadded()]):
            new_node = LatticeNode(nodes_prev=[seq_len - n - 1 for n in node.nodes_next],
                                   nodes_next=[seq_len - p - 1 for p in node.nodes_prev],
                                   value=node.value,
                                   fwd_log_prob=node.bwd_log_prob,
                                   marginal_log_prob=node.marginal_log_prob,
                                   bwd_log_prob=node.bwd_log_prob)
            rev_nodes.append(new_node)
        return Lattice(idx=self.idx, nodes=rev_nodes, num_padded=self.num_padded)

    def str_tokens(self, **kwargs) -> List[str]:
        """
        Return list of readable string tokens.

        Args:
          **kwargs: ignored

        Returns: list of tokens of linearized lattice.
        """
        return [node.value for node in self.nodes]

    def sent_str(self, custom_output_procs=None, **kwargs) -> str:
        """
        Return a single string containing the readable version of the sentence.

        Args:
          custom_output_procs: ignored
          **kwargs: ignored

        Returns: readable string
        """
        out_str = str([self.str_tokens(**kwargs), [node.nodes_next for node in self.nodes]])
        return out_str

    def plot(self, out_file, show_log_probs=["fwd_log_prob", "marginal_log_prob", "bwd_log_prob"]):
        from graphviz import Digraph
        dot = Digraph(comment='Lattice', format="png")
        for i, node in enumerate(self.nodes):
            node_id = i
            log_prob_strings = [f"{math.exp(getattr(node,field)):.3f}" for field in show_log_probs]
            node_label = f"{node.value} {'|'.join(log_prob_strings)}"
            node.id = node_id
            dot.node(str(node_id), f"{node_id} : {node_label}")
        for node_i, node in enumerate(self.nodes):
            for node_next in node.nodes_next:
                edge_from, edge_to = node_i, node_next
                dot.edge(str(edge_from), str(edge_to), "")
        try:
            dot.render(out_file)
        except RuntimeError as e:
            print(e.what())

    def longest_distances(self) -> List[int]:
        """
        Compute the longest distance to the start node for every node in the lattice.

        For padded nodes, the distance will be set to the highest value among unpadded elements

        Args:
        lattice: A possibly padded lattice. Padded elements have to appear at the end of the node list.

        Returns:
        List of longest distances.
        """
        num_nodes = self.len_unpadded()
        adj_matrix = np.full((num_nodes, num_nodes), -np.inf)
        for node_i in range(self.len_unpadded()):
            node = self.nodes[node_i]
            for next_node in node.nodes_next:
                adj_matrix[node_i, next_node] = -1
        # computing longest paths
        dist_from_start = scipy.sparse.csgraph.dijkstra(csgraph=adj_matrix,
                                                        indices=[0])
        max_pos = int(-np.min(dist_from_start))
        dist_from_start = [int(-d) for d in dist_from_start[0]] + [max_pos] * (self.sent_len()-self.len_unpadded())
        return dist_from_start

    
    def compute_pairwise_log_conditionals(self, direction="fwd", probabilistic_masks=True):
        """
        Compute pairwise log conditionals.

        For row i and column j, the result is log(Pr(j in path | i in path)).

        Runs in O(|V|^3).

        Args:
          lattice: The input lattice.

        Returns:
          A list of numpy arrays of dimensions NxN, where N is the unpadded lattice size.
          The list contains as many entries as there are different masks, e.g. 2 items for fwd- and bwd-directed masks.
        """
        assert self.sent_len() == self.len_unpadded()
        if direction != 'bwd':
            pairwise = []
            for node_i in range(self.len_unpadded()):
                pairwise.append(self.compute_log_conditionals_one(node_i, probabilistic_masks))
            pairwise_fwd = np.asarray(pairwise)

        if direction != 'fwd':
            pairwise = []
            for node_i in range(self.len_unpadded()):
                pairwise.append(list(reversed(self.reversed().compute_log_conditionals_one(self.len_unpadded()-1-node_i, probabilistic_masks))))
            pairwise_bwd = np.asarray(pairwise)

        if direction is None:
            ret = [np.maximum(pairwise_fwd, pairwise_bwd)]
        elif direction=="fwd":
            ret = [pairwise_fwd]
        elif direction=="bwd":
            ret = [pairwise_bwd]
        else:
            if direction!="split": raise ValueError(f"unknown direction argument '{direction}'")
            ret = [pairwise_fwd, pairwise_bwd]

        return ret

    def compute_log_conditionals_one(self, condition_on, probabilistic_masks=True):
        """
        Compute conditional log probabilities for every node being visited after a given node has been visited.

        Note that this is directional: If V1 comes before V2 in a path, then the conditional will be zero.

        Runs in O(|E|) = O(|V|^2) for a lattice with nodes V and edges E.

        Args:
          lattice: The lattice
          condition_on: index of node that must be traversed

        Returns:
          List of log conditionals with same node ordering as for input lattice. Note that padded nodes are ignored and have
          no corresponding entry in the returned list.
        """
        cond_log_probs = [LOG_ZERO] * self.len_unpadded()
        cond_log_probs[condition_on] = 0.0
        for node_i in range(self.len_unpadded()): # nodes are in topological order so we can simply loop in order
            node = self.nodes[node_i]
            for next_node in node.nodes_next:
                if probabilistic_masks:
                    next_log_prob = self.nodes[next_node].fwd_log_prob
                    next_cond_prob = math.exp(cond_log_probs[next_node]) + math.exp(next_log_prob) * math.exp(cond_log_probs[node_i])
                    cond_log_probs[next_node] = math.log(next_cond_prob) if next_cond_prob>0.0 else LOG_ZERO
                else:
                    next_log_prob = 0.0
                    next_cond_prob = max(math.exp(cond_log_probs[next_node]), math.exp(next_log_prob) * math.exp(cond_log_probs[node_i]))
                    cond_log_probs[next_node] = math.log(next_cond_prob) if next_cond_prob > 0.0 else LOG_ZERO
        return cond_log_probs


class LatticeReader:
    def __init__(self, text_input: bool = False, flatten: Union[bool,str] = False):
        self.text_input = text_input
        self.flatten = flatten
        if isinstance(flatten, str): assert flatten=="keep_marginals"

    def read_sent(self, line, idx):
        if self.text_input:
            nodes = [LatticeNode(nodes_prev=[], nodes_next=[1], value="<s>",
                                      fwd_log_prob=0.0, marginal_log_prob=0.0, bwd_log_prob=0.0)]
            for word in line.strip().split():
                nodes.append(
                  LatticeNode(nodes_prev=[len(nodes)-1], nodes_next=[len(nodes)+1], value=word,
                                   fwd_log_prob=0.0, marginal_log_prob=0.0, bwd_log_prob=0.0))
            nodes.append(
                LatticeNode(nodes_prev=[len(nodes) - 1], nodes_next=[], value="</s>",
                                 fwd_log_prob=0.0, marginal_log_prob=0.0, bwd_log_prob=0.0))
        else:
            node_list, arc_list = ast.literal_eval(line)
            nodes = [LatticeNode(nodes_prev=[], nodes_next=[],
                                      value=item[0],
                                      fwd_log_prob=item[1], marginal_log_prob=item[2], bwd_log_prob=item[3])
                     for item in node_list]
            if self.flatten:
                for node_i in range(len(nodes)):
                    if node_i < len(nodes)-1: nodes[node_i].nodes_next.append(node_i+1)
                    if node_i > 0: nodes[node_i].nodes_prev.append(node_i-1)
                    nodes[node_i].fwd_log_prob = nodes[node_i].bwd_log_prob = 0.0
                    if self.flatten != "keep_marginals": nodes[node_i].marginal_log_prob = 0.0
            else:
                for from_index, to_index in arc_list:
                    nodes[from_index].nodes_next.append(to_index)
                    nodes[to_index].nodes_prev.append(from_index)

            assert nodes[0].value == "<s>"
            assert nodes[-1].value == "</s>"

        return Lattice(idx=idx, nodes=nodes)


if __name__ == "__main__":
    line = "[('<s>', 0, 0.0, 0.0), ('i', 0, 0.0, 0.0), ('want', 0, 0.0, 0.0), ('to', 0, 0.0, 0.0), ('know', 0, 0.0, 0.0), ('the', 0, 0.0, 0.0), ('keep', 0, 0.0, 0.0), ('its', 0, 0.0, 0.0), ('way', 0, 0.0, 0.0), ('to', 0, 0.0, 0.0), ('fly', 0, 0.0, 0.0), ('from', 0, 0.0, 0.0), ('denver', 0, 0.0, 0.0), ('two', -2.06376934, -2.06376934, -0.6931471805599453), ('to', -0.135790452, -0.135790452, -0.6931471805599453), ('oakland', 0, 0.0, 0.0), ('</s>', 0, 0.0, 0.0)],[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (12, 14), (13, 15), (14, 15), (15, 16)]"
    lattice = LatticeReader().read_sent(line, 0)
    print(lattice)
    print(lattice.compute_pairwise_log_conditionals("fwd", True))
    print(lattice.compute_pairwise_log_conditionals("fwd", False))
