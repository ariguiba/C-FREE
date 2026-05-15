# Source: https://github.com/KemperNiklas/FragNet/blob/view/publication/data/count_substructures.py
from torch_geometric.transforms import BaseTransform
# from utils.fragmentation.fragmentations import FragmentRepresentation

class CountSubstructures(BaseTransform):
    """
    Transformation that adds the counts of a Magnet substructure as target.
    Used for the substructure count experiment.

    Args:
        vocab (list): The vocabulary for magnet, defaults to a pre-computed list of the 100 most common substructures in ZINC-subset .
        substructure_idx (int, optional): The index of the substructure to count. If None, all substructure counts are added.
    """

    def __init__(self, vocab_size=100, substructure_idx=None):
        # self.magnet = Magnet(vocab)
        self.fragment_representation = FragmentRepresentation(vocab_size)
        assert (substructure_idx is None or substructure_idx < vocab_size)
        self.substructure_idx = substructure_idx

    def __call__(self, data):
        # data = self.magnet(data)
        data = self.fragment_representation(data)
        data.y_count = data.fragments.sum(dim=0)
        if self.substructure_idx is not None:
            data.y_count = data.y_count[self.substructure_idx]
        return data