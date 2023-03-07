from bokeh.models import Rect
from tree_structure import VariableTree, _Node


class TreeTranscriber:

    def __init__(self, tree: VariableTree):
        self.tree = tree

    def render_to_bokeh(self, pos_x=0, pos_y=0, width=20, height=10, min_ratio_leafs: float = 0.5, depth=None, *kwargs):
        if depth is None:
            depth = self.tree.max_depth

        self.__add_node(pos_x=pos_x, pos_y=pos_y, width=width, height=height, min_ratio_leafs=min_ratio_leafs,
                        remaining_depth=depth, *kwargs)

    def __add_node(self, pos_x, pos_y, width, height, min_ratio_leafs: float, remaining_depth, *kwargsf ):
        rect = Rect(x=20, y=-15, width=20, height=10, fill_color='white', line_color='black', line_width=2)

        return [rect]
