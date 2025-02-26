import random

import networkx as nx

from DecesionTreeTools import Tree


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def drawTree(tree: Tree, title: str = f"Decision Tree", **kwargs):
    import networkx as nx
    import matplotlib.pyplot as plt
    import os
    G = nx.Graph()

    currId = 0
    nodeIds = {}
    nodes = []
    nodeLabels = {}
    for parent, child in tree.getNodesAsTuples():
        if parent is None or child is None:
            continue
        if parent in nodeIds:
            parentId = nodeIds[parent]
        else:
            nodeIds[parent] = currId
            nodeLabels[currId] = str(parent)
            parentId = currId
            currId += 1
        if child in nodeIds:
            childId = nodeIds[child]
        else:
            nodeIds[child] = currId
            nodeLabels[currId] = str(child)
            childId = currId
            currId += 1
        nodes.append((parentId, childId))

    # plt.clf()
    fig = plt.figure(figsize=(14, 11))
    G.add_edges_from(nodes)
    pos = hierarchy_pos(G, nodeIds[tree.root])
    plt.title(title)
    nx.draw(G, pos=pos, labels=nodeLabels, with_labels=True, node_shape='s',
            node_color='w', font_size=7)
    if kwargs.get("save", True):
        os.makedirs("Figures", exist_ok=True)
        plt.savefig(f"Figures/{title.replace(' ', '_')}.png", dpi=500)
    plt.show(block=True)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (2, 7), (3, 8), (3, 9), (4, 10),
                      (5, 11), (5, 12), (6, 13)])
    pos = hierarchy_pos(G, 1, width=100.)
    nx.draw(G, pos=pos, with_labels=True, node_shape='s', node_size=5_000)
    # plt.savefig('hierarchy.png', dpi=300)
    plt.show()
