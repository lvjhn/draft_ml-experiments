
from collections import deque
import pandas as pd 

pd.set_option('display.max_columns', None)
DEBUG = False

def flatten(tree, mode="bfs", as_tuple=False): 
    queue = deque()

    def get_label(parent, key, mode, level):
        label = key
        if not as_tuple:
                label = key
                if level > 0: 
                    label = parent + "." + key
        else: 
            label = (key, )
            if level > 0: 
                label = list(parent)
                label.append(key)
                label = tuple(label)
        return label 
   
    def visit_node_dfs(node, parent, level = 0): 
        for key in node:
            label = get_label(parent, key, mode, level)
            yield label
            yield from visit_node_dfs(node[key], label, level + 1)

    def visit_node_bfs(node, parent, level = 0): 
        for key in node:
            label = get_label(parent, key, mode, level)
            yield label
            queue.append((node[key], label, level + 1)) 
        
        while len(queue) > 0:
            item = queue[0]
            node = item[0]
            parent = item[1]
            level = item[2]
            if len(queue) > 0: 
                queue.popleft()
            yield from visit_node_bfs(node, parent, level)
    
    if mode == "dfs":
        yield from visit_node_dfs(tree, "", 0)
    
    elif mode == "bfs":
        yield from visit_node_bfs(tree, "", 0)
    
    else: 
        raise Exception(f"Unknown traversal mode {mode}")
    

def fix_mun_psgc(code):
    return "0".join([code[0:-9], code[-9:]])

def unfix_mun_psgc(code):
    return "".join([code[0:-8], code[-7:]])