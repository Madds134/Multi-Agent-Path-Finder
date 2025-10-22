import heapq
from collections import defaultdict

ROWS, COLS = 3,3
def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent):
    ##############################
    # Task 1.2/1.3: Return a table that constains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.
    table = defaultdict(lambda: {'vertex' : set(), 'edge' : set()})
    if constraints is None:
        return table
    for c in constraints:
        if c['agent'] != agent: # Only store the constraints that belong to the specific agent
            continue
        timestep = c['timestep']
        loc = c['loc']  
        # Vertex Constraint, when loc is a single cell list 
        if len(loc) == 1:
            table[timestep]['vertex'].add(loc[0])
        # Edge Constraint, when loc is a two cell list
        elif len(loc) == 2:
            node_from = loc[0]
            node_to = loc[1]
            table[timestep]['edge'].add((node_from, node_to))
    return table


def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    """
    Checks if a move from the current location to the next location violates a vertex or edge constraint

    Args:
        curr_loc (tuple[int, int]): Current location
        next_loc (tuple[int, int]): Next location
        next_time (int): Timestep after the move
        constraint_table (dict): Table of vertex/edge constraints

    Returns:
        bool : True if move violates a constrant, else False
    """
    # Get all the forbidden cells at this timestep
    forbidden_entry = constraint_table.get(next_time)
    if forbidden_entry is None:
        return False
    # Vertex Constraint
    if next_loc in forbidden_entry['vertex']:
        return True
    # Edge constraint
    if (curr_loc, next_loc) in forbidden_entry['edge']:
        return True
    return False

def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']

def get_earliest_goal_timestep(constraint_table):
    """
    Return the maximum time step that appear in the constraint table
    """
    return max(constraint_table.keys()) if len(constraint_table) > 0 else 0

def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints, max_timestep=None):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """
    constraint_table = build_constraint_table(constraints, agent) # Get the constraint table
    open_list = []
    closed_list = dict()
    earliest_goal_timestep = get_earliest_goal_timestep(constraint_table)
    h_value = h_values[start_loc]
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 't' : 0, 'parent': None}

    # If the start is forbidden at t = 0, there is no feasible plan
    if is_constrained(root['loc'], root['loc'], 0, constraint_table):
        return None
    
    push_node(open_list, root)
    closed_list[(root['loc']), (root['t'])] = root # The key is (loc, t)

    while len(open_list) > 0:
        curr = pop_node(open_list)

        #############################
        # Task 1.4: Adjust the goal test condition to handle goal constraints
        if curr['loc'] == goal_loc and curr['t'] >= earliest_goal_timestep \
            and not is_constrained(curr['loc'], curr['loc'], curr['t'], constraint_table):
            return get_path(curr)
        
        # Generate the successors
        next_timestep = curr['t'] + 1
        # Prune the successors with timestep larger than the max
        if max_timestep is not None and next_timestep > max_timestep:
            continue

        for dir in range(4):
            child_loc = move(curr['loc'], dir)
            r,c = child_loc
            # For Task 1,5
            # if r < 0 or r >= ROWS or c < 0 or c >= COLS:
                # continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    't' : curr['t'] + 1,
                    'parent': curr}

            # Prune by the vertex constraint at the next timestep
            if is_constrained(curr['loc'], child['loc'], child['t'], constraint_table):
                continue
            key = (child['loc'], child['t'])

            if key in closed_list:
                existing_node = closed_list[key]
                if compare_nodes(child, existing_node):
                    closed_list[key] = child
                    push_node(open_list, child)
            else:
                closed_list[key] = child
                push_node(open_list, child)
                    # Wait dictionary for new node, stay in place for one step
            wait = {
                'loc' : curr['loc'],                # Keep the same cell
                'g_val' : curr['g_val'] + 1,        # Increment the cost by 1
                'h_val' : h_values[curr['loc']],    # Keep the same heuristic
                't' : curr['t'] + 1,                # Advance the timestamp by 1
                'parent' : curr                     # Maintain t the backpointer
            }
            if not is_constrained(curr['loc'], wait['loc'], wait['t'], constraint_table):
                wait_key = (wait['loc'], wait['t']) # Create tuple of (loc, t) for wait node
                if wait_key in closed_list:         
                    if compare_nodes(wait, closed_list[wait_key]): # If new path is better than old
                        closed_list[wait_key] = wait
                        push_node(open_list, wait) # Since better, reconsider for expansion
                else:
                    closed_list[wait_key] = wait
                    push_node(open_list, wait)
    return None  # Failed to find solution