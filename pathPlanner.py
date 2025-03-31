# AERO60492 – Autonomous Mobile Robots
# Coursework 2 – Path Planning
# Euan Baldwin - 10818421
# --------------------------------------------------------------------------------
# A* PATH PLANNING ALGORITHM
#
# This module implements A* pathfinding for a 2D grid map. A* is an informed search
# algorithm that aims to find the least-cost path from a start node to a goal node
# while also using a heuristic to guide exploration. In this context:
#   - Each grid cell is a node.
#   - Moving between two adjacent (non-obstacle) cells has a cost of 1.
#   - A priority queue (open_list) orders cells by f = g + h, where:
#       g = cost to move from the start to the current cell
#       h = heuristic; an admissible estimate of distance from the current cell
#           to the goal (Euclidean distance).
#
# The algorithm terminates when we either reach the goal cell or run out of nodes
# to explore. If the goal cannot be reached, no valid path exists, and an empty list
# is returned.
# --------------------------------------------------------------------------------

import heapq  # Provides an implementation of the heap queue algorithm to sort the queue

def do_a_star(grid, start, end, display_message):
    """
    Performs A* path planning on a 2D grid, returning a list of (col, row) coordinates.

    The function restricts movement to the four cardinal directions: up, down, left, right.
    The cost of moving between adjacent cells is 1.

    Args:
        grid (list[list[int]]): 2D array where 1=open cell, 0=obstacle.
                                Accessed by grid[col][row].
        start (tuple[int, int]): (col, row) start cell.
        end (tuple[int, int]): (col, row) goal cell.
        display_message (function): A callable for printing debug messages into the GUI.

    Returns:
        list[tuple[int, int]]: A path of (col, row) coordinates from start to end, inclusive.
                               Returns an empty list if no path is found or if invalid input.
    """
    # --------------------------------------------------------------------------------
    # STEP 1: VALIDITY CHECKS
    # Ensures we have a non-empty grid, valid start and end coordinates, and that
    # the start/end are not positioned on obstacles. If any condition fails, we
    # cannot run the pathfinding.
    # --------------------------------------------------------------------------------

    COLS = len(grid)     # Number of columns in the top-level list
    ROWS = len(grid[0])  # Number of rows in each sub-list

    # Decompose the start and end tuples for convenience
    start_col, start_row = start
    end_col, end_row = end

    if not grid or not grid[0]:
        display_message("Grid is empty or malformed.")
        return []
    
    # Check if start or end are missing (None, invalid)
    if start is None or end is None:
        display_message("Start or End position is not defined.")
        return []
    
    # Check if start or end are within grid boundaries
    if not (0 <= start_col < COLS and 0 <= start_row < ROWS):
        display_message("Start position is out of grid bounds.")
        return []
    if not (0 <= end_col < COLS and 0 <= end_row < ROWS):
        display_message("End position is out of grid bounds.")
        return []

    # Check if start or end cells are blocked by obstacles
    if grid[start_col][start_row] == 0:
        display_message("Start position is on an obstacle.")
        return []
    if grid[end_col][end_row] == 0:
        display_message("End position is on an obstacle.")
        return []
    
    # If start == end, just return that single cell
    if start == end:
        display_message("Start and End are the same cell.")
        return [start]

    # --------------------------------------------------------------------------------
    # STEP 2: ALGORITHM INITIALISATION
    #
    # Set up:
    #   - g_cost to track how far each cell is from the start (default = 'infinity').
    #   - visited array to mark cells as they are explored, so we don't reprocess them.
    #   - parent array to record each cell's predecessor, which later enables us
    #     to reconstruct the path by backtracking from goal to start.
    #   - open_list (a priority queue) where we push cells with their f cost,
    #     ensuring that the most promising node is always explored next.
    # --------------------------------------------------------------------------------

    INF = float('inf')  # Symbolic representation of a very large cost

    # g_cost: Cost from start to each cell
    g_cost = [[INF]*ROWS for _ in range(COLS)]
    g_cost[start_col][start_row] = 0

    # visited: Tracks cells whose neighbours have been processed
    visited = [[False]*ROWS for _ in range(COLS)]

    # parent: To reconstruct the path. parent[c][r] gives the predecessor of (c, r).
    parent = [[None]*ROWS for _ in range(COLS)]

    # Priority queue (min-heap) storing (f_cost, col, row)
    open_list = []

    # Initial node has heuristic = distance from start to end
    f_start = euclidean_distance(start, end)
    heapq.heappush(open_list, (f_start, start_col, start_row))

    # Movement offsets for Up, Down, Left, Right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # --------------------------------------------------------------------------------
    # STEP 3: A* SEARCH LOOP
    #
    # Repeatedly extract the node with the smallest f cost from open_list. If that
    # node is the goal, reconstruct our path. If not, examine its neighbours
    # (the four adjacent cells), computing each neighbour’s potential new g cost.
    #
    # If that potential g cost is smaller than the previously recorded g cost,
    # update it and set the parent pointer. Then, compute a new f cost and add
    # the neighbour to the priority queue for further exploration.
    #
    # This process continues until the open_list is empty (no path) or we reach the goal.
    # --------------------------------------------------------------------------------

    while open_list:
        # Pop the cell with the smallest f_cost
        f_val, current_col, current_row = heapq.heappop(open_list)

        # If we've already visited, skip
        if visited[current_col][current_row]:
            continue
        visited[current_col][current_row] = True

        # If goal is reached, reconstruct and return path
        if (current_col, current_row) == (end_col, end_row):
            display_message("Goal reached. Reconstructing path.")
            return reconstruct_path(parent, start, end)

        # Explore neighbouring cells (up, down, left, right)
        for offset_col, offset_row in directions:
            neighbour_col = current_col + offset_col
            neighbour_row = current_row + offset_row

            # Check bounds and confirm the cell isn't blocked
            if 0 <= neighbour_col < COLS and 0 <= neighbour_row < ROWS and grid[neighbour_col][neighbour_row] == 1:
                new_g = g_cost[current_col][current_row] + 1 # Calculate potential new cost

                # If we found a cheaper way to reach (neighbour_col, neighbour_row), update and re-queue
                if new_g < g_cost[neighbour_col][neighbour_row]:
                    g_cost[neighbour_col][neighbour_row] = new_g
                    parent[neighbour_col][neighbour_row] = (current_col, current_row)
                    f_cost = new_g + euclidean_distance((neighbour_col, neighbour_row), end) # f_cost = g_cost + heuristic
                    heapq.heappush(open_list, (f_cost, neighbour_col, neighbour_row))

    # If we exhaust open_list without reaching goal, no path exists
    display_message("No path found.")
    return []


def euclidean_distance(node, goal):
    """
    Calculates the Euclidean (straight-line) distance between two grid cells.
    This is the heuristic function for A*, providing an admissible estimate of
    how far the node is from the goal in actual spatial terms.
    """
    dx = node[0] - goal[0]
    dy = node[1] - goal[1]
    return (dx*dx + dy*dy) ** 0.5


def reconstruct_path(parent, start, end):
    """
    Reconstructs the path from 'start' to 'end' using the parent matrix.
    This function iterates backwards from the end cell until it reaches
    the start cell. It returns the path as a list of (col, row) coordinates,
    starting at 'start' and concluding at 'end'.
    """
    path = []
    current = end

    # Follow the chain of parents from the goal back to the start
    while current is not None and current != start:
        path.append(current)
        current_col, current_row = current
        current = parent[current_col][current_row]

    # If we reached the start successfully, include it
    if current is not None:
        path.append(current)

    # Reverse the path so it is ordered from start to end
    path.reverse()
    return path
