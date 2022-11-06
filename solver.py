import cProfile
from cmath import inf, sqrt
from collections import namedtuple
from random import randint
import random
import time

# Type for keeping a single puzzle instance
# puzzle: 2d square array for the current puzzle
# zero_row, zero_col: position of the empty square
class PuzzleInstance:
    def __init__(self, puzzle: list[list[int]], zero_row: int, zero_col: int):
        self.puzzle = puzzle
        self.zero_row = zero_row
        self.zero_col = zero_col

# Type for keeping a single node from the search space
# puzzle_instance: keep the current puzzle instance
# cost_to_reach: cost to reach node
# cost_to_goal: estimated cost of reaching goal, starting from current node
# cost_to_goal is total sum of manhattan distances from current position of tile to its solved position
# total_cost: total cost of reaching goal, starting from some initial node, passing through current node
class Node:
    def __init__(self, puzzle_instance:PuzzleInstance, cost_to_reach:int,cost_to_goal:int, total_cost:int):
        self.puzzle_instance = puzzle_instance
        self.cost_to_reach = cost_to_reach
        self.cost_to_goal = cost_to_goal
        self.total_cost = total_cost

# begin from solved puzzle and make valid, random moves to generate random puzzle
def random_puzzle(n: int, swapC: int) -> list[list[int]]:
    def swap_positions(p:PuzzleInstance, i0, j0, i1, j1):
        if(manhattan_distance(i0, j0, i1, j1) == 1 and p[i0][j0] == 0):
            p[i0][j0] = p[i1][j1]
            p[i1][j1] = 0
    puzzle = [[i*n + j + 1 for j in range(0, n)] for i in range(0, n)]
    puzzle[n-1][n-1] = 0
    for i in range(swapC):
        swap_positions(puzzle, randint(0, n-1), randint(0, n-1), randint(0, n-1), randint(0, n-1))
    return puzzle

def is_position_in_puzzle(pos: tuple, p: PuzzleInstance) -> bool:
    return pos[0] >=0 and pos[0] < len(p.puzzle) and pos[1] >= 0 and pos[1] < len(p.puzzle)

def get_position_of_zero(p: list[list[int]]) -> tuple[int, int]:
    return [(i, j) for i in range(0, len(p)) for j in range(0, len(p)) if p[i][j] == 0][0]

def get_ordered_pos(i, l, ordered_puzzle_instance = []):
    if len(ordered_puzzle_instance) == 0:
        i -= 1
        return (int(i/l), i%l)
    else:
        k = ordered_puzzle_instance.index(i)
        return (int(k/l), i%l)

# Return list of valid positions with which the 0 could be swapped
def get_possible_positions(n: Node) -> list[(int, int)]:
    p = n.puzzle_instance
    possiblePositions = [
        (p.zero_row - 1, p.zero_col),
        (p.zero_row + 1, p.zero_col),
        (p.zero_row, p.zero_col - 1),
        (p.zero_row, p.zero_col + 1)]
    newPositions = [pos for pos in possiblePositions if is_position_in_puzzle(pos, p)]
    return newPositions

def manhattan_distance(x1:int,y1:int,x2:int,y2:int) -> int:
    return abs(x1-x2) + abs(y1-y2)

# def get_misplaced_tiles_count(p: PuzzleInstance):
#     return sum([p.puzzle[i][j] != (i*len(p.puzzle) + j + 1)
#     for i in range(0, len(p.puzzle)) for j in range(0, len(p.puzzle[i]))]) - 1

# Swaps 2 tiles from puzzle
# Recalculate distance_to_goal, cost_to_reach, total_cost
# Does not create new node - writes changes to node passed by reference
# Does not check if positions are valid or adjacent
# Returns reference to same node that has been passed as argument
# Can be used for backwards moves - to return to previously visited nodes
def make_move(n:Node, zero_row:int, zero_col:int, swap_row:int, swap_col:int, backwards_move = False, ordered_puzzle_instance = []):
    ordered_pos = get_ordered_pos(n.puzzle_instance.puzzle[swap_row][swap_col], len(n.puzzle_instance.puzzle), ordered_puzzle_instance)
    distance_to_goal = n.cost_to_goal
    cur_dist_to_goal = manhattan_distance(ordered_pos[0], ordered_pos[1], swap_row, swap_col) 
    swap_dist_to_goal = manhattan_distance(ordered_pos[0], ordered_pos[1], zero_row, zero_col)

    # Check if move is going to increase distance to goal
    if  cur_dist_to_goal < swap_dist_to_goal:
        distance_to_goal += 1
    else:
        distance_to_goal -= 1

    # Swap tiles from puzzle 
    temp = n.puzzle_instance.puzzle[zero_row][zero_col]
    n.puzzle_instance.puzzle[zero_row][zero_col] = n.puzzle_instance.puzzle[swap_row][swap_col]
    n.puzzle_instance.puzzle[swap_row][swap_col] = temp

    # Update zero tile position
    n.puzzle_instance.zero_row = swap_row
    n.puzzle_instance.zero_col = swap_col

    # If making a backwards_move then decrease cost_to_reach
    if backwards_move:
        n.cost_to_reach -= 1
    else:
        n.cost_to_reach += 1

    # Update node cost_to_goal and total_cost
    n.cost_to_goal = distance_to_goal
    n.total_cost = n.cost_to_reach + n.cost_to_goal

    return n 

# Calculate sum of manhattan distances for each tile between,
#       cur tile position and cur tile goal position
# Used initially to calculate first node distance to goal
# Not used during search
def h_manhattan_distance_to_goal(p:PuzzleInstance, ordered_puzzle_instance):
    l = len(p.puzzle)
    dist = 0
    for i in range(0, l):
        for j in range(0, l):
            cur_num = p.puzzle[i][j] 
            if cur_num == 0: continue
            # cur_num -= 1
            goal_num_pos = get_ordered_pos(cur_num, l, ordered_puzzle_instance)
            # goal_num_pos = (int(cur_num/l), int(cur_num%l))
            dist += manhattan_distance(i, j, goal_num_pos[0], goal_num_pos[1])
    return dist

def is_goal(n: Node) -> bool:
    return n.cost_to_goal == 0

def deep_copy_node(cur_node):
    n = len(cur_node.puzzle_instance.puzzle)
    return Node(
        PuzzleInstance(
            [[cur_node.puzzle_instance.puzzle[i][j] for j in range(0, n)] for i in range(0, n)],
            cur_node.puzzle_instance.zero_row,
            cur_node.puzzle_instance.zero_col
        ),
        cur_node.cost_to_reach,
        cur_node.cost_to_goal,
        cur_node.total_cost
    )

def bounded_a_star_search_(cur_node: Node, path: list[Node], total_cost_limit: int, prev_zero_row:int, prev_zero_col:int, ordered_puzzle_instance:list[list[int]]):
    # End if cur_node is goal or total_cost_limit has been exceeded
    if is_goal(cur_node):
        return cur_node
    if cur_node.total_cost > total_cost_limit:
        return cur_node

    min_node = deep_copy_node(cur_node)
    zero_pos = (cur_node.puzzle_instance.zero_row, cur_node.puzzle_instance.zero_col)
    positions = get_possible_positions(cur_node)
    for swap_pos in positions:
        # Do not visit the node that has been visited on the previous turn
        if swap_pos[0] == prev_zero_row and swap_pos[1] == prev_zero_col:
            continue
    
        path.append(deep_copy_node(cur_node))

        # Make forward move and update cur_node by reference
        make_move(cur_node, zero_pos[0], zero_pos[1], swap_pos[0], swap_pos[1], False, ordered_puzzle_instance)

        # Initiate search from new cur_node
        # Deep copy node to stop passing by reference of cur_node
        search_res = deep_copy_node(
            bounded_a_star_search_(cur_node, path, total_cost_limit, zero_pos[0], zero_pos[1], ordered_puzzle_instance)
        )

        # End if search_res is goal
        if is_goal(search_res):
            return search_res

        # Find search res with smallest total_cost
        if min_node.total_cost > search_res.total_cost:
            min_node = search_res

        # Make backwards move, reverting any changes made in current loop iteration
        make_move(cur_node, swap_pos[0], swap_pos[1], zero_pos[0], zero_pos[1], True, ordered_puzzle_instance)
        path.pop()   
    return min_node

# Call bounded_a_star_search_ many times with increasing total_cost_limit
def ida_star(p: PuzzleInstance, ordered_puzzle_instance):
    start_node = Node(p, 0, h_manhattan_distance_to_goal(p, ordered_puzzle_instance), h_manhattan_distance_to_goal(p, ordered_puzzle_instance) + 0)

    for l in range(0, 1000):
        path = []
        search_res = bounded_a_star_search_(start_node, path, l, -1, -1, ordered_puzzle_instance)
        if is_goal(search_res):
            return (search_res, path)

def is_solvable(puzzle: list[list[int]]) -> int:
    n = len(puzzle)
    p: list[int] = [puzzle[i][j] for i in range(0, n) for j in range(0, n)]
    l = n*n
    inv_c = sum([int(p[i] != 0 and p[j] != 0 and p[i] > p[j]) for i in range(0, l) for j in range(i+1, l)])
    return inv_c%2 == 0

def main():
    N = int(input())
    W = int(sqrt(float(N + 1)).real)
    I = int(input())
    ordered_puzzle_instance = []
    if I != -1:
        ordered_puzzle_instance = [i for i in range(1, N+1)]
        ordered_puzzle_instance.insert(I, 0)
    p = [
        [int(input()) for j in range(0, W)] for i in range(0, W)
    ]
    if not is_solvable(p):
        print("NOT SOLVABLE")
    else:
        zero_pos = get_position_of_zero(p)
        puzzle_instance = PuzzleInstance(p, zero_pos[0], zero_pos[1])
        start = time.time()
        result = ida_star(puzzle_instance, ordered_puzzle_instance)
        end = time.time()
        t = end - start
        print("ALGORITHM RUN TIME", t)
        print("COST TO REACH GOAL", result[0].cost_to_reach)
        result[1].append(result[0])
        for i in range(0, len(result[1]) - 1):

            node = result[1][i]
            next_node = result[1][i+1]
            delta_row = next_node.puzzle_instance.zero_row - node.puzzle_instance.zero_row
            delta_col = next_node.puzzle_instance.zero_col - node.puzzle_instance.zero_col
            if delta_col == 1:
                print("left")
            elif delta_col == -1:
                print("right")
            elif delta_row == 1:
                print("up")
            elif delta_row == -1:
                print("down")
            
            # for k in range(0, W):
            #     for j in range(0, W):
                    
            #         print(next_node.puzzle_instance.puzzle[k][j], end=" ")
            #     print("\n", end="")
            # print("\n")

if __name__ == "__main__":
    main()