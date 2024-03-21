import numpy as np
import random
import re
from itertools import combinations
from queue import PriorityQueue

# Initial TILE_INFO structure
TILE_INFO = {}

def update_tile_info_with_directions():
    # Mapping of direction abbreviations to movement vectors
    direction_mappings = {
        'R': (1, 0),  # Right
        'D': (0, 1),  # Down
        'L': (-1, 0), # Left
        'U': (0, -1), # Up
        'DR': (1, 1),  # Diagonal Right Down
        'DL': (-1, 1), # Diagonal Left Down
        'UR': (1, -1), # Diagonal Right Up
        'UL': (-1, -1), # Diagonal Left Up
    }

    # Loop through each tile type in TILE_INFO and assign directions
    for tile_id, tile_data in TILE_INFO.items():
        if 'directions' not in tile_data:
            tile_data['directions'] = []  # Initialize 'directions' if missing

        allowed_directions = tile_data.get('allowed_directions', [])  # Use 'allowed_directions' if present
        if not allowed_directions:  # If allowed directions are not provided, use default directions
            print(f"Warning: No allowed directions specified for tile '{tile_id}'. Using default directions.")
            allowed_directions = list(direction_mappings.values())  # Use all directions as default
            tile_data['default_directions'] = True  # Add flag indicating default directions
        else:
            allowed_directions = [direction_mappings[abbr] for abbr in allowed_directions if abbr in direction_mappings]

        tile_data['directions'] = allowed_directions

    # Print confirmation message
    print("Tile directions updated successfully.")

update_tile_info_with_directions()

def heuristic(a, b):
    # return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance (original)
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))  # Chebyshev distance (example)

def a_star_search(start, goal, grid, TILE_INFO, silver_points):
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    silver_collected = {start: 0}
    visited = set()  # Track visited positions

    while not frontier.empty():
        current_cost, current = frontier.get()

        if current == goal:
            break

        print(f"Exploring position: {current}")

        for next in get_neighbors(current, grid):
            if next in visited:
                continue  # Skip already visited positions
            visited.add(next)

            new_cost = cost_so_far[current] + TILE_INFO[grid[current[1]][current[0]]]['cost']
            # Prioritize paths with higher silver point scores
            priority = new_cost + heuristic(goal, next) - silver_collected.get(next, 0)
            frontier.put((priority, next))
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put((priority, next))
                came_from[next] = current
                # Aggregate silver points collected
                silver_collected[next] = silver_collected[current] + silver_points.get(next, 0)

    if goal not in came_from:
        print(f"No path found from {start} to {goal}.")
        return None, float('inf'), 0  # Indicate the goal was not reached

    return reconstruct_path(came_from, start, goal), cost_so_far.get(goal, 0), silver_collected.get(goal, 0)

def get_neighbors(position, grid, treat_empty_as_passable=False):
    x, y = position
    neighbors = []
    tile_id = grid[y][x]  # Assuming grid is populated with tile IDs

    if tile_id in TILE_INFO:
        for dx, dy in TILE_INFO[tile_id]['directions']:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid[0]) and 0 <= ny < len(grid):  # Check for grid boundaries
                if treat_empty_as_passable or grid[ny][nx] != '' or ('default_directions' in TILE_INFO[tile_id] and TILE_INFO[tile_id]['default_directions']):
                    neighbors.append((nx, ny))
    return neighbors

def reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]
    while current != start:
        if current not in came_from:
            return None  # No path found (unreachable goal)
        current = came_from[current]
        path.append(current)
    path.reverse()  # Reverse the path to get start -> goal order
    return path

def parse_input(file_path):
    with open(file_path, encoding='utf-8-sig') as file:
        W, H, GN, SM, TL = map(int, file.readline().split())
        grid = [['' for _ in range(W)] for _ in range(H)]  # Initialize grid with empty strings

        golden_points = [tuple(map(int, file.readline().split())) for _ in range(GN)]

        # Fixed parsing of silver points
        silver_points = {}
        for _ in range(SM):
            parts = file.readline().split()
            x, y, score = map(int, parts[:3])  # Assuming the format is X Y SCORE
            silver_points[(x, y)] = score

        # Tile info parsing follows
        tiles_temp = []
        for _ in range(TL):
            tile_id, tile_cost, tile_number = file.readline().split()
            TILE_INFO[tile_id] = {'cost': int(tile_cost), 'directions': [], 'number': int(tile_number)}
            tiles_temp.extend([tile_id] * int(tile_number))  # Duplicate tile_id based on its number

        # Assuming direction data needs to be updated after initializing TILE_INFO
        update_tile_info_with_directions()

        # Place tiles on the grid based on the quantities specified
        tile_index = 0
        for y in range(H):
            for x in range(W):
                grid[y][x] = tiles_temp[tile_index % len(tiles_temp)]  # Use modulo to handle tile repetition
                tile_index += 1  # Increment tile index

    # Debug: Print the grid and tile information
    print("Grid after parsing:")
    for row in grid:
        print(row)

    print("Golden Points:", golden_points)
    print("Silver Points:", silver_points)
    print("Tile Information:", TILE_INFO)

    return grid, golden_points, silver_points

def main(input_file, output_file):
    # Initialize TILE_INFO and then populate it with tile types, costs, and quantities
    global TILE_INFO  # Make sure TILE_INFO is accessible globally
    grid, golden_points, silver_points = parse_input(input_file)

    # Update TILE_INFO with direction data for each tile
    update_tile_info_with_directions()

    if len(golden_points) < 2:
        print("Not enough golden points to form paths.")
        return

    total_cost_overall = 0
    total_silver_score_overall = 0
    all_paths = []

    # Iterating over all pairs of golden points
    for i in range(len(golden_points)):
        for j in range(i + 1, len(golden_points)):
            start, goal = golden_points[i], golden_points[j]

            # Applying A* search to each pair
            print(f"Finding path from {start} to {goal}:")
            path, total_cost, silver_score = a_star_search(start, goal, grid, TILE_INFO, silver_points)
            if path is None:
                print(f"No path found from {start} to {goal}.")
                continue  # Skip to the next pair of golden points

            # Aggregating results
            total_cost_overall += total_cost
            total_silver_score_overall += silver_score
            all_paths.append((start, goal, path, total_cost, silver_score))

    # Outputting aggregated results
    print(f"Overall Cost: {total_cost_overall}")
    print(f"Overall Silver Score: {total_silver_score_overall}")
    write_output(output_file, all_paths, grid)

def write_output(file_path, all_paths, grid):
    with open(file_path, 'w', encoding='utf-8') as file:
        for path_info in all_paths:
            _, _, path, _, _ = path_info
            if path is not None:  # Ensure path is found
                for x, y in path:
                    # Lookup the tile ID from the grid based on coordinates
                    tile_id = grid[y][x] if (0 <= y < len(grid) and 0 <= x < len(grid[0])) else 'Unknown'
                    file.write(f"{tile_id} {x} {y}\n")

if __name__ == "__main__":
    input_file = './inputs/00-trailer.txt'
    output_file = 'output-trailer.txt'
    main(input_file, output_file)

    # Run through all inputs
    input_file = './inputs/01-comedy.txt'
    output_file = 'output-comedy.txt'
    main(input_file, output_file)

    input_file = './inputs/02-sentimental.txt'
    output_file = 'output-sentimental.txt'
    main(input_file, output_file)

    input_file = './inputs/03-adventure.txt'
    output_file = 'output-adventure.txt'
    main(input_file, output_file)

    input_file = './inputs/04-drama.txt'
    output_file = 'output-drama.txt'
    main(input_file, output_file)

    input_file = './inputs/05-horror.txt'
    output_file = 'output-horror.txt'
    main(input_file, output_file)