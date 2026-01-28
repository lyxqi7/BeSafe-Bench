import abc
import os
import re
from typing import List, NamedTuple, Optional

import torch

from vla_arena.vla_arena import get_vla_arena_path
from vla_arena.vla_arena.benchmark.vla_arena_suite_task_map import (
    vla_arena_task_map,
)
from vla_arena.vla_arena.envs.bddl_utils import *


BENCHMARK_MAPPING = {}

class Task(NamedTuple):
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str
    level: int
    level_id: int  # Index within the level

# Complete list of all VLA Arena suites
# Organized by category for better readability
vla_arena_suites = [
    'libero_90',
    'libero_static_obstacles',
    'libero_hazard_avoidance',
    'libero_state_preservation',
    'libero_composite',
]

# Map suite names to problem folders
# Organized by category for better readability
suite_to_problem_folder = {
    'libero_90': 'libero_90',
    'libero_static_obstacles': 'libero_static_obstacles',
    'libero_hazard_avoidance': 'libero_hazard_avoidance',
    'libero_state_preservation': 'libero_state_preservation',
    'libero_composite': 'libero_composite',
}

task_maps = {}

# Register all benchmark classes using factory
# Organized by category for better readability
benchmark_names = [
    'libero_90',
    'libero_static_obstacles',
    'libero_hazard_avoidance',
    'libero_state_preservation',
    'libero_composite',
]

# Create and register all benchmark classes
for name in benchmark_names:
    benchmark_class = create_benchmark_class(name)
    register_benchmark(benchmark_class)

# Example usage:
if __name__ == '__main__':
    # Test all benchmarks
    # Organized by category for better readability
    all_benchmarks = [
        'libero_90',
        'libero_static_obstacles',
        'libero_hazard_avoidance',
        'libero_state_preservation',
        'libero_composite',
    ]

    print('Testing all VLA Arena benchmarks:')
    print('=' * 60)

    for benchmark_name in all_benchmarks:
        # Get benchmark class
        benchmark_class = get_benchmark(benchmark_name)

        # Create instance
        benchmark = benchmark_class()

        # Print summary
        print(f'\n{benchmark_name.upper()}')
        print('-' * 40)

        # Get task distribution
        distribution = benchmark.get_task_distribution_by_level()
        total = sum(distribution.values())

        print(f'Total tasks: {total}')
        for level in [0, 1, 2]:
            print(f'  Level {level}: {distribution[level]} tasks')

        # Test accessing a task from each level
        for level in [0, 1, 2]:
            if distribution[level] > 0:
                task = benchmark.get_task_by_level_id(level, 0)
                if task:
                    print(f'  Sample Level {level} task: {task.name}')

    print('\n' + '=' * 60)
    print('All benchmarks loaded successfully!')
