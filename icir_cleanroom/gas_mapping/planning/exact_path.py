"""Generic deterministic exact path algorithms shared by LRS and HRS."""

import math


def _euclidean(first, second):
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _farthest_cell_index(cells, start_xy, distance_fn):
    return max(
        range(len(cells)),
        key=lambda index: (
            distance_fn(
                tuple(start_xy), (cells[index].x, cells[index].y)),
            -cells[index].row, -cells[index].col),
    )


def solve_held_karp_open_path(cells, start_xy, distance_fn=None):
    """Return the exact open path ending at the farthest selected cell."""
    if not cells:
        return None
    distance_fn = distance_fn or _euclidean
    endpoint = _farthest_cell_index(cells, start_xy, distance_fn)
    if len(cells) == 1:
        length = distance_fn(
            tuple(start_xy), (cells[0].x, cells[0].y))
        return [cells[0]], length

    interior = [index for index in range(len(cells)) if index != endpoint]
    bit_for = {node: 1 << bit for bit, node in enumerate(interior)}

    def distance(first, second):
        first_xy = start_xy if first == -1 else (
            cells[first].x, cells[first].y)
        second_xy = (cells[second].x, cells[second].y)
        return distance_fn(first_xy, second_xy)

    dynamic = {}
    for node in interior:
        dynamic[(bit_for[node], node)] = (distance(-1, node), -1)

    full_mask = (1 << len(interior)) - 1
    for mask in range(1, full_mask + 1):
        for last in interior:
            last_bit = bit_for[last]
            if not mask & last_bit or mask == last_bit:
                continue
            previous_mask = mask ^ last_bit
            best = None
            for previous in interior:
                state = dynamic.get((previous_mask, previous))
                if state is None:
                    continue
                candidate = state[0] + distance(previous, last)
                key = (candidate, cells[previous].row, cells[previous].col)
                if best is None or key < best[0]:
                    best = (key, candidate, previous)
            if best is not None:
                dynamic[(mask, last)] = (best[1], best[2])

    finish = None
    for last in interior:
        state = dynamic[(full_mask, last)]
        candidate = state[0] + distance(last, endpoint)
        key = (candidate, cells[last].row, cells[last].col)
        if finish is None or key < finish[0]:
            finish = (key, candidate, last)

    reverse_order = []
    mask = full_mask
    last = finish[2]
    while last != -1:
        reverse_order.append(last)
        previous = dynamic[(mask, last)][1]
        mask ^= bit_for[last]
        last = previous
    ordered_indices = list(reversed(reverse_order)) + [endpoint]
    return [cells[index] for index in ordered_indices], finish[1]


def solve_held_karp_free_end_path(cells, start_xy, distance_fn=None):
    """Return the exact shortest path from ``start_xy`` with a free end."""
    distance_fn = distance_fn or _euclidean
    if not cells:
        return None
    if len(cells) == 1:
        length = distance_fn(
            tuple(start_xy), (cells[0].x, cells[0].y))
        return [cells[0]], length

    count = len(cells)
    full_mask = (1 << count) - 1
    dynamic = {}
    for node, cell in enumerate(cells):
        dynamic[(1 << node, node)] = (
            distance_fn(tuple(start_xy), (cell.x, cell.y)), -1)

    for mask in range(1, full_mask + 1):
        for last in range(count):
            last_bit = 1 << last
            if not mask & last_bit or mask == last_bit:
                continue
            previous_mask = mask ^ last_bit
            best = None
            for previous in range(count):
                state = dynamic.get((previous_mask, previous))
                if state is None:
                    continue
                candidate = state[0] + distance_fn(
                    (cells[previous].x, cells[previous].y),
                    (cells[last].x, cells[last].y))
                key = (candidate, cells[previous].row, cells[previous].col)
                if best is None or key < best[0]:
                    best = (key, candidate, previous)
            if best is not None:
                dynamic[(mask, last)] = (best[1], best[2])

    finish = min(
        range(count),
        key=lambda node: (
            dynamic[(full_mask, node)][0],
            cells[node].row, cells[node].col))
    length = float(dynamic[(full_mask, finish)][0])
    reverse_order = []
    mask = full_mask
    last = finish
    while last != -1:
        reverse_order.append(last)
        previous = dynamic[(mask, last)][1]
        mask ^= 1 << last
        last = previous
    ordered = [cells[index] for index in reversed(reverse_order)]
    return ordered, length


__all__ = ['solve_held_karp_free_end_path', 'solve_held_karp_open_path']
