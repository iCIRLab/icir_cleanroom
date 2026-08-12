from typing import Any, Optional


GridIndex = tuple[int, int]


class GridGraphPlanner:
    """Build a four-neighbor grid graph and return a continuous DFS walk."""

    # 오른쪽, 위, 왼쪽, 아래 순서로 동일 거리 이웃을 탐색한다.
    DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))

    def __init__(self, grid_data: list[list[Any]]) -> None:
        self.grid_data = grid_data
        self.height = len(grid_data)
        self.width = len(grid_data[0]) if self.height else 0

    def build_neighbors(self) -> int:
        """Connect every sampling cell to all valid four-neighbor cells."""
        edge_count = 0

        for row in self.grid_data:
            for cell in row:
                cell.neighbors.clear()
                if not cell.sampling_enabled:
                    continue

                for delta_row, delta_col in self.DIRECTIONS:
                    neighbor_row = cell.row + delta_row
                    neighbor_col = cell.col + delta_col

                    if not self._is_inside(neighbor_row, neighbor_col):
                        continue

                    neighbor = self.grid_data[neighbor_row][neighbor_col]
                    if neighbor.sampling_enabled:
                        cell.neighbors.append((neighbor_row, neighbor_col))
                        edge_count += 1

        # 양방향 간선이 양쪽 셀에서 한 번씩 집계되므로 2로 나눈다.
        return edge_count // 2

    def nearest_sampling_cell(
        self, x: float, y: float
    ) -> Optional[GridIndex]:
        """Return the sampling cell center nearest to the given map position."""
        candidates = (
            cell
            for row in self.grid_data
            for cell in row
            if cell.sampling_enabled
        )
        nearest = min(
            candidates,
            key=lambda cell: (cell.center_x - x) ** 2
            + (cell.center_y - y) ** 2,
            default=None,
        )
        return None if nearest is None else (nearest.row, nearest.col)

    def create_dfs_route(self, start: GridIndex) -> list[GridIndex]:
        """Return a graph-continuous DFS route including backtracking steps."""
        if not self._is_inside(*start):
            return []
        if not self.grid_data[start[0]][start[1]].sampling_enabled:
            return []

        visited = {start}
        route = [start]
        last_discovery_index = 0
        # stack item: (row, col, index of next neighbor to examine)
        stack = [(start[0], start[1], 0)]

        while stack:
            row, col, next_index = stack[-1]
            neighbors = self.grid_data[row][col].neighbors

            if next_index >= len(neighbors):
                stack.pop()
                if stack:
                    # 실제 이동 경로가 이어지도록 부모 셀 복귀를 기록한다.
                    route.append((stack[-1][0], stack[-1][1]))
                continue

            neighbor = neighbors[next_index]
            stack[-1] = (row, col, next_index + 1)

            if neighbor in visited:
                continue

            visited.add(neighbor)
            route.append(neighbor)
            last_discovery_index = len(route) - 1
            stack.append((neighbor[0], neighbor[1], 0))

        # 마지막 새 셀을 방문한 뒤 시작점까지 복귀하는 꼬리 구간은 제외한다.
        # 탐색 도중 다른 분기로 이동하기 위한 백트래킹은 그대로 유지한다.
        return route[:last_discovery_index + 1]

    def sampling_cell_count(self) -> int:
        return sum(
            1
            for row in self.grid_data
            for cell in row
            if cell.sampling_enabled
        )

    def _is_inside(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width
