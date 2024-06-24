ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # Down, Left, Up, Right
ACTION_DIC = {(-1, 0): "UP", (1, 0): "DOWN", (0, -1): "LEFT", (0, 1): "RIGHT"}
MAX_ERROR = 10 ** (-3)


def printTable(table):
    column_width = max(len(str(val)) if val is not None else 1 for row in table for val in row) + 1
    partition_line = '+' + '+'.join(['-' * column_width for _ in range(len(table[0]))]) + '+'
    print(partition_line)
    for row in table:
        row_str = '|'.join(f' {val if val is not None else ".":<{column_width - 1}}' for val in row)
        print(f'|{row_str}|')
        print(partition_line)


class MDFValueIteration:
    def __init__(self, height: int, width: int, probability: float, score_data: list[tuple[int, int, int]], cost_per_action: float):
        self.width = width
        self.height = height
        self.score_loc = {(self.height - t[1] - 1, t[0]): t[2] for t in score_data}
        self.probability = probability
        self.un_p = (1 - self.probability) / 2
        self.reward = cost_per_action
        self.v = self.generateTable()
        self.next_v = self.generateTable()
        self.starting_table = self.generateTable()
        self.discount_factor = 0.5
        self.epsilon = 0.000001
        self.delta = float('inf')
        self.policy = [["UP" for _ in range(width)] for _ in range(height)]

    def valueIteration(self):
        self.next_v = self.generateTable()
        while True:
            current_v = [row[:] for row in self.next_v]
            self.delta = 0
            for row in range(self.height):
                for col in range(self.width):
                    if (row, col) in self.score_loc:
                        self.next_v[row][col] = self.score_loc[(row, col)]
                        continue  # Don't update walls and sink states'

                    self.next_v[row][col] = max([self.calculateU(row, col, action) for action in range(4)])
                    self.delta = max(self.delta, abs(self.next_v[row][col] - self.v[row][col]))

            self.v = current_v
            if self.delta < self.epsilon * (1 - self.discount_factor) / self.discount_factor:
                break

            # print("Iteration utilities:")
            # printTable(self.next_v)
            # time.sleep(0.2)

        return self.v

    def generateTable(self):
        width = self.width
        height = self.height
        score_loc = self.score_loc
        # Initialize an empty 2D list (matrix)
        value = [[self.reward for _ in range(width)] for _ in range(height)]

        # Update specific locations with scores
        for key1, key2 in score_loc:
            value[key1][key2] = score_loc[(key1, key2)]  # Update the score locations utility value to

        return value

    def getU(self, row, col, action):
        x, y = ACTIONS[action]
        new_row, new_col = row + x, col + y
        if new_row < 0 or new_row >= self.height or new_col < 0 or new_col >= self.width:
            return self.v[row][col]  # Return the current states utility value
        else:
            return self.v[new_row][new_col]

    def calculateU(self, row, col, action):
        u = 0  # Start with the action cost
        u += self.un_p * self.discount_factor * self.getU(row, col, (action - 1) % 4)  # Transition to left
        u += self.probability * self.discount_factor * self.getU(row, col, action)  # Transition to intended direction
        u += self.un_p * self.discount_factor * self.getU(row, col, (action + 1) % 4)  # Transition to right
        return u

    def getPolicyFromV(self):
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) in self.score_loc:
                    self.policy[i][j] = self.score_loc[(i, j)]  # Use the fixed reward for terminal states
                    continue

                maxAction, max_u = None, float('-inf')
                for action in range(4):
                    u = self.calculateU(i, j, action)
                    if u > max_u:
                        max_u = u
                        maxAction = action
                self.policy[i][j] = ACTION_DIC[ACTIONS[maxAction]]  # Assign the corresponding action
        return self.policy
