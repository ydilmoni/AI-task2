from MDFValueIteration import MDFValueIteration

w = 7
h = 7
L = [(1, 1, -4), (1, 5, -6), (5, 1, 1), (5, 5, 4)]
p = 0.8
r = -0.5


def printTable(table):
    column_width = max(len(str(val)) if val is not None else 1 for row in table for val in row) + 1
    partition_line = '+' + '+'.join(['-' * column_width for _ in range(len(table[0]))]) + '+'
    print(partition_line)
    for row in table:
        row_str = '|'.join(f' {val if val is not None else ".":<{column_width - 1}}' for val in row)
        print(f'|{row_str}|')
        print(partition_line)


def main():
    myGrid = MDFValueIteration(h, w, p, L, r)
    printTable(myGrid.v)
    myGrid.valueIteration()
    printTable(myGrid.v)
    print("\n\n\nPolicy Table: \n")
    myGrid.policy = myGrid.getPolicyFromV()
    printTable(myGrid.policy)


if __name__ == "__main__":
    main()
