from copy import deepcopy

class SodukuSolver():

    def __init__(self):
        self.board = []
        self.solution = []

    def read_board(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                self.board.append(list(line.split()))
        self.to_int()

    def to_int(self):
        for row in range(len(self.board)):
            for col in range(len(self.board)):
                self.board[row][col] = int(self.board[row][col])
        self.solution = deepcopy(self.board)
    
    def print_board(self):
        for lines in self.board:
            for num in lines:
                print(num, end=' ')
            print()

    def print_solution(self):
        for lines in self.solution:
            for num in lines:
                print(num, end=' ')
            print()

    def valid_col(self, col, num):
        if sum(row[col] == num for row in self.solution) != 1:
            return False
        return True

    def valid_row(self, row, num):
        if self.solution[row].count(num) != 1:
            return False
        return True

    def valid_box(self, start_row, start_col, num):
        start_row = start_row//3*3
        start_col = start_col//3*3
        if sum(row[start_col:start_col+3].count(num) for row in self.solution[start_row:start_row+3]) != 1:
            return False
        return True
    
    def solve(self, cur_row=0, cur_col=0):
        # Base cases
        if cur_col == 9:
            cur_row, cur_col = cur_row+1, 0
        if cur_row == 9:
            return self.solution
        print(cur_row, cur_col)
        # Solve
        if self.board[cur_row][cur_col] == 0:
            for num in range(1, 10):
                self.solution[cur_row][cur_col] = num
                self.print_solution()
                print('-----------------------')
                if self.valid_row(cur_row, num) and self.valid_col(cur_col, num) and self.valid_box(cur_row, cur_col, num):
                    self.solution = self.solve(cur_row, cur_col + 1)
                    # if not self.solution:
                    #     self.solution
                    #     continue
                else:
                    continue
        else:
            self.solution = self.solve(cur_row, cur_col + 1)

if __name__ == '__main__':
    solver = SodukuSolver()
    solver.read_board('input.txt')
    solver.print_board()
    print("Solution")
    solver.solve()
    # solver.print_solution()