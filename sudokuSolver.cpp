#include <vector>
#include <iostream>

#define BLANK 0

using namespace std;

class Solver {
public:
//    vector<string> digits = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
//    vector<string> rows = {'A', 'B', 'C',' D', 'E', 'F', 'G', 'H', 'I'};
//    vector<string> cols = digits;
//    vector<string> squares = cross(rows, cols);
    vector<int> serialGridToSolve;
    vector<vector<int>> workingGrid;

    vector<int> solveSudoku(vector<int> serialGrid) {
        if (serialGrid != serialGridToSolve) {
//            printf("Found new puzzle\n");
            serialGridToSolve = serialGrid;
            workingGrid = deserialize(serialGridToSolve);
            solve(workingGrid);
        }
        return serialize(workingGrid);
    }

    bool solve(vector<vector<int>> grid) {
        int row;
        int col;

        if (!findBlank(grid, row, col)) {
            workingGrid = grid;
            return true;
        }

        for (int num = 1; num <= 9; num++) {
            if (isSafe(grid, row, col, num)) {
                grid[row][col] = num;
                if (solve(grid)) {
                    return true;
                } else {
                    grid[row][col] = BLANK;
                }
            }
        }
        return false;


    }

private:
    vector<vector<int>> deserialize(vector<int> serialGrid) {
        vector<vector<int>> grid;
        for (int i = 0; i < 9; i++) {
            vector<int> subsection(serialGrid.begin() + i*9, serialGrid.begin() + (i+1)*9);
            grid.push_back(subsection);
        }
        return grid;
    }

    vector<int> serialize(vector<vector<int>> grid) {
        vector<int> serialGrid;
        for (vector<int> row : grid) {
            for (int num : row) {
                serialGrid.push_back(num);
            }
        }
        return serialGrid;
    }

//    vector<string> cross(vector<string> a, vector<string> b) {
//        vector<string> product;
//        for (string x: a) {
//            for (string y: b) {
//                product.push_back(x+y);
//            }
//        }
//        return product;
//    }

    bool findBlank(vector<vector<int>> grid, int &row, int &col) {
        for (row = 0; row < 9; row++) {
            for (col = 0; col < 9; col++) {
                if (grid[row][col] == BLANK) return true;
            }
        }
        return false;
    }

    bool usedInRow(vector<vector<int>> grid, int row, int num) {
        for (int col = 0; col < 9; col++) {
            if (grid[row][col] == num) return true;
        }
        return false;
    }

    bool usedInCol(vector<vector<int>> grid, int col, int num) {
        for (int row = 0; row < 9; row++) {
            if (grid[row][col] == num) return true;
        }
        return false;
    }

    bool usedInBox(vector<vector<int>> grid, int boxStartRow, int boxStartCol, int num) {
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                if (grid[row+boxStartRow][col+boxStartCol] == num) return true;
            }
        }
        return false;
    }

    bool isSafe(vector<vector<int>> grid, int row, int col, int num) {
        return !usedInRow(grid, row, num) &&
               !usedInCol(grid, col, num) &&
               !usedInBox(grid, row - row%3, col - col%3, num);
    }
};