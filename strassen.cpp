#include <iostream>
#include <fstream>

using namespace std;

// returns a pointer to matrix element m[i][j]
static inline int* me(int* m, size_t sz, size_t i, size_t j) {
    return &m[i * sz + j];
}

// naive O(n^3) matrix multiplication algorithm. computes a x b and stores
// the result in c
void matrix_mult_naive(int* a, int* b, size_t sz, int* c) {
    for (size_t i = 0; i < sz; i++) {
        for (size_t j = 0; j < sz; j++) {
            for (size_t k = 0; k < sz; k++) {
                *me(c, sz, i, j) += *me(a, sz, i, k) * *me(b, sz, k, j);
            }
        }
    }
}

// naive O(n^3) matrix multiplication algorithm with optimization for cache
// alignment. computes a x b and stores the result in c
void matrix_mult_naive_optimized(int* a, int* b, size_t sz, int* c) {
    for (size_t i = 0; i < sz; i++) {
        for (size_t k = 0; k < sz; k++) {
            for (size_t j = 0; j < sz; j++) {
                *me(c, sz, i, j) += *me(a, sz, i, k) * *me(b, sz, k, j);
            }
        }
    }
}

// strassen's matrix multiplication algorithm with crossover to naive algorithm
// when matrices are of size cp or smaller
void matrix_mult_strassen(int* a, int* b, size_t sz, int* c, int cp) {

}

// fills square matrices a and b (of dimension sz) with random integer values
// depending on flag. writes matrices to a file if write_file is 1
void fill_rand_matrices(int* a, int* b, size_t sz, int flag, int write_file) {
    for (size_t i = 0; i < sz; i++) {
        for (size_t j = 0; j < sz; j++) {
            if (flag == 0) {
                // random integer 0 or 1
                *me(a, sz, i, j) = rand() % 2;
                *me(b, sz, i, j) = rand() % 2;
            } else if (flag == 1) {
                // random integer 0, 1, or 2
                *me(a, sz, i, j) = rand() % 3;
                *me(b, sz, i, j) = rand() % 3;
            } else if (flag == 2) {
                // random integer 0, 1, or -1
                *me(a, sz, i, j) = (rand() % 3) - 1;
                *me(b, sz, i, j) = (rand() % 3) - 1;
            }
        }
    }

    if (write_file) {
        ofstream fout ("input.txt");
        // write matrix a to file
        for (size_t i = 0; i < sz; i++) {
            for (size_t j = 0; j < sz; j++) {
                fout << *me(a, sz, i, j) << endl;
            }
        }
        // write matrix b to file
        for (size_t i = 0; i < sz; i++) {
            for (size_t j = 0; j < sz; j++) {
                fout << *me(b, sz, i, j) << endl;
            }
        }
    }
}

// reads square matrices a and b (of dimension sz) from an input file
void read_matrices_from_file(char* file, size_t sz, int* a, int* b) {
    ifstream fin (file);
    // read matrix a from file
    for (size_t i = 0; i < sz; i++) {
        for (size_t j = 0; j < sz; j++) {
            fin >> *me(a, sz, i, j);
        }
    }
    // read matrix b from file
    for (size_t i = 0; i < sz; i++) {
        for (size_t j = 0; j < sz; j++) {
            fin >> *me(b, sz, i, j);
        }
    }
}

int main(int argc, char *argv[]) {
    srand(time(0));
    if (argc != 4 && argc != 3) {
        cout << "Usage: ./strassen 0 dimension inputfile" << endl;
        return 0;
    }

    int flag = stoi(argv[1]);
    int dim = stoi(argv[2]);
    char* input_file = NULL;
    if (argc == 4) {
        input_file = argv[3];
    }

    // allocate matrices
    int* a = (int*) calloc(dim * dim, sizeof(int));
    int* b = (int*) calloc(dim * dim, sizeof(int));
    int* c = (int*) calloc(dim * dim, sizeof(int));

    // populate matrices
    if (input_file) {
        read_matrices_from_file(input_file, dim, a, b);
    } else {
        fill_rand_matrices(a, b, dim, flag, 1);
    }

    // multiply matrices and print diagonal of result
    matrix_mult_naive(a, b, dim, c);
    for (int i = 0; i < dim; i++) {
        cout << *me(c, dim, i, i) << endl;
    }

    return 0;
}
