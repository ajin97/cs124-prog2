#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
using namespace std;

// returns a pointer to matrix element m[i][j]
static inline int* me(int* m, size_t sz, size_t i, size_t j) {
    return &m[i * sz + j];
}

// naive O(n^3) matrix multiplication algorithm. computes a x b and stores
// the result in c
void matrix_mult_naive(int* a, int* b, size_t sz, int* c) {
    memset(c, 0, sz * sz * sizeof(int));
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
    memset(c, 0, sz * sz * sizeof(int));
    for (size_t i = 0; i < sz; i++) {
        for (size_t k = 0; k < sz; k++) {
            for (size_t j = 0; j < sz; j++) {
                *me(c, sz, i, j) += *me(a, sz, i, k) * *me(b, sz, k, j);
            }
        }
    }
}

// adds specified portion of matrix b to whatever is in specified portion
// of matrix a (i.e. a += b)
void matrix_add(int* a, int row_a, int col_a, size_t sz_a,
                int* b, int row_b, int col_b, size_t sz_b, size_t sz) {
    for (size_t i = 0; i < sz; i++) {
        for (size_t j = 0; j < sz; j++) {
            *me(a, sz_a, row_a + i, col_a + j) +=
                *me(b, sz_b, row_b + i, col_b + j);
        }
    }
}

// subtracts specified portion of matrix b from whatever is in specified
// portion of matrix a (i.e. a -= b)
void matrix_subtract(int* a, int row_a, int col_a, size_t sz_a,
                     int* b, int row_b, int col_b, size_t sz_b, size_t sz) {
    for (size_t i = 0; i < sz; i++) {
        for (size_t j = 0; j < sz; j++) {
            *me(a, sz_a, row_a + i, col_a + j) -=
                *me(b, sz_b, row_b + i, col_b + j);
        }
    }
}

// strassen's matrix multiplication algorithm with crossover to naive algorithm
// when matrices are of size cp or smaller
void matrix_mult_strassen(int* a, int* b, size_t sz, int* c, unsigned cp) {
    if (sz <= cp) {
        // use optimized naive implementation
        matrix_mult_naive_optimized(a, b, sz, c);
    } else {
        // strassen's algorithm
        memset(c, 0, sz * sz * sizeof(int));
        size_t new_sz = sz / 2;

        // allocate three temporary square matrices of dimension new_sz
        int* temp1 = (int*) calloc(new_sz * new_sz, sizeof(int));
        int* temp2 = (int*) calloc(new_sz * new_sz, sizeof(int));
        int* temp3 = (int*) calloc(new_sz * new_sz, sizeof(int));

        // M1
        matrix_add(temp1, 0, 0, new_sz, a, 0, 0, sz, new_sz); // temp1 += A_11
        matrix_add(temp1, 0, 0, new_sz, a, new_sz, new_sz, sz, new_sz); // temp1 += A_22
        matrix_add(temp2, 0, 0, new_sz, b, 0, 0, sz, new_sz); // temp2 += B_11
        matrix_add(temp2, 0, 0, new_sz, b, new_sz, new_sz, sz, new_sz); // temp2 += B_22
        matrix_mult_strassen(temp1, temp2, new_sz, temp3, cp); // temp3 = temp1 * temp2
        matrix_add(c, 0, 0, sz, temp3, 0, 0, new_sz, new_sz); // C_11 += temp3
        matrix_add(c, new_sz, new_sz, sz, temp3, 0, 0, new_sz, new_sz); // C_22 += temp3

        // M2
        memset(temp1, 0, new_sz * new_sz * sizeof(int));
        memset(temp2, 0, new_sz * new_sz * sizeof(int));
        matrix_add(temp1, 0, 0, new_sz, a, new_sz, 0, sz, new_sz); // temp1 += A_21
        matrix_add(temp1, 0, 0, new_sz, a, new_sz, new_sz, sz, new_sz); // temp1 += A_22
        matrix_add(temp2, 0, 0, new_sz, b, 0, 0, sz, new_sz); // temp2 += B_11
        matrix_mult_strassen(temp1, temp2, new_sz, temp3, cp); // temp3 = temp1 * temp2
        matrix_add(c, new_sz, 0, sz, temp3, 0, 0, new_sz, new_sz); // C_21 += temp3
        matrix_subtract(c, new_sz, new_sz, sz, temp3, 0, 0, new_sz, new_sz); // C_22 -= temp3

        // M3
        memset(temp1, 0, new_sz * new_sz * sizeof(int));
        memset(temp2, 0, new_sz * new_sz * sizeof(int));
        matrix_add(temp1, 0, 0, new_sz, a, 0, 0, sz, new_sz); // temp1 += A_11
        matrix_add(temp2, 0, 0, new_sz, b, 0, new_sz, sz, new_sz); // temp2 += B_12
        matrix_subtract(temp2, 0, 0, new_sz, b, new_sz, new_sz, sz, new_sz); // temp2 -= B_22
        matrix_mult_strassen(temp1, temp2, new_sz, temp3, cp); // temp3 = temp1 * temp2
        matrix_add(c, 0, new_sz, sz, temp3, 0, 0, new_sz, new_sz); // C_12 += temp3
        matrix_add(c, new_sz, new_sz, sz, temp3, 0, 0, new_sz, new_sz); // C_22 += temp3

        // M4
        memset(temp1, 0, new_sz * new_sz * sizeof(int));
        memset(temp2, 0, new_sz * new_sz * sizeof(int));
        matrix_add(temp1, 0, 0, new_sz, a, new_sz, new_sz, sz, new_sz); // temp1 += A_22
        matrix_add(temp2, 0, 0, new_sz, b, new_sz, 0, sz, new_sz); // temp2 += B_21
        matrix_subtract(temp2, 0, 0, new_sz, b, 0, 0, sz, new_sz); // temp2 -= B_11
        matrix_mult_strassen(temp1, temp2, new_sz, temp3, cp); // temp3 = temp1 * temp2
        matrix_add(c, 0, 0, sz, temp3, 0, 0, new_sz, new_sz); // C_11 += temp3
        matrix_add(c, new_sz, 0, sz, temp3, 0, 0, new_sz, new_sz); // C_21 += temp3

        // M5
        memset(temp1, 0, new_sz * new_sz * sizeof(int));
        memset(temp2, 0, new_sz * new_sz * sizeof(int));
        matrix_add(temp1, 0, 0, new_sz, a, 0, 0, sz, new_sz); // temp1 += A_11
        matrix_add(temp1, 0, 0, new_sz, a, 0, new_sz, sz, new_sz); // temp1 += A_12
        matrix_add(temp2, 0, 0, new_sz, b, new_sz, new_sz, sz, new_sz); // temp2 += B_22
        matrix_mult_strassen(temp1, temp2, new_sz, temp3, cp); // temp3 = temp1 * temp2
        matrix_subtract(c, 0, 0, sz, temp3, 0, 0, new_sz, new_sz); // C_11 -= temp3
        matrix_add(c, 0, new_sz, sz, temp3, 0, 0, new_sz, new_sz); // C_12 += temp3

        // M6
        memset(temp1, 0, new_sz * new_sz * sizeof(int));
        memset(temp2, 0, new_sz * new_sz * sizeof(int));
        matrix_add(temp1, 0, 0, new_sz, a, new_sz, 0, sz, new_sz); // temp1 += A_21
        matrix_subtract(temp1, 0, 0, new_sz, a, 0, 0, sz, new_sz); // temp1 -= A_11
        matrix_add(temp2, 0, 0, new_sz, b, 0, 0, sz, new_sz); // temp2 += B_11
        matrix_add(temp2, 0, 0, new_sz, b, 0, new_sz, sz, new_sz); // temp2 += B_12
        matrix_mult_strassen(temp1, temp2, new_sz, temp3, cp); // temp3 = temp1 * temp2
        matrix_add(c, new_sz, new_sz, sz, temp3, 0, 0, new_sz, new_sz); // C_22 += temp3

        // M7
        memset(temp1, 0, new_sz * new_sz * sizeof(int));
        memset(temp2, 0, new_sz * new_sz * sizeof(int));
        matrix_add(temp1, 0, 0, new_sz, a, 0, new_sz, sz, new_sz); // temp1 += A_12
        matrix_subtract(temp1, 0, 0, new_sz, a, new_sz, new_sz, sz, new_sz); // temp1 -= A_22
        matrix_add(temp2, 0, 0, new_sz, b, new_sz, 0, sz, new_sz); // temp2 += B_21
        matrix_add(temp2, 0, 0, new_sz, b, new_sz, new_sz, sz, new_sz); // temp2 += B_22
        matrix_mult_strassen(temp1, temp2, new_sz, temp3, cp); // temp3 = temp1 * temp2
        matrix_add(c, 0, 0, sz, temp3, 0, 0, new_sz, new_sz); // C_11 += temp3

        // free temporary matrices
        free(temp1);
        free(temp2);
        free(temp3);
    }
}

// fills square matrices a and b (of dimension sz) with random integer values
// depending on flag. writes matrices to a file if write_file is 1
void fill_rand_matrices(int* a, int* b, size_t sz, size_t padded_sz,
                        int flag, int write_file) {
    for (size_t i = 0; i < sz; i++) {
        for (size_t j = 0; j < sz; j++) {
            if (flag == 0) {
                // random integer 0 or 1
                *me(a, padded_sz, i, j) = rand() % 2;
                *me(b, padded_sz, i, j) = rand() % 2;
            } else if (flag == 1) {
                // random integer 0, 1, or 2
                *me(a, padded_sz, i, j) = rand() % 3;
                *me(b, padded_sz, i, j) = rand() % 3;
            } else if (flag == 2) {
                // random integer 0, 1, or -1
                *me(a, padded_sz, i, j) = (rand() % 3) - 1;
                *me(b, padded_sz, i, j) = (rand() % 3) - 1;
            }
        }
    }

    if (write_file) {
        ofstream fout ("input.txt");
        // write matrix a to file
        for (size_t i = 0; i < sz; i++) {
            for (size_t j = 0; j < sz; j++) {
                fout << *me(a, padded_sz, i, j) << endl;
            }
        }
        // write matrix b to file
        for (size_t i = 0; i < sz; i++) {
            for (size_t j = 0; j < sz; j++) {
                fout << *me(b, padded_sz, i, j) << endl;
            }
        }
    }
}

// reads square matrices a and b (of dimension sz) from an input file
void read_matrices_from_file(char* file, size_t sz, size_t padded_sz,
                             int* a, int* b) {
    ifstream fin (file);
    // read matrix a from file
    for (size_t i = 0; i < sz; i++) {
        for (size_t j = 0; j < sz; j++) {
            fin >> *me(a, padded_sz, i, j);
        }
    }
    // read matrix b from file
    for (size_t i = 0; i < sz; i++) {
        for (size_t j = 0; j < sz; j++) {
            fin >> *me(b, padded_sz, i, j);
        }
    }
}

int padding(int dim, int cp) {
    int i = 0;
    float n = dim * 1.;
    while (n > cp) {
        i++;
        n = n / 2;
    }
    int j = ceil(n);
    for (int k = 0; k < i; k++){
        j = j * 2;
    }
    return j;
}

int main(int argc, char *argv[]) {
    srand(time(0));
    if (argc != 4 && argc != 3) {
        cout << "Usage: ./strassen 0 dimension inputfile" << endl;
        return 0;
    }

    int flag = atoi(argv[1]);
    int dim = atoi(argv[2]);
    char* input_file = NULL;
    if (argc == 4) {
        input_file = argv[3];
    }

    int cp = 64;
    int padded_dim = padding(dim, cp);

    // allocate matrices
    int* a = (int*) calloc(padded_dim * padded_dim, sizeof(int));
    int* b = (int*) calloc(padded_dim * padded_dim, sizeof(int));
    int* c = (int*) calloc(padded_dim * padded_dim, sizeof(int));

    // populate matrices
    if (input_file) {
        read_matrices_from_file(input_file, dim, padded_dim, a, b);
    } else {
        fill_rand_matrices(a, b, dim, padded_dim, flag, 1);
    }

    // multiply matrices and print diagonal of result
    clock_t t = clock();
    matrix_mult_strassen(a, b, padded_dim, c, cp);
    // matrix_mult_naive_optimized(a, b, padded_dim, c);
    t = clock() - t;
    for (int i = 0; i < dim; i++) {
        cout << *me(c, padded_dim, i, i) << endl;
    }
    cout << ((float) t) / CLOCKS_PER_SEC << " seconds" << endl;

    // free matrices
    free(a);
    free(b);
    free(c);

    return 0;
}
