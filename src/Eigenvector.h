//
// Created by Olcay Taner YILDIZ on 13.02.2023.
//

#ifndef MATH_EIGENVECTOR_H
#define MATH_EIGENVECTOR_H

#include "Vector.h"

struct eigenvector {
    Vector_ptr vector;
    double eigenvalue;
};

typedef struct eigenvector Eigenvector;
typedef Eigenvector *Eigenvector_ptr;

Eigenvector_ptr create_eigenvector(double eigenvalue, Array_list_ptr vector);

void free_eigenvector(Eigenvector_ptr eigenvector);

int compare_eigenvector(Eigenvector_ptr first, Eigenvector_ptr second);

#endif //MATH_EIGENVECTOR_H
