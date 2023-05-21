//
// Created by Olcay Taner YILDIZ on 13.02.2023.
//

#include <stdlib.h>
#include "Eigenvector.h"

Eigenvector_ptr create_eigenvector(double eigenvalue, Array_list_ptr vector) {
    Eigenvector_ptr result = malloc(sizeof(Eigenvector));
    result->eigenvalue = eigenvalue;
    result->vector = create_vector(vector);
    return result;
}

void free_eigenvector(Eigenvector_ptr eigenvector) {
    free_vector(eigenvector->vector);
    free(eigenvector);
}

int compare_eigenvector(const Eigenvector* first, const Eigenvector* second) {
    if (first->eigenvalue < second->eigenvalue) {
        return -1;
    } else {
        if (first->eigenvalue > second->eigenvalue) {
            return 1;
        } else {
            return 0;
        }
    }
}
