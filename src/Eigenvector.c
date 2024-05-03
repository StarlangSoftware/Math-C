//
// Created by Olcay Taner YILDIZ on 13.02.2023.
//

#include "Eigenvector.h"
#include "Memory/Memory.h"

/**
 * A constructor of Eigenvector which takes a double eigenValue and an array list values as inputs.
 * It calls its super class Vector with values array list and initializes eigenValue variable with its
 * eigenValue input.
 *
 * @param eigenValue double input.
 * @param vector     Array list input.
 */
Eigenvector_ptr create_eigenvector(double eigenvalue, Array_list_ptr vector) {
    Eigenvector_ptr result = malloc_(sizeof(Eigenvector), "create_eigenvector");
    result->eigenvalue = eigenvalue;
    result->vector = create_vector(vector);
    return result;
}

void free_eigenvector(Eigenvector_ptr eigenvector) {
    free_vector(eigenvector->vector);
    free_(eigenvector);
}

/**
 * The method which takes two eigenvectors as input. If the eigenvalue of the first vector is less than the eigenvalue
 * of the second vector it returns -1, if the eigenvalue of the first vector is larger than the eigenvalue
 * of the second vector it returns 1, 0 otherwise.
 *
 * @param first First eigenvector.
 * @param second Second eigenvector.
 * @return 1 if first is less than the second, 1 if first is larger than the second, and 0 otherwise.
 */
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
