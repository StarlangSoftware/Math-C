//
// Created by Olcay Taner YILDIZ on 9.02.2023.
//

#ifndef MATH_DISCRETEDISTRIBUTION_H
#define MATH_DISCRETEDISTRIBUTION_H


#include <HashMap/LinkedHashMap.h>
#include <stdio.h>

struct discrete_distribution {
    Linked_hash_map_ptr map;
    double sum;
};

typedef struct discrete_distribution Discrete_distribution;
typedef Discrete_distribution *Discrete_distribution_ptr;

Discrete_distribution_ptr create_discrete_distribution();

Discrete_distribution_ptr create_discrete_distribution2(FILE* input_file);

void free_discrete_distribution(Discrete_distribution_ptr discrete_distribution);

void add_item(Discrete_distribution_ptr discrete_distribution, char *item);

void remove_item(Discrete_distribution_ptr discrete_distribution, char *item);

void add_distribution(Discrete_distribution_ptr dst, const Discrete_distribution* added);

void remove_distribution(Discrete_distribution_ptr dst, const Discrete_distribution* removed);

int get_count(const Discrete_distribution* discrete_distribution, const char *item);

char *get_max_item(const Discrete_distribution* discrete_distribution);

char *get_max_item_include_only(const Discrete_distribution* discrete_distribution, const Array_list* include_these_only);

double get_probability(const Discrete_distribution* discrete_distribution, const char *item);

double get_probability_laplace_smoothing(const Discrete_distribution* discrete_distribution, const char *item);

double entropy(const Discrete_distribution* discrete_distribution);

int get_index(const Discrete_distribution* discrete_distribution, const char *item);

Array_list_ptr get_items(const Discrete_distribution* discrete_distribution);

Hash_map_ptr get_probability_distribution(const Discrete_distribution* discrete_distribution);

#endif //MATH_DISCRETEDISTRIBUTION_H
