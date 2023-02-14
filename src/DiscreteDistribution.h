//
// Created by Olcay Taner YILDIZ on 9.02.2023.
//

#ifndef MATH_DISCRETEDISTRIBUTION_H
#define MATH_DISCRETEDISTRIBUTION_H


#include <HashMap/LinkedHashMap.h>

struct discrete_distribution{
    Linked_hash_map_ptr map;
    double sum;
};

typedef struct discrete_distribution Discrete_distribution;
typedef Discrete_distribution* Discrete_distribution_ptr;

Discrete_distribution_ptr create_discrete_distribution();
void free_discrete_distribution(Discrete_distribution_ptr discrete_distribution);
void add_item(Discrete_distribution_ptr discrete_distribution, char* item);
void remove_item(Discrete_distribution_ptr discrete_distribution, char* item);
void add_distribution(Discrete_distribution_ptr dst, Discrete_distribution_ptr added);
void remove_distribution(Discrete_distribution_ptr dst, Discrete_distribution_ptr removed);
int get_count(Discrete_distribution_ptr discrete_distribution, char* item);
char* get_max_item(Discrete_distribution_ptr discrete_distribution);
char* get_max_item_include_only(Discrete_distribution_ptr discrete_distribution, Array_list_ptr include_these_only);
double get_probability(Discrete_distribution_ptr discrete_distribution, char* item);
double get_probability_laplace_smoothing(Discrete_distribution_ptr discrete_distribution, char* item);
double entropy(Discrete_distribution_ptr discrete_distribution);
int get_index(Discrete_distribution_ptr discrete_distribution, char* item);
Array_list_ptr get_items(Discrete_distribution_ptr discrete_distribution);
Hash_map_ptr get_probability_distribution(Discrete_distribution_ptr discrete_distribution);

#endif //MATH_DISCRETEDISTRIBUTION_H
