# REMOVE_AFTER_READING: avoid defining things using String as it may lead to typo and introduce bugs.

# List the possinle models here 
class Approach:
    LINEAR_TREE = 1
    K_MEANS = 2
    SUBSPACE_CLUSTERING = 3
    EXPLAIN_DA_V = 4

    def get_name(m):
        if m == 1:
            return 'linear_tree'
        if m == 2:
            return 'k_means_clustering'
        if m == 3:
            return 'subspace_clustering'
        if m == 4:
            return 'explain_da_v'

# List the possible metrics here
class Metric:
    MEAN_SQUARED_ERROR = 1
    PARTITION_WISE_MATCH = 2

    def get_name(m):
        if m == 1:
            return 'mean_squared_error'
        if m == 2:
            return 'partition_wise_match'