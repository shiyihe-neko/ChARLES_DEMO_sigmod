import numpy as np

UNIQUE_THRESHOLD = 10
UNIQUENESS_RATIO_THRESHOLD = 0.9
RELATIONAL_OPERATORS =          ['==','!=', '>', '>=', '<', '<=']
RELATIONAL_OPERATORS_FLIPPED =  ['!=','==', '<=', '<', '>=', '>']

def is_categorical(values):
    return len(np.unique(values)) <= UNIQUE_THRESHOLD

def maybe_primary_key(values):
    return len(np.unique(values))/float(len(values)) >= UNIQUENESS_RATIO_THRESHOLD

def flip(condition):
    attr, op, value = condition
    return (attr, RELATIONAL_OPERATORS_FLIPPED[RELATIONAL_OPERATORS.index(op)], value)