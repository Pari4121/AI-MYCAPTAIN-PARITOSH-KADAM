# Define two sets
E = {0, 2, 4, 6, 8}
N = {1, 2, 3, 4, 5}

# Union
union = E.union(N)
print("Union of E and N is", union)

# Intersection
intersection = E.intersection(N)
print("Intersection of E and N is", intersection)

# Difference
difference = E.difference(N)
print("Difference of E and N is", difference)

# Symmetric Difference
symmetric_difference = E.symmetric_difference(N)
print("Symmetric difference of E and N is", symmetric_difference)
