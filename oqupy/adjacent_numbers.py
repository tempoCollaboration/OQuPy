'''
Test module to find two adjacent numbers, and then find adjacent numbers where
the first one is odd

'''
import numpy as np

test_array = np.array([10,1,0,2,3,4,4,5,4,7,8,9])
# test_array = np.array([1,1,0,2,3,4,4,5,4,7,8])
# subtract adjacent elements
diff = test_array[1:]-test_array[:-1]

#find where the two elements differ by a single one
indices = np.where(diff==1)
indices = indices[0]

values = test_array[indices]
modulo = values%2
# second_one = np.where(modulo==0)
# find where they differ by a single one and the first element is even
indices_where_even = np.where(test_array[indices]%2==0)

indicies_of_array = indices[indices_where_even]

# print(indices)
# print(indices_where_even)

print(indicies_of_array)