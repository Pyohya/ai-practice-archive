import math
from collections import Counter

# 1. Sample dataset: [x_coordinate, y_coordinate, class_label]
dataset = [
    [2, 3, 'A'],
    [4, 7, 'A'],
    [5, 4, 'A'],
    [7, 2, 'B'],
    [8, 5, 'B'],
    [9, 6, 'B']
]

# 2. Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    """Calculates the Euclidean distance between two points (ignoring class labels)."""
    distance = 0
    # Assumes the last element is the class label, so iterate up to the second to last element
    for i in range(len(point1) - 1):
        distance += (point1[i] - point2[i])**2
    return math.sqrt(distance)

# 3. Function to find k nearest neighbors
def get_neighbors(train_data, test_instance, k):
    """Finds the k nearest neighbors of a test instance from the training data."""
    distances = []
    for train_point in train_data:
        dist = euclidean_distance(test_instance, train_point)
        distances.append((train_point, dist))
    distances.sort(key=lambda tup: tup[1]) # Sort by distance
    
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

# 4. Function to predict the class of a new data point
def predict_classification(train_data, test_instance, k):
    """Predicts the class of a test instance based on k nearest neighbors."""
    neighbors = get_neighbors(train_data, test_instance, k)
    output_values = [row[-1] for row in neighbors] # Get the class labels of neighbors
    prediction = Counter(output_values).most_common(1)[0][0] # Majority vote
    return prediction

# 5. Simple example of how to use the functions
if __name__ == '__main__':
    # New data point to classify (features only, no label yet)
    new_point = [5, 5] 
    k_value = 3

    # Predict the classification for the new_point
    predicted_class = predict_classification(dataset, new_point, k_value)
    print(f"The new point {new_point} is classified as: {predicted_class}")

    # Example with another point
    new_point_2 = [1, 1]
    predicted_class_2 = predict_classification(dataset, new_point_2, k_value)
    print(f"The new point {new_point_2} is classified as: {predicted_class_2}")
