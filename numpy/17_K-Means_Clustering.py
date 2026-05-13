import math
import numpy as np

def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
	# Your code here
    copy_of_centroids = initial_centroids.copy()

    for i in range(max_iterations):

        list_cluster = [[] for x in range(k)]

        for data in points:
            min_distance = math.inf
            x, y = data[0], data[1]

            chosen_cluster = -1

            for j in range(len(copy_of_centroids)):
                xc, yc = copy_of_centroids[j][0], copy_of_centroids[j][1]

                distance = math.sqrt((x - xc) ** 2 + (y - yc) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    chosen_cluster = j

            list_cluster[chosen_cluster].append(data)

        update_centroids = []
        for each_cluster in list_cluster:
            update_each_centroid = tuple(map(float, np.mean(each_cluster, axis=0)))
            update_centroids.append(update_each_centroid)

        copy_of_centroids = update_centroids

    final_centroids = copy_of_centroids
    return final_centroids