def smart_greedy(drivers, trips, costs):
    n_t = len(trips)
    used_d, assignment, total = set(), {}, 0.0
    trip_options = {j: [i for (i, jj) in costs.keys() if jj == j] for j in range(n_t)}
    sorted_trips = sorted(range(n_t), key=lambda j: len(trip_options[j]))

    for j in sorted_trips:
        available = [(i, costs[(i, j)]) for i in trip_options[j] if i not in used_d]
        if not available:
            return None, float('inf')
        available.sort(key=lambda x: x[1])
        best_d, cost = available[0]
        assignment[(best_d, j)] = 1
        used_d.add(best_d)
        total += cost
    return assignment, total