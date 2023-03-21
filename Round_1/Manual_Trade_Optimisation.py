import itertools

prices = {
    'pizza': { 'pizza': 1, 'wasabi': 0.5, 'snowball': 1.45, 'shells': 0.75 },
    'wasabi': { 'pizza': 1.95, 'wasabi': 1, 'snowball': 3.1, 'shells': 1.49 },
    'snowball': { 'pizza': 0.67, 'wasabi': 0.31, 'snowball': 1, 'shells': 0.48 },
    'shells': { 'pizza': 1.34, 'wasabi': 0.64, 'snowball': 1.98, 'shells': 1 }
}

max_score = 1
for path in itertools.product(['pizza', 'wasabi', 'snowball', 'shells'], repeat=4):
    score = 1
    score *= prices['shells'][path[0]]
    score *= prices[path[0]][path[1]]
    score *= prices[path[1]][path[2]]
    score *= prices[path[2]][path[3]]
    score *= prices[path[3]]['shells']

    if score >= max_score:
        print(path, score)
        max_score = score

# ('pizza', 'pizza', 'pizza', 'pizza') 1.0050000000000001
# ('pizza', 'pizza', 'pizza', 'shells') 1.0050000000000001
# ('pizza', 'pizza', 'shells', 'pizza') 1.0100250000000002
# ('pizza', 'wasabi', 'snowball', 'pizza') 1.0436925000000001