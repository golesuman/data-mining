from itertools import combinations, chain

def get_unique_items(transactions):
    unique_items = set()
    for transaction in transactions:
        unique_items.update(transaction)
    return sorted(list(unique_items))

def get_frequent_1_itemsets(transactions, min_support):
    unique_items = get_unique_items(transactions)
    item_counts = {item: 0 for item in unique_items}

    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1

    frequent_1_itemsets = {frozenset([item]): count for item, count in item_counts.items() if count >= min_support}
    return frequent_1_itemsets

def get_candidate_itemsets(prev_frequent_itemsets, k):
    candidates = set()
    for itemset1 in prev_frequent_itemsets:
        for itemset2 in prev_frequent_itemsets:
            union = itemset1.union(itemset2)
            if len(union) == k:
                candidates.add(union)
    return candidates

def get_frequent_itemsets(transactions, min_support):
    frequent_itemsets = {}
    frequent_1_itemsets = get_frequent_1_itemsets(transactions, min_support)
    frequent_itemsets.update(frequent_1_itemsets)

    k = 2
    while len(frequent_1_itemsets) > 0:
        candidate_itemsets = get_candidate_itemsets(frequent_1_itemsets.keys(), k)
        item_counts = {itemset: 0 for itemset in candidate_itemsets}

        for transaction in transactions:
            subsets = combinations(transaction, k)
            for subset in subsets:
                if frozenset(subset) in item_counts:
                    item_counts[frozenset(subset)] += 1

        frequent_k_itemsets = {itemset: count for itemset, count in item_counts.items() if count >= min_support}
        frequent_itemsets.update(frequent_k_itemsets)
        frequent_1_itemsets = frequent_k_itemsets
        k += 1

    return frequent_itemsets

def generate_association_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset in frequent_itemsets.keys():
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    confidence = frequent_itemsets[itemset] / frequent_itemsets[antecedent]
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence))
    return rules

if __name__ == "__main__":
    transactions = [
        ["apple", "banana", "cherry"],
        ["apple", "banana"],
        ["apple", "cherry"],
        ["banana", "cherry", "apple"],
        ["banana"]
    ]

    min_support = 2
    min_confidence = 0.5

    frequent_itemsets = get_frequent_itemsets(transactions, min_support)
    
    print("Frequent Itemsets:")
    for itemset, support in frequent_itemsets.items():
        print(f"Itemset: {list(itemset)}, Support: {support}")

    association_rules = generate_association_rules(frequent_itemsets, min_confidence)

    print("\nAssociation Rules:")
    for antecedent, consequent, confidence in association_rules:
        print(f"Rule: {list(antecedent)} -> {list(consequent)}, Confidence: {confidence:.2f}")
