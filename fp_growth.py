class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None

    def increment(self, count):
        self.count += count

def build_tree(transactions, min_support):
    header_table = {}
    for transaction in transactions:
        for item in transaction:
            header_table[item] = header_table.get(item, 0) + 1

    header_table = {k: v for k, v in header_table.items() if v >= min_support}
    frequent_items = set(header_table.keys())

    if len(frequent_items) == 0:
        return None, None

    for k in header_table:
        header_table[k] = [header_table[k], None]

    root = FPNode(None, None, None)

    for transaction in transactions:
        transaction = [item for item in transaction if item in frequent_items]
        transaction.sort(key=lambda item: header_table[item][0], reverse=True)
        current_node = root
        for item in transaction:
            current_node = update_tree(item, current_node, header_table)

    return root, header_table

def update_tree(item, node, header_table):
    if item in node.children:
        child = node.children[item]
        child.increment(1)
    else:
        child = FPNode(item, 1, node)
        node.children[item] = child
        update_header_table(item, child, header_table)

    return child

def update_header_table(item, target_node, header_table):
    if header_table[item][1] is None:
        header_table[item][1] = target_node
    else:
        current = header_table[item][1]
        while current.next is not None:
            current = current.next
        current.next = target_node

def ascend_fp_tree(node, prefix_path):
    if node.parent is not None:
        prefix_path.append(node.item)
        ascend_fp_tree(node.parent, prefix_path)

def find_frequent_itemsets(tree, header_table, min_support, prefix, frequent_itemsets):
    items = [item[0] for item in sorted(header_table.items(), key=lambda x: x[1][0])]

    for item in items:
        new_prefix = prefix.copy()
        new_prefix.append(item)
        support = header_table[item][0]
        frequent_itemsets.append((new_prefix, support))

        conditional_tree, conditional_header = build_conditional_tree(item, tree, header_table, min_support)

        if conditional_header is not None:
            find_frequent_itemsets(conditional_tree, conditional_header, min_support, new_prefix, frequent_itemsets)

def build_conditional_tree(item, tree, header_table, min_support):
    prefix_path = []
    node = header_table[item][1]

    while node is not None:
        path = []
        ascend_fp_tree(node, path)
        if len(path) > 1:
            prefix_path.append(path[1:])
        node = node.next

    conditional_tree, conditional_header = build_tree(prefix_path, min_support)

    return conditional_tree, conditional_header

def fp_growth(transactions, min_support):
    tree, header_table = build_tree(transactions, min_support)
    frequent_itemsets = []
    find_frequent_itemsets(tree, header_table, min_support, [], frequent_itemsets)
    return frequent_itemsets

# Example Usage
if __name__ == "__main__":
    transactions = [
        ["A", "B", "C", "D"],
        ["A", "C", "D", "E"],
        ["A", "D", "E"],
        ["B", "D"],
        ["B", "C", "E"]
    ]

    min_support = 2

    frequent_itemsets = fp_growth(transactions, min_support)
    for itemset, support in frequent_itemsets:
        print(f"Itemset: {itemset}, Support: {support}")
