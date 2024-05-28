from collections import defaultdict

def find_shortest_chain(word1, word2):
    """
    Finds the shortest word chain connecting two given words using a graph-based approach.

    Args:
        word1: The first word in the chain.
        word2: The last word in the chain.

    Returns:
        A list representing the shortest word chain, or None if no chain is found.
    """

    # Read the word list from the file
    with open("wordsEn.txt", "r") as f:
        word_list = [word.strip() for word in f]

    # Check if the input words are in the word list
    if word1 not in word_list or word2 not in word_list:
        return None

    # Build a graph where nodes are words and edges connect words
    # that differ by one letter
    graph = defaultdict(list)
    for word in word_list:
        for other_word in word_list:
            if len(word) >= 3 and len(other_word) >= 3 and word != other_word and word[-2:] == other_word[:2]:
                graph[word].append(other_word)

    # Perform a breadth-first search to find the shortest path
    queue = [(word1, [word1])]
    visited = set()

    while queue:
        current_word, path = queue.pop(0)
        visited.add(current_word)
        if current_word == word2:
            return path
        for neighbor in graph[current_word]:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))

    # No chain found
    return None

# Get the user input
word1 = input("Enter the first word: ")
word2 = input("Enter the last word: ")

# Find the shortest chain
shortest_chain = find_shortest_chain(word1, word2)

# Print the results
if shortest_chain is None:
    print("No chain found between the two words.")
else:
    print("Shortest word chain:")
    print(" -> ".join(shortest_chain))