

import heapq
import random
import time
from collections import defaultdict

# ------------------------------
# DATA STRUCTURE 1: User-Item Graph
# ------------------------------
class UserItemGraph:
    """
    Graph-based data structure to represent relationships between users and items.
    Each user is connected to items they have interacted with.
    """

    def __init__(self):
        self.graph = defaultdict(set)

    def add_interaction(self, user, item):
        """Add a relationship between user and item."""
        self.graph[user].add(item)

    def get_items(self, user):
        """Return all items interacted by a user."""
        return self.graph.get(user, set())

    def get_users(self):
        """Return all users."""
        return list(self.graph.keys())


# ------------------------------
# DATA STRUCTURE 2: Inverted Index (Item → Users)
# ------------------------------
class InvertedIndex:
    """
    Reverse lookup for items to users who interacted with them.
    Useful for item-based collaborative filtering.
    """

    def __init__(self):
        self.index = defaultdict(set)

    def add_interaction(self, user, item):
        """Record that a user interacted with an item."""
        self.index[item].add(user)

    def get_users(self, item):
        """Return users who interacted with a specific item."""
        return self.index.get(item, set())

    def get_items(self):
        """Return all items."""
        return list(self.index.keys())


# ------------------------------
# DATA STRUCTURE 3: MinHeap for Top-K Recommendations
# ------------------------------
class TopKHeap:
    """
    Custom MinHeap implementation for efficiently maintaining top-k recommendations.
    """

    def __init__(self, k):
        self.k = k
        self.heap = []

    def push(self, score, item):
        """Push new (score, item) pair and maintain top-k items."""
        heapq.heappush(self.heap, (score, item))
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)

    def get_topk(self):
        """Return items sorted by score (highest first)."""
        return sorted(self.heap, key=lambda x: x[0], reverse=True)


# ------------------------------
# RECOMMENDATION ENGINE
# ------------------------------
class RecommendationSystem:
    """
    Core class that ties together data structures to provide item recommendations.
    """

    def __init__(self):
        self.user_item_graph = UserItemGraph()
        self.inverted_index = InvertedIndex()

    def add_interaction(self, user, item):
        """Register an interaction between user and item."""
        self.user_item_graph.add_interaction(user, item)
        self.inverted_index.add_interaction(user, item)

    def recommend(self, target_user, top_k=3):
        """
        Recommend items to a user based on similarity with other users.
        Uses shared interactions for similarity scoring.
        """
        target_items = self.user_item_graph.get_items(target_user)
        candidate_scores = defaultdict(int)

        # Iterate over all items user interacted with
        for item in target_items:
            similar_users = self.inverted_index.get_users(item)
            for user in similar_users:
                if user == target_user:
                    continue
                # Increment similarity score based on shared items
                for candidate_item in self.user_item_graph.get_items(user):
                    if candidate_item not in target_items:
                        candidate_scores[candidate_item] += 1

        # Use MinHeap for Top-K selection
        topk_heap = TopKHeap(k=top_k)
        for item, score in candidate_scores.items():
            topk_heap.push(score, item)

        return topk_heap.get_topk()


# ------------------------------
# TESTING AND PERFORMANCE ANALYSIS
# ------------------------------
def generate_sample_data(rec_sys, num_users=10, num_items=8):
    """Generate synthetic user-item interactions."""
    for user in range(1, num_users + 1):
        for _ in range(random.randint(2, 5)):
            item = f"Item-{random.randint(1, num_items)}"
            rec_sys.add_interaction(f"User-{user}", item)


def test_recommendation_system():
    rec_sys = RecommendationSystem()
    generate_sample_data(rec_sys)

    print("=== USERS AND THEIR ITEMS ===")
    for user in rec_sys.user_item_graph.get_users():
        print(f"{user} → {rec_sys.user_item_graph.get_items(user)}")

    target_user = "User-1"
    start_time = time.time()
    recommendations = rec_sys.recommend(target_user, top_k=3)
    end_time = time.time()

    print(f"\n=== RECOMMENDATIONS FOR {target_user} ===")
    for score, item in recommendations:
        print(f"Item: {item}, Score: {score}")

    print(f"\nExecution Time: {end_time - start_time:.6f} seconds")


# ------------------------------
# OPTIMIZATION TECHNIQUES (PHASE 3)
# ------------------------------
def optimized_recommend(rec_sys, target_user, top_k=3):
    """
    Optimized recommendation function using caching and pruning.
    Avoids redundant similarity computation.
    """
    target_items = rec_sys.user_item_graph.get_items(target_user)
    candidate_scores = defaultdict(int)
    user_cache = {}

    for item in target_items:
        similar_users = rec_sys.inverted_index.get_users(item)
        for user in similar_users:
            if user == target_user:
                continue
            if user not in user_cache:
                user_cache[user] = rec_sys.user_item_graph.get_items(user)
            for candidate_item in user_cache[user]:
                if candidate_item not in target_items:
                    candidate_scores[candidate_item] += 1

    topk_heap = TopKHeap(k=top_k)
    for item, score in candidate_scores.items():
        topk_heap.push(score, item)

    return topk_heap.get_topk()


def performance_comparison():
    rec_sys = RecommendationSystem()
    generate_sample_data(rec_sys, num_users=200, num_items=100)

    target_user = "User-5"

    # Base version
    start = time.time()
    _ = rec_sys.recommend(target_user, top_k=5)
    base_time = time.time() - start

    # Optimized version
    start = time.time()
    _ = optimized_recommend(rec_sys, target_user, top_k=5)
    opt_time = time.time() - start

    print("\n=== PERFORMANCE COMPARISON ===")
    print(f"Base Recommendation Time: {base_time:.6f} sec")
    print(f"Optimized Recommendation Time: {opt_time:.6f} sec")
    print(f"Speedup: {base_time / opt_time:.2f}x")


# ------------------------------
# MAIN EXECUTION
# ------------------------------
if __name__ == "__main__":
    test_recommendation_system()
    performance_comparison()

