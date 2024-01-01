from collections import deque
# excrsise 1 solution:
def ReversInPlace(str):
    return str[ ::-1]


def MinMax(arr):
    min1 = arr[0]
    max1 = arr[0]
    for i in range(0,len(arr)-1):
        if arr[i] < min1:
            min1 = arr[i]
        if arr[i]>max1:
            max1 = arr[i]
    return {max1 , min1}


def rmDup(arr):
    myArr = [arr[0]]
    for i in range(0,len(arr)-1):
        if arr[i+1] != myArr[-1] :
            myArr.append(arr[i+1])

    return myArr


# excrsise2 solution:
#def ReversInPlace(list):
 #   size = len(list)
  #  for i in range(0,size//2):
   #     temp = list[i]
    #    list[i]= list[size-i-1]
     #   list[size-i-1] = temp
    #return list

def MidElement(list):
    return list[len(list)//2]

class ListNode:
    def __init__(self, value):
        self.value = value
        self.next = None

def Iscycle(head):
    if not head or not head.next:
        return False

    tortoise = head
    hare = head.next

    while hare and hare.next:
        if tortoise == hare:
            return True

        tortoise = tortoise.next
        hare = hare.next.next

    return False

def isBalancedParentheses(s):
    size = len(s)
    x =0
    y =0
    z =0
    for i in range(0,size):
            if s[i] == '(':
                x =x+1
            elif s[i] == '{':
                y=y+1
            elif s[i] == '[':
                z=z+1

            if s[size-i-1] == '(':
                x =x-1
            elif s[size-i-1] == '{':
                y = y-1
            elif s[size-i-1] == '[':
                z = z-1

    if x!=0 or y!=0 or z!=0:
        return False
    return True


def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif target < arr[mid]:
            right = mid - 1
        else:
            left = mid + 1

    return -1



def bubble_sort_array(arr):
    n = len(arr)

    for i in range(n):
        last_index = n

        for j in range(last_index - 1):
            if arr[j] > arr[j + 1]:
                # Swap elements if they are in the wrong order
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

                # Update the last index to reduce the inner loop range
                last_index = j + 1

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        # Recursive call to sort the first and second halves
        merge_sort(left_half)
        merge_sort(right_half)

        i, j, k = 0, 0, 0

        # Merge the sorted halves
        while i < len(left_half) and j < len(right_half):
            if left_half[i] <= right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        # Copy remaining elements of left_half[] if any
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        # Copy remaining elements of right_half[] if any
        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

def quick_sort(array, low, high):
    if low < high:
        # Find pivot such that elements smaller than pivot are on the left and larger are on the right
        pivot_index = partition(array, low, high)

        # Recursively sort the sub-arrays
        quick_sort(array, low, pivot_index - 1)
        quick_sort(array, pivot_index + 1, high)

def partition(array, low, high):
    # Choose the rightmost element as the pivot
    pivot = array[high]

    # Index of the smaller element
    i = low - 1

    # Traverse the array and swap elements such that smaller elements are on the left
    for j in range(low, high):
        if array[j] <= pivot:
            i += 1
            swap(array, i, j)

    # Swap the pivot element with the element at (i + 1), so the pivot is in its final sorted position
    swap(array, i + 1, high)

    # Return the index of the pivot element
    return i + 1

def swap(array, i, j):
    # Swap elements at indices i and j
    array[i], array[j] = array[j], array[i]

def bitMan(n):
    if n & 1 == 0:
        print("even")
    else:
        print("odd")

def numBits(n):
    count = 0
    while n > 0:
        if n & 1 == 1:
            count += 1
        n >>= 1
    return count

def linearsearch(arr_linear, check):
    for i, element in enumerate(arr_linear):
        if element == check:
            return i
    return -1

class TreeNode:
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None

def binarysearch(root, p, q):
    if not root or root == p:
        return root

    left = binarysearch(root.left, p, q)
    right = binarysearch(root.right, p, q)

    if left and right:
        # Nodes p and q are on different sides of the current root
        return root
    elif left:
        # Both nodes are on the left side
        return left
    else:
        # Both nodes are on the right side
        return right

def PermutationOfString(input_str):
    permutations = []

    def permutation_helper(current_str, remaining_str):
        n = len(remaining_str)
        if n == 0:
            permutations.append(current_str)
        else:
            for i in range(n):
                chosen = remaining_str[i]
                new_str = current_str + chosen
                new_remaining_str = remaining_str[:i] + remaining_str[i + 1:]
                permutation_helper(new_str, new_remaining_str)

    permutation_helper("", input_str)
    return permutations

def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n - 1)

# Let's analyze the time complexity of each sorting algorithm:
#bubble sort:
# - In the worst-case scenario, where the array is in reverse order, it will take \(O(n^2)\) comparisons and swaps.
#- In the best-case scenario, where the array is already sorted, it will take \(O(n)\) comparisons.
#- Therefore, the average-case time complexity is \(O(n^2)\).
#merge sort:
#- The time complexity of merge sort is \(O(n \log n)\) in all cases (worst-case, best-case, and average-case). This is because it consistently divides the array in half and merges the subarrays
#quick sort:
#- In the worst-case scenario, where the pivot consistently selects the maximum or minimum element, the time complexity is \(O(n^2)\).
#- In the average and best-case scenarios, the time complexity is \(O(n \log n)\), where the pivot selection and partitioning steps are reasonably balanced.



def canReachTarget(grid, start, target, k):
    n = len(grid)
    m = len(grid[0])

    # Initialize a 3D array to keep track of visited positions and remaining moves
    visited = [[[False] * (k + 1) for _ in range(m)] for _ in range(n)]

    return dfs(grid, start[0], start[1], target[0], target[1], k, visited)

def dfs(grid, x, y, tx, ty, k, visited):
    # Base case: If out of moves or out of bounds, return False
    if k < 0 or x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]) or grid[x][y] == 1 or visited[x][y][k]:
        return False

    # Base case: If reached the target, return True
    if x == tx and y == ty:
        return True

    # Mark the current position as visited with remaining moves
    visited[x][y][k] = True

    # Explore all four possible directions
    result = dfs(grid, x + 1, y, tx, ty, k - 1, visited) or \
             dfs(grid, x - 1, y, tx, ty, k - 1, visited) or \
             dfs(grid, x, y + 1, tx, ty, k - 1, visited) or \
             dfs(grid, x, y - 1, tx, ty, k - 1, visited)

    # Mark the current position as not visited for backtracking
    visited[x][y][k] = False

    return result

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#classes
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, val):
        self.root = self._insert(self.root, val)

    def _insert(self, root, val):
        if not root:
            return TreeNode(val)

        if val < root.val:
            root.left = self._insert(root.left, val)
        elif val > root.val:
            root.right = self._insert(root.right, val)

        return root

    def delete(self, val):
        self.root = self._delete(self.root, val)

    def _delete(self, root, val):
        if not root:
            return root

        if val < root.val:
            root.left = self._delete(root.left, val)
        elif val > root.val:
            root.right = self._delete(root.right, val)
        else:
            if not root.left:
                return root.right
            elif not root.right:
                return root.left

            root.val = self._min_value(root.right)
            root.right = self._delete(root.right, root.val)

        return root

    def _min_value(self, root):
        min_val = root.val
        while root.left:
            min_val = root.left.val
            root = root.left
        return min_val

    def search(self, val):
        return self._search(self.root, val)

    def _search(self, root, val):
        if not root:
            return False

        if val == root.val:
            return True

        if val < root.val:
            return self._search(root.left, val)
        else:
            return self._search(root.right, val)


from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.adjacency_list = defaultdict(list)

    def add_edge(self, source, destination):
        self.adjacency_list[source].append(destination)
        self.adjacency_list[destination].append(source)

    def get_neighbors(self, vertex):
        return self.adjacency_list.get(vertex, [])

    def get_adjacency_list(self):
        return dict(self.adjacency_list)

    def get_shortest_path(self, start, end):
        distance = {}
        parent = {}
        queue = deque()

        distance[start] = 0
        parent[start] = None
        queue.append(start)

        while queue:
            current = queue.popleft()

            for neighbor in self.get_neighbors(current):
                if neighbor not in distance:
                    distance[neighbor] = distance[current] + 1
                    parent[neighbor] = current
                    queue.append(neighbor)

                    if neighbor == end:
                        # Reconstruct the path
                        path = []
                        while parent[neighbor] is not None:
                            path.append(neighbor)
                            neighbor = parent[neighbor]
                        path.reverse()
                        return path

        return []


#from collections import LinkedList

class HashTables:
    def __init__(self):
        self.array_hash_list = [[] for _ in range(10)]

    def insert(self, element):
        index = element % 10
        self.array_hash_list[index].insert(0, element)

    def delete(self, element):
        index = element % 10
        temp_list = self.array_hash_list[index]
        if element in temp_list:
            temp_list.remove(element)
        else:
            print("Element not found to be deleted")

    def Search(self, element):
        index = element % 10
        temp_list = self.array_hash_list[index]
        return element in temp_list



class QueueByTwoStacks:
    def __init__(self):
        self.st1 = []
        self.st2 = []

    def enqueue(self, element):
        if not self.st1:
            self.st1.append(element)
            while self.st2:
                self.st1.append(self.st2.pop())
        elif not self.st2:
            while self.st1:
                self.st2.append(self.st1.pop())
            self.st2.append(element)
        else:
            print("Something went wrong in pushing the number")

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        if not self.st2:
            return self.st1.pop()
        elif not self.st1:
            while self.st2:
                first = self.st2.pop()
                if not self.st2:
                    break
                self.st1.append(first)
            return first

    def is_empty(self):
        return not self.st1 and not self.st2

    def peek(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        if not self.st2:
            return self.st1[-1]
        elif not self.st1:
            while self.st2:
                first = self.st2.pop()
                if not self.st2:
                    break
                self.st1.append(first)
            self.st1.append(first)
        return first

    def size(self):
        if not self.st2:
            return len(self.st1)
        elif not self.st1:
            return len(self.st2)
        return -1


from queue import Queue

class StackByTwoQueues:
    def __init__(self):
        self.queue1 = Queue()
        self.queue2 = Queue()

    def push(self, element):
        if self.queue2.empty():
            self.queue1.put(element)
        elif self.queue1.empty():
            self.queue2.put(element)
        else:
            print("Something went wrong in pushing the number")

    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")

        last = 0
        if self.queue2.empty():
            while not self.queue1.empty():
                last = self.queue1.get()
                if self.queue1.empty():
                    break
                self.queue2.put(last)
        elif self.queue1.empty():
            while not self.queue2.empty():
                last = self.queue2.get()
                if self.queue2.empty():
                    break
                self.queue1.put(last)

        return last

    def is_empty(self):
        return self.queue1.empty() and self.queue2.empty()

    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")

        last = 0
        if self.queue2.empty():
            while not self.queue1.empty():
                last = self.queue1.get()
                if self.queue1.empty():
                    break
                self.queue2.put(last)
            self.queue2.put(last)
        elif self.queue1.empty():
            while not self.queue2.empty():
                last = self.queue2.get()
                if self.queue2.empty():
                    break
                self.queue1.put(last)
            self.queue1.put(last)

        return last

    def size(self):
        if self.queue2.empty():
            return self.queue1.qsize()
        elif self.queue1.empty():
            return self.queue2.qsize()

        return -1





# Exercise 1
arr = [2, 4, 4, 6, 6, 7, 7, 7, 8, 9]
out = MinMax(arr)
rmDup(arr)
print("min and max in array:", out)
print("array without duplications:")
for i in range(len(arr)):
    if arr[i] != -1:
        print(arr[i], end=", ")
print()

# Exercise 2
str_val = "String"
str_val = ReversInPlace(str_val)
print("Reversed the word String:", str_val)

# Exercise 3
linkedList = [1, 2, 3, 4]
ReversInPlace(linkedList)
print("Reversed linked list of 1 2 3 4:", linkedList)
print("Middle Element:", MidElement(linkedList))
node1 = ListNode(1)
node2 = ListNode(2)
node3 = ListNode(3)
node4 = ListNode(4)
node5 = ListNode(5)

node1.next = node2
node2.next = node3
node3.next = node4
node4.next = node5
node5.next = node2  # Creating a cycle

# Check if the linked list has a cycle
result = Iscycle(node1)
print("Linked List has a cycle:", result)
print("Linked List has a cycle:", result)

# Exercise 4
st = StackByTwoQueues()
st.push(1)
st.push(2)
st.push(3)
st.push(4)
st.push(5)
st.push(6)
print("StackByTwoQueues output")
print("6:", st.pop())
print("5:", st.pop())
print("4:", st.pop())

st1 = QueueByTwoStacks()
st1.enqueue(1)
st1.enqueue(2)
st1.enqueue(3)
st1.enqueue(4)
st1.enqueue(5)
print("QueueByTwoStacks is empty:", st1.is_empty())
print("1:", st1.dequeue())
st1.enqueue(6)
print("2:", st1.dequeue())
st1.enqueue(7)
print("3:", st1.dequeue())
print("4:", st1.dequeue())
print("size of QueueByTwoStacks:", st1.size())
print("5:", st1.peek())

# Exercise 5
st2 = "((())){}[[]]"
istrue = isBalancedParentheses(st2)
print("Is Balanced Parentheses:", istrue)

# Exercise 6
arrlinear = [1, 6, 4, 9, 13, 87, 43, 0]
chek = 13
print("Linear search algo: element found at index:", linearsearch(arrlinear, chek))

root = TreeNode(3)
root.left = TreeNode(5)
root.right = TreeNode(1)
root.left.left = TreeNode(6)
root.left.right = TreeNode(2)
root.right.left = TreeNode(0)
root.right.right = TreeNode(8)
root.left.right.left = TreeNode(7)
root.left.right.right = TreeNode(4)

# Test case 1
node_p = root.left  # Node with value 5
node_q = root.right.right  # Node with value 8
result = binarysearch(root, node_p, node_q)
print("Lowest Common Ancestor (5, 8):", result.val)  # Expected output: 3

# Test case 2
node_p = root.left  # Node with value 5
node_q = root.left.right.right  # Node with value 4
result = binarysearch(root, node_p, node_q)
print("Lowest Common Ancestor (5, 4):", result.val)


# Exercise 7
factn = 6
print("The factorial number of n is:", factorial(factn))

permutation = "abc"
print("The permutations of the string are:")
PermutationOfString(permutation)

# Exercise 8
hash_table = HashTables()
hash_table.insert(3)
hash_table.insert(43)
hash_table.insert(12)
hash_table.insert(9)
hash_table.insert(1)
hash_table.insert(29)
hash_table.insert(6)
print("Is 43 exist?:", hash_table.Search(43))
print("Is 9 exist? before been deleted:", hash_table.Search(9))
hash_table.delete(9)
print("Is 9 exist? after been deleted:", hash_table.Search(9))
hash_table.delete(13)
print("Is 1 exist?:", hash_table.Search(1))

# Exercise 9
iseven = 128
print("Bit Manipulation: is even or odd?")
bitMan(iseven)
setnum = 128
res = numBits(setnum)
print("Number of bits is:", res)

# Exercise 11
n, m = 3, 3
start = [0, 0]
target = [2, 2]
k = 6
grid = [[0, 0, 0], [1, 1, 0], [0, 0, 0]]
isTrue = canReachTarget(grid, start, target, k)
print("Is there a path to the target?:", isTrue)

