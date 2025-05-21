'''
🧠 NumPy Overview - NumPy (Numerical Python) is the core library for numerical computing in Python.
It provides:
    1. Powerful N-dimensional arrays
    2. Fast mathematical operations
    3. Tools for linear algebra, Fourier transforms, and random number generation
    4. Interoperability with C/C++/Fortran

📚 Topics from NumPy Documentation
1. 🧱 ndarray: N-dimensional array
    Core object in NumPy.
    Stores elements of the same type in a grid of dimensions (axes).
    Very efficient for large datasets.

Key concepts: 
    .shape, .dtype, .ndim, .size
    Slicing and indexing
    Views vs. copies

2. 🔁 Array Creation
Methods to create arrays:
    np.array(), np.zeros(), np.ones(), np.empty()
    np.arange(), np.linspace()
    np.eye(), np.identity()

3. 🔍 Indexing, Slicing, and Iterating
    Basic slicing: a[1:3]
    Boolean indexing: a[a > 0]
    Fancy indexing: using arrays of indices
    Iterating with np.nditer

4. 🔣 Array Manipulation
    Reshape: reshape(), ravel(), flatten()
    Transpose: .T, swapaxes()
    Stacking: hstack(), vstack(), concatenate()
    Splitting: split(), hsplit(), vsplit()

5. 🧮 Universal Functions (ufuncs)
    Fast element-wise operations.
    Examples: np.add, np.subtract, np.multiply, np.divide
    Also includes: np.exp, np.log, np.sin, np.cos, etc.

6. 📊 Broadcasting
    Allows operations between arrays of different shapes.
    Core rule: smaller array gets stretched to match the shape of the larger one.

7. 🔢 Data Types (dtypes)
    NumPy supports many data types: int32, float64, complex, bool, str, etc.
    You can specify the type with dtype and convert with astype().

8. ➕ Arithmetic and Math Operations
    Element-wise operations: +, -, , /
    Aggregate functions: sum(), mean(), std(), min(), max()
    Matrix operations: dot(), matmul()

9. 🧮 Linear Algebra
    Matrix multiplication: dot(), matmul()
    Matrix inverse: np.linalg.inv()
    Determinant: np.linalg.det()
    Eigenvalues: np.linalg.eig()
    Solving linear systems: np.linalg.solve()

10. 🔁 Random Number Generation
    Using np.random module:
    Random numbers: rand(), randn(), randint()
    Distributions: normal(), uniform(), binomial()
    Shuffling: shuffle()
    Seed control: np.random.seed()

11. 🧠 Statistical Functions
    mean(), median(), std(), var(), percentile()
    corrcoef(), cov()

12. 🧰 Set Operations
    np.unique(), np.intersect1d(), np.union1d(), np.setdiff1d()

13. 🔄 I/O with NumPy
    Save/load data:
        np.save(), np.load() for binary .npy files
        np.savetxt(), np.loadtxt() for text files
        np.genfromtxt() for structured data

14. 🧱 Structured Arrays
    Arrays with custom data types (like records): dtype = [('name', 'S10'), ('age', int)]

15. 🧮 Masked Arrays
    Arrays with some invalid or missing data marked as "masked".
    Use: np.ma.array()

16. 🧮 Mathematical Functions
    Trigonometric: sin(), cos(), tan()
    Exponentials and logs: exp(), log()
    Rounding: round(), floor(), ceil()
    Cumulative: cumsum(), cumprod()

17. ⚙️ Memory Layout
    C (row-major) vs. F (column-major) order
    np.ascontiguousarray(), np.asfortranarray()

18. 🧪 Testing and Debugging
    np.testing.assert_array_equal(), etc.
    Useful for checking array equality and tolerance in tests.

19. 🧩 Integration with Other Libraries
    Compatible with:
    Pandas
    SciPy
    Scikit-learn
    TensorFlow/PyTorch
    Can exchange data with C/C++ using ndarray.ctypes

20. 🚀 Performance Tips
    Avoid Python loops — use vectorized operations
    Use broadcasting
    Use in-place operations: a += b
    Profile your code with %timeit
'''
'''
🆚 Python Built-in Sequences vs NumPy Arrays

| Aspect                    | Python Built-in Sequences (list, tuple)                                | NumPy Arrays (ndarray)                                             |
| ------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Size Mutability           | Growable — you can append, insert, delete elements anytime             | Fixed-size — once created, the size cannot be changed              |
| Data Type                 | Can store mixed data types (e.g., [1, "a", 3.14])                      | Only stores elements of the same data type (e.g., all int)         |
| Memory Efficiency         | Less efficient — objects take more memory                              | More memory-efficient using compact C-style arrays                 |
| Speed                     | Slower for large data and mathematical operations                      | Much faster due to vectorized operations and C backend             |
| Mathematical Operations   | Manual loops required (for, map, zip)                                  | Supports element-wise arithmetic directly                          |
| Broadcasting              | Not supported                                                          | Supported — allows operations between different-sized arrays       |
| Multi-dimensional Support | Nested lists (e.g., [[1, 2], [3, 4]]) are possible but not efficient   | Native support for multi-dimensional arrays (2D, 3D, etc.)         |
| Functionality             | Basic functionality, limited to iteration, indexing                    | Rich library: linear algebra, statistics, Fourier, random, etc.    |
| Indexing                  | Basic positive/negative indexing, slicing                              | Supports advanced indexing: boolean, fancy, multidimensional       |
| Performance               | Slow for large datasets or numeric operations                          | Optimized for large-scale numerical computing                      |
| In-place Modification     | Lists are mutable; tuples are immutable                                | Arrays are mutable (can modify in place)                           |
| Type Enforcement          | No enforcement, can mix types                                          | Enforces uniform data type across the array                        |
| Built-in Aggregates       | Use Python sum(), min(), etc.                                          | Built-in methods: .sum(), .mean(), .max(), .std() etc.             |
| Third-party Integration   | Limited use in data science libraries                                  | Integrated with pandas, matplotlib, TensorFlow, scikit-learn       |
| Interfacing with C/C++    | Difficult                                                              | Easy via NumPy’s C APIs and ndarray.ctypes                         |
| Array Reshaping           | Not directly supported                                                 | Supports reshape(), transpose(), flatten()                         |
| Broadcasted Assignment    | Not available (e.g., [1]5 copies reference)                            | Easily assign values across shapes using broadcasting              |
| File I/O                  | Uses built-in open(), csv module                                       | np.save(), np.load(), np.savetxt(), np.loadtxt() available         |
| Data Masking              | Not built-in                                                           | Supports masked arrays via np.ma                                   |
| Vectorization             | Requires for loops                                                     | Fully vectorized operations for performance                        |
'''

'''
🚀 Why is NumPy so fast?
1. ✅ Uses C under the hood
➡️ NumPy is written in a very fast language called C (which is much faster than Python).
➡️ So when you run NumPy code, it actually uses C speed behind the scenes.

2. ✅ Avoids Python loops
➡️ Instead of doing things one by one with for loops (which are slow in Python), 
    NumPy does vectorized operations — all at once!
➡️ Like doing 1000 additions in one go instead of 1000 separate steps.

3. ✅ Works with arrays, not lists
➡️ NumPy uses arrays, which store data more efficiently than Python lists.
➡️ Arrays take less memory and are faster to process.

4. ✅ Better use of CPU
➡️ NumPy is optimized to use the CPU more efficiently, and sometimes even uses multiple cores.

🔥 Simple Analogy:
➡️ Imagine Python loops are like using a spoon to fill a bucket with water (slow).
➡️ NumPy is like using a hose — fast and powerful!
'''

'''
🚀 Vectorization And Broadcasting
✅ Vectorization:
➡️ Doing operations (like addition, multiplication) on entire arrays at once without using loops.
➡️ It makes the code faster and easier to read.

Example:
'''
import numpy as np
a = np.array([1, 2, 3])
b = a * 2  # [2, 4, 6]

'''
✅ Broadcasting:
➡️ Allows NumPy to automatically match shapes of arrays so you can do operations between arrays of different sizes.

Example:
'''
a = np.array([1, 2, 3])
b = 5
c = a + b  # [6, 7, 8] — 5 is "broadcast" to all elements

# Example 2
a = np.array([[1, 2, 3], 
              [4, 5, 6]])
b = np.array([10, 20, 30])
result = a + b

'''
Here:
a is 2 rows × 3 columns
b is just 1 row of 3 numbers
NumPy broadcasts b to both rows in a like this:
[1+10, 2+20, 3+30]
[4+10, 5+20, 6+30]

Final Result :->
[[11, 22, 33],
 [14, 25, 36]]

🟡 Summary
| Term          | What it means (simple)                             | Example                              |
| ------------- | -------------------------------------------------- | ------------------------------------ |
| Vectorization | Do math on many numbers at once (no loop!)         | array * 2                            |
| Broadcasting  | Adjust shapes so math works across different sizes | array + single number or 2D + 1D     |

'''

'''

NumPy supports a full object-oriented style of programming. This means it uses classes and objects 
    — and it starts with the ndarray, which is the main type of array in NumPy.
For example, ndarray is a class (a kind of blueprint), and it comes with many built-in features:
    1. These include attributes (which give information about the array, like its shape or data type)
    2. And methods (which are actions the array can perform, like calculating the sum or reshaping itself)
    3. Most of these methods also have matching functions in the main NumPy module. So, you can choose to:
    4. Call a method on the array itself (like a.sum()
    5. Or use a NumPy function that does the same thing (like np.sum(a))
    
This gives you the freedom to write code in the way you prefer — whether you like working with objects 
(object-oriented style) or just using functions (functional style). Because of this flexibility, the 
NumPy array syntax (ndarray) has become the standard way to work with multi-dimensional data in Python. 
It's now commonly used for sharing and working with data in many Python libraries and tools.
'''
