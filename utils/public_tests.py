import numpy as np

def compute_cost_test(target):
    # print("Using X with shape (4, 1)")
    # Case 1
    x = np.array([2, 4, 6, 8]).T
    y = np.array([7, 11, 15, 19]).T
    initial_w = 2
    initial_b = 3.0
    cost = target(x, y, initial_w, initial_b)
    assert cost == 0, f"Case 1: Cost must be 0 for a perfect prediction but got {cost}"
    
    # Case 2
    x = np.array([2, 4, 6, 8]).T
    y = np.array([7, 11, 15, 19]).T
    initial_w = 2.0
    initial_b = 1.0
    cost = target(x, y, initial_w, initial_b)
    assert cost == 2, f"Case 2: Cost must be 2 but got {cost}"
    
    # print("Using X with shape (5, 1)")
    # Case 3
    x = np.array([1.5, 2.5, 3.5, 4.5, 1.5]).T
    y = np.array([4, 7, 10, 13, 5]).T
    initial_w = 1
    initial_b = 0.0
    cost = target(x, y, initial_w, initial_b)
    assert np.isclose(cost, 15.325), f"Case 3: Cost must be 15.325 for a perfect prediction but got {cost}"
    
    # Case 4
    initial_b = 1.0
    cost = target(x, y, initial_w, initial_b)
    assert np.isclose(cost, 10.725), f"Case 4: Cost must be 10.725 but got {cost}"
    
    # Case 5
    y = y - 2
    initial_b = 1.0
    cost = target(x, y, initial_w, initial_b)
    assert  np.isclose(cost, 4.525), f"Case 5: Cost must be 4.525 but got {cost}"
    
    print("\033[92mAll tests passed!")
    
def compute_gradient_test(target):
    print("Using X with shape (4, 1)")
    # Case 1
    x = np.array([2, 4, 6, 8]).T
    y = np.array([4.5, 8.5, 12.5, 16.5]).T
    initial_w = 2.
    initial_b = 0.5
    dj_dw, dj_db = target(x, y, initial_w, initial_b)
    #assert dj_dw.shape == initial_w.shape, f"Wrong shape for dj_dw. {dj_dw} != {initial_w.shape}"
    assert dj_db == 0.0, f"Case 1: dj_db is wrong: {dj_db} != 0.0"
    assert np.allclose(dj_dw, 0), f"Case 1: dj_dw is wrong: {dj_dw} != [[0.0]]"
    
    # Case 2 
    x = np.array([2, 4, 6, 8]).T
    y = np.array([4, 7, 10, 13]).T + 2
    initial_w = 1.5
    initial_b = 1
    dj_dw, dj_db = target(x, y, initial_w, initial_b)
    #assert dj_dw.shape == initial_w.shape, f"Wrong shape for dj_dw. {dj_dw} != {initial_w.shape}"
    assert dj_db == -2, f"Case 2: dj_db is wrong: {dj_db} != -2"
    assert np.allclose(dj_dw, -10.0), f"Case 1: dj_dw is wrong: {dj_dw} != -10.0"   
    
    print("\033[92mAll tests passed!")
    

def zscore_normalize_features_test(target):
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
    X_norm, mu, sigma = target(X)
    assert mu.shape == (2,), f"mu must have shape (2,) but got {mu.shape}"
    assert sigma.shape == (2,), f"sigma must have shape (2,) but got {sigma.shape}"
    assert np.allclose(mu, np.array([3.0, 4.0])), f"mu is wrong: {mu}"
    assert np.allclose(sigma, X.std(axis=0)), f"sigma is wrong: {sigma}"
    assert np.allclose(X_norm.mean(axis=0), np.array([0.0, 0.0])), "X_norm mean must be 0 for each feature"
    assert np.allclose(X_norm.std(axis=0), np.array([1.0, 1.0])), "X_norm std must be 1 for each feature"
    print("\033[92mAll tests passed!")


def compute_cost_multi_test(target):
    # Case 1: perfect prediction => cost 0
    X = np.array([[1.0, 1.0],
                  [2.0, 1.0],
                  [3.0, 1.0]])
    w = np.array([1.0, 2.0])
    b = 0.0
    y = X @ w + b
    cost = target(X, y, w, b)
    assert np.isclose(cost, 0.0), f"Case 1: Cost must be 0 but got {cost}"

    # Case 2
    w = np.array([0.0, 0.0])
    b = 0.0
    y = np.array([1.0, 2.0, 3.0])
    cost = target(X, y, w, b)
    expected = (1 / (2 * 3)) * np.sum((0 - y) ** 2)  # (1/6)*14
    assert np.isclose(cost, expected), f"Case 2: Cost must be {expected} but got {cost}"
    print("\033[92mAll tests passed!")


def compute_gradient_multi_test(target):
    # Case 1: perfect prediction => gradients 0
    X = np.array([[1.0, 2.0],
                  [2.0, 3.0]])
    w = np.array([1.0, 2.0])
    b = 0.0
    y = X @ w + b
    dj_dw, dj_db = target(X, y, w, b)
    assert np.allclose(dj_dw, np.zeros_like(w)), f"Case 1: dj_dw must be 0 but got {dj_dw}"
    assert np.isclose(dj_db, 0.0), f"Case 1: dj_db must be 0 but got {dj_db}"

    # Case 2: w,b zeros
    w = np.array([0.0, 0.0])
    b = 0.0
    y = np.array([5.0, 8.0])
    dj_dw, dj_db = target(X, y, w, b)
    expected_db = -np.mean(y)  # -6.5
    expected_dw = (1 / 2) * (X.T @ (-y))  # (1/m) X^T (f-y) where f=0
    assert np.allclose(dj_dw, expected_dw), f"Case 2: dj_dw is wrong: {dj_dw} != {expected_dw}"
    assert np.isclose(dj_db, expected_db), f"Case 2: dj_db is wrong: {dj_db} != {expected_db}"
    print("\033[92mAll tests passed!")

