import numpy as np

from anomaly_detector.outlier_estimator import MissingValueDetector, WindowOutlierDetector


print("Testing `MissingValueDetector`")

nanDetector = MissingValueDetector()

assert nanDetector.predict(np.nan) == True
assert np.array_equal(nanDetector.predict([np.nan]), [True])

assert nanDetector.predict(0) == False
assert nanDetector.predict(1) == False
assert nanDetector.predict(-1) == False

assert nanDetector.predict([0]) == False
assert nanDetector.predict([1]) == False
assert nanDetector.predict([-1]) == False

print("Test completed")


print("Testing `WindowOutlierDetector`")

X = np.arange(0, 100)

windowOutlierDetector1 = WindowOutlierDetector(iqr_fence=1.5, upper_only=False)
windowOutlierDetector1.fit(X)

assert windowOutlierDetector1.predict(-50) == True

assert windowOutlierDetector1.predict(np.nan) == False
assert windowOutlierDetector1.predict(0) == False
assert windowOutlierDetector1.predict(99) == False
assert windowOutlierDetector1.predict(120) == False
assert windowOutlierDetector1.predict(145) == False

assert windowOutlierDetector1.predict(150) == True

assert np.array_equal(
    windowOutlierDetector1.predict([-50, 0, 99, 120, 145, 150]),
    [True, False, False, False, False, True])


windowOutlierDetector2 = WindowOutlierDetector(iqr_fence=1.5, upper_only=True)
windowOutlierDetector2.fit(X)

assert windowOutlierDetector2.predict(np.nan) == False
assert windowOutlierDetector2.predict(-50) == False
assert windowOutlierDetector2.predict(0) == False
assert windowOutlierDetector2.predict(99) == False
assert windowOutlierDetector2.predict(120) == False
assert windowOutlierDetector2.predict(145) == False

assert windowOutlierDetector2.predict(150) == True

assert np.array_equal(
    windowOutlierDetector2.predict([-50, 0, 99, 120, 145, 150]),
    [False, False, False, False, False, True])

X = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0])

windowOutlierDetector3 = WindowOutlierDetector(iqr_fence=1.5, upper_only=True)
windowOutlierDetector3.fit(X)

assert windowOutlierDetector3.predict(np.nan) == False

print("Test completed")

# TODO: T-digest
