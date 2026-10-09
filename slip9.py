#Q1
# Write a program to demonstrate the Hill Climbing Algorithm
# for finding the maximum value of a simple objective function.

def f(x):
    return -(x - 3) ** 2 + 9


x = 0

while True:
    current = f(x)

    if f(x + 1) > current:
        x = x + 1

    elif f(x - 1) > current:
        x = x - 1

    else:
        break


print("Maximum x:", x)
print("Maximum value:", f(x))
#Q2
# Write a program to implement the Forward Chaining
# inference mechanism.

facts = {"A", "B"}

rules = [
    ({"A", "B"}, "C"),
    ({"C"}, "D"),
    ({"D"}, "E")
]

print("\nInitial Facts:", facts)

changed = True

while changed:
    changed = False

    for condition, result in rules:
        if condition.issubset(facts) and result not in facts:
            facts.add(result)

            print("Rule applied:", condition, "->", result)

            changed = True


print("Final Facts:", facts)

#OR
# Write a program to implement the Decision Tree
# Classifier for a classification problem.

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


X = [
    [6, 148, 72, 35, 0, 33.6, 0.627, 50],
    [1, 85, 66, 29, 0, 26.6, 0.351, 31],
    [8, 183, 64, 0, 0, 23.3, 0.672, 32],
    [1, 89, 66, 23, 94, 28.1, 0.167, 21],
    [0, 137, 40, 35, 168, 43.1, 2.288, 33],
    [5, 116, 74, 0, 0, 25.6, 0.201, 30],
    [3, 78, 50, 32, 88, 31, 0.248, 26],
    [10, 115, 0, 0, 0, 35.3, 0.134, 29],
    [2, 197, 70, 45, 543, 30.5, 0.158, 53],
    [8, 125, 96, 0, 0, 0, 0.232, 54],
    [4, 110, 92, 0, 0, 37.6, 0.191, 30]
]

y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]


# Split dataset into 70:30
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


# Tune criterion and max_depth
for criterion in ["gini", "entropy"]:
    for depth in [2, 3, 4]:

        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print(
            "Criterion:", criterion,
            "Max Depth:", depth,
            "Accuracy:", accuracy
        )
