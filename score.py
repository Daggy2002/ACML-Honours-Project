import matplotlib.pyplot as plt
import numpy as np

# Test Accuracy: 0.8388
#               precision    recall  f1-score   support

#     airplane       0.83      0.86      0.84      1000
#   automobile       0.91      0.95      0.93      1000
#         bird       0.81      0.74      0.78      1000
#          cat       0.72      0.65      0.68      1000
#         deer       0.83      0.85      0.84      1000
#          dog       0.75      0.78      0.77      1000
#         frog       0.84      0.91      0.87      1000
#        horse       0.89      0.87      0.88      1000
#         ship       0.90      0.90      0.90      1000
#        truck       0.91      0.88      0.89      1000

#     accuracy                           0.84     10000
#    macro avg       0.84      0.84      0.84     10000
# weighted avg       0.84      0.84      0.84     10000

# This data comes from analysis.py
classes = ['airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
precision = [0.83, 0.91, 0.81, 0.72, 0.83, 0.75, 0.84, 0.89, 0.90, 0.91]
recall = [0.86, 0.95, 0.74, 0.65, 0.85, 0.78, 0.91, 0.87, 0.90, 0.88]
f1_score = [0.84, 0.93, 0.78, 0.68, 0.84, 0.77, 0.87, 0.88, 0.90, 0.89]

# Plotting
x = np.arange(len(classes))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width, precision, width, label='Precision')
bars2 = ax.bar(x, recall, width, label='Recall')
bars3 = ax.bar(x + width, f1_score, width, label='F1 Score')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Precision, Recall and F1-Score by Class')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

# Function to add labels on top of the bars


def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

fig.tight_layout()

plt.show()

# Save image to file
fig.savefig('images/scores.png')
