import matplotlib.pyplot as plt
import pandas as pd

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

# Data
data = {
    'Class': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'Precision': [0.83, 0.91, 0.81, 0.72, 0.83, 0.75, 0.84, 0.89, 0.90, 0.91],
    'Recall': [0.86, 0.95, 0.74, 0.65, 0.85, 0.78, 0.91, 0.87, 0.90, 0.88],
    'F1-Score': [0.84, 0.93, 0.78, 0.68, 0.84, 0.77, 0.87, 0.88, 0.90, 0.89]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plotting
fig, ax = plt.subplots(figsize=(10, 2))  # Adjust the size as needed
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(cellText=df.values, colLabels=df.columns,
                 cellLoc='center', loc='center')

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.auto_set_column_width(col=list(range(len(df.columns))))

plt.show()

# Save the table to a file
fig.savefig('images/table.png')
