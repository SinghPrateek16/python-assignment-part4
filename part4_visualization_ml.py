
Part 4: Data Visualization & Machine Learning

Task 1 — Data Exploration with Pandas
# Task 1 — Student Performance Analysis & Prediction

import pandas as pd

# Prevent Pandas from wrapping rows
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Load dataset
df = pd.read_csv("students.csv")

# 1. First 5 rows
print("=== First 5 Rows ===\n", df.head(5), "\n")

# 2. Shape and data types
print("=== Shape (rows × columns) ===\n", f"{df.shape[0]} rows × {df.shape[1]} columns\n")

print("=== Data Types ===")
print("{:<22} {:<15}".format("Column", "Data Type"))
print("-" * 40)
for col, dtype in df.dtypes.items():
    print("{:<22} {:<15}".format(col, str(dtype)))
print("\n")

# 3. Summary statistics
print("=== Summary Statistics ===")
stats = df.describe()
print("{:<22} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(
    "Column", "count", "mean", "std", "min", "25%", "50%", "max"))
print("-" * 80)
for col in stats.columns:
    row = stats[col]
    print("{:<22} {:<8.0f} {:<8.2f} {:<8.2f} {:<8.0f} {:<8.0f} {:<8.0f} {:<8.0f}".format(
        col, row['count'], row['mean'], row['std'], row['min'], row['25%'], row['50%'], row['max']))
print("\n")

# 4. Count of students who passed and failed
print("=== Pass/Fail Counts ===")
print("{:<10} {:<10}".format("Result", "Count"))
print("-" * 22)
counts = df['passed'].value_counts()
for status, count in counts.items():
    label = "Pass" if status == 1 else "Fail"
    print("{:<10} {:<10}".format(label, count))
print("\n")

# 5. Average score per subject for passing vs failing students
subject_cols = ['math', 'science', 'english', 'history', 'pe']

print("=== Average Scores (Passing Students) ===")
print("{:<12} {:<8}".format("Subject", "Average"))
print("-" * 25)
for subj, avg in df[df['passed'] == 1][subject_cols].mean().items():
    print("{:<12} {:<8.2f}".format(subj, avg))
print("\n")

print("=== Average Scores (Failing Students) ===")
print("{:<12} {:<8}".format("Subject", "Average"))
print("-" * 25)
for subj, avg in df[df['passed'] == 0][subject_cols].mean().items():
    print("{:<12} {:<8.2f}".format(subj, avg))
print("\n")

# 6. Student with highest overall average across subjects
df['avg_score'] = df[subject_cols].mean(axis=1)
best_index = df['avg_score'].idxmax()
best_student = df.loc[best_index]

print("=== Student with Highest Overall Average ===")
print("{:<12} {:<12}".format("Field", "Value"))
print("-" * 25)
print("{:<12} {:<12}".format("Name", best_student['name']))
print("{:<12} {:<12.2f}".format("Average", best_student['avg_score']))
=== First 5 Rows ===
       name  math  science  english  history  pe  attendance_pct  study_hours_per_day  passed
0    Alice    88       92       76       80  95              92                  4.5       1
1      Bob    42       55       48       50  60              65                  1.2       0
2  Charlie    75       70       80       68  88              85                  3.0       1
3    Diana    95       98       91       89  97              98                  6.0       1
4      Eve    38       42       50       45  55              58                  0.8       0 

=== Shape (rows × columns) ===
 15 rows × 9 columns

=== Data Types ===
Column                 Data Type      
----------------------------------------
name                   str            
math                   int64          
science                int64          
english                int64          
history                int64          
pe                     int64          
attendance_pct         int64          
study_hours_per_day    float64        
passed                 int64          


=== Summary Statistics ===
Column                 count    mean     std      min      25%      50%      max     
--------------------------------------------------------------------------------
math                   15       65.00    20.06    30       52       65       95      
science                15       66.73    18.97    35       54       65       98      
english                15       66.20    17.77    40       49       70       91      
history                15       63.40    16.94    28       54       62       92      
pe                     15       74.80    16.66    45       61       75       97      
attendance_pct         15       75.80    14.72    50       64       78       98      
study_hours_per_day    15       2.89     1.66     0        2        3        6       
passed                 15       0.60     0.51     0        0        1        1       


=== Pass/Fail Counts ===
Result     Count     
----------------------
Pass       9         
Fail       6         


=== Average Scores (Passing Students) ===
Subject      Average 
-------------------------
math         78.22   
science      78.56   
english      79.11   
history      73.44   
pe           86.00   


=== Average Scores (Failing Students) ===
Subject      Average 
-------------------------
math         45.17   
science      49.00   
english      46.83   
history      48.33   
pe           58.00   


=== Student with Highest Overall Average ===
Field        Value       
-------------------------
Name         Diana       
Average      94.00       
Task 2 — Data Visualization with Matplotlib

# Task 2 — Data Visualization with Matplotlib
# This script demonstrates creating bar, histogram, scatter,
# box, and line plots to analyze student performance.

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("students.csv")

# Add average score column
subject_cols = ['math', 'science', 'english', 'history', 'pe']
df['avg_score'] = df[subject_cols].mean(axis=1)

# -------------------------------
# Plot 1: Bar Chart — Average score per subject
# -------------------------------
avg_scores = df[subject_cols].mean()
plt.figure(figsize=(6,4))
bars = plt.bar(subject_cols, avg_scores, color='skyblue')

# Title and labels in different colors
plt.title("Average Score per Subject", color='darkred', fontsize=14)
plt.xlabel("Subject", color='purple', fontsize=12)
plt.ylabel("Average Score", color='green', fontsize=12)

# Expand y-axis limit to give space above tallest bar
plt.ylim(0, max(avg_scores) * 1.20)

# Annotate each bar with its average score
for bar, score in zip(bars, avg_scores):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + (bar.get_height() * 0.02),
        f"{score:.2f}",
        ha='center', va='bottom',
        fontsize=10, color='black'
    )

plt.tight_layout()
plt.savefig("plot1_bar.png")
plt.show()

# -------------------------------
# Plot 2: Histogram — Distribution of math scores
# -------------------------------
plt.figure(figsize=(6,4))
plt.hist(df['math'], bins=5, color='lightgreen', edgecolor='black')

# Mean line
mean_value = df['math'].mean()
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f"Mean = {mean_value:.2f}")

# Annotate mean value directly on plot
plt.text(mean_value, plt.ylim()[1]*0.9, f"{mean_value:.2f}",
         color='red', ha='center', va='bottom', fontsize=10)

plt.title("Distribution of Math Scores", color='darkblue', fontsize=14)
plt.xlabel("Math Score", color='brown', fontsize=12)
plt.ylabel("Frequency", color='darkgreen', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("plot2_histogram.png")
plt.show()

# -------------------------------
# Plot 3: Scatter Plot — Study hours vs Avg score
# -------------------------------
plt.figure(figsize=(6,4))
plt.scatter(df[df['passed']==1]['study_hours_per_day'], df[df['passed']==1]['avg_score'], color='blue', label='Pass')
plt.scatter(df[df['passed']==0]['study_hours_per_day'], df[df['passed']==0]['avg_score'], color='orange', label='Fail')
plt.title("Study Hours vs Average Score", color='darkred', fontsize=14)
plt.xlabel("Study Hours per Day", color='purple', fontsize=12)
plt.ylabel("Average Score", color='green', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("plot3_scatter.png")
plt.show()

# -------------------------------
# Plot 4: Box Plot — Attendance distribution (Pass vs Fail)
# -------------------------------
pass_attendance = df[df['passed']==1]['attendance_pct'].tolist()
fail_attendance = df[df['passed']==0]['attendance_pct'].tolist()
plt.figure(figsize=(6,4))
plt.boxplot([pass_attendance, fail_attendance], tick_labels=['Pass', 'Fail'])
plt.title("Attendance Percentage Distribution", color='darkblue', fontsize=14)
plt.ylabel("Attendance %", color='green', fontsize=12)
plt.tight_layout()
plt.savefig("plot4_boxplot.png")
plt.show()

# -------------------------------
# Plot 5: Line Plot — Math vs Science scores per student
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(df['name'], df['math'], marker='o', linestyle='-', color='blue', label='Math')
plt.plot(df['name'], df['science'], marker='s', linestyle='--', color='green', label='Science')
plt.title("Math and Science Scores per Student", color='darkred', fontsize=14)
plt.xlabel("Student Name", color='purple', fontsize=12)
plt.ylabel("Score", color='green', fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("plot5_line.png")
plt.show()





Task 3 — Data Visualization with Seaborn

# Task 3 — Data Visualization with Seaborn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("students.csv")

# Add average score column
subject_cols = ['math', 'science', 'english', 'history', 'pe']
df['avg_score'] = df[subject_cols].mean(axis=1)

# -------------------------------
# Bar plots: Average math and science scores split by pass/fail
# -------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

# Math bar plot
sns.barplot(data=df, x='passed', y='math', ax=ax1,
            hue='passed', palette='Blues', legend=False)
ax1.set_title("Average Math Score by Pass/Fail", color='darkblue')
ax1.set_xlabel("Pass/Fail Status", color='purple')
ax1.set_ylabel("Average Math Score", color='green')
ax1.set_xticks([0,1])
ax1.set_xticklabels(['Fail','Pass'])

# Annotate average values on bars (bold, colored)
for p in ax1.patches:
    value = f"{p.get_height():.1f}"
    ax1.annotate(value,
                 (p.get_x() + p.get_width()/2., p.get_height()),
                 ha='center', va='bottom',
                 fontsize=10, fontweight='bold', color='darkred')

# Science bar plot
sns.barplot(data=df, x='passed', y='science', ax=ax2,
            hue='passed', palette='Greens', legend=False)
ax2.set_title("Average Science Score by Pass/Fail", color='darkblue')
ax2.set_xlabel("Pass/Fail Status", color='purple')
ax2.set_ylabel("Average Science Score", color='green')
ax2.set_xticks([0,1])
ax2.set_xticklabels(['Fail','Pass'])

# Annotate average values on bars (bold, colored)
for p in ax2.patches:
    value = f"{p.get_height():.1f}"
    ax2.annotate(value,
                 (p.get_x() + p.get_width()/2., p.get_height()),
                 ha='center', va='bottom',
                 fontsize=10, fontweight='bold', color='darkred')

fig.tight_layout()
fig.savefig("plot6_seaborn_bar.png")

# -------------------------------
# Scatter plot: Attendance vs Avg Score with regression lines
# -------------------------------
plt.figure(figsize=(6,4))

sns.regplot(data=df[df['passed']==1], x='attendance_pct', y='avg_score',
            scatter=True, label='Pass', color='blue')
sns.regplot(data=df[df['passed']==0], x='attendance_pct', y='avg_score',
            scatter=True, label='Fail', color='orange')

plt.title("Attendance vs Average Score (Pass vs Fail)", color='darkred')
plt.xlabel("Attendance %", color='purple')
plt.ylabel("Average Score", color='green')
plt.legend()

plt.tight_layout()
plt.savefig("plot7_seaborn_scatter.png")

# -------------------------------
# Comment: Seaborn vs Matplotlib
# -------------------------------
# Seaborn made these plots easier because it automatically computes averages
# and regression lines with simple function calls. Matplotlib was needed for
# subplot layout, saving, and adding styled annotations, which required extra steps.


Task 4 — Machine Learning with scikit-learn

# Task 4 — Machine Learning with scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Step 1 — Prepare Data
# -------------------------------
df = pd.read_csv("students.csv")

X = df[['math','science','english','history','pe','attendance_pct','study_hours_per_day']]
y = df['passed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -------------------------------
# Step 2 — Train Model
# -------------------------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

print("Training accuracy:", f"{model.score(X_train_scaled, y_train)*100:.1f}%")

# -------------------------------
# Step 3 — Evaluate Model
# -------------------------------
y_pred = model.predict(X_test_scaled)
print("Test accuracy:", f"{model.score(X_test_scaled, y_test)*100:.1f}%")

# Build results table with formatted output
names_test = df.loc[X_test.index, 'name']
results = pd.DataFrame({
    "Name": names_test,
    "Actual": ["Pass" if a==1 else "Fail" for a in y_test.values],
    "Predicted": ["Pass" if p==1 else "Fail" for p in y_pred],
    "Result": ["✅ Correct" if a==p else "❌ Wrong" for a,p in zip(y_test,y_pred)]
})

print("\n=== Test Predictions ===")
print("Name           Actual   Predicted   Result")
print("-------------------------------------------")
for i,row in results.iterrows():
    print(f"{row['Name']:<14}{row['Actual']:<8}{row['Predicted']:<11}{row['Result']}")

# -------------------------------
# Step 4 — Feature Importance
# -------------------------------
coeffs = model.coef_[0]
features = X.columns
coef_pairs = sorted(zip(features, coeffs), key=lambda x: abs(x[1]), reverse=True)

print("\n=== Feature Importance (Logistic Regression) ===")
print("Feature               Coefficient")
print("----------------------------------")
for feat, coef in coef_pairs:
    print(f"{feat:<20}{coef:.3f}")

# Horizontal bar chart with values and coloured labels
colors = ['green' if c > 0 else 'red' for c in coeffs]
plt.figure(figsize=(8,5))
bars = plt.barh(features, coeffs, color=colors)

plt.title("Feature Coefficients (Logistic Regression)", color='darkblue', fontsize=14)
plt.xlabel("Coefficient Value", color='purple', fontsize=12)
plt.ylabel("Feature", color='green', fontsize=12)

# Annotate coefficient values on each bar
for bar, coef in zip(bars, coeffs):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f"{coef:.2f}", ha='left', va='center',
             fontsize=10, fontweight='bold', color='black')

plt.tight_layout()
plt.show()

# -------------------------------
# Step 5 — Predict for New Student (Bonus)
# -------------------------------
new_student_df = pd.DataFrame(
    [[75, 70, 68, 65, 80, 82, 3.2]],
    columns=['math','science','english','history','pe','attendance_pct','study_hours_per_day']
)

new_student_scaled = scaler.transform(new_student_df)
pred = model.predict(new_student_scaled)[0]
proba = model.predict_proba(new_student_scaled)[0]

print("\n=== New Student Prediction ===")
print(f"Predicted: {'Pass' if pred==1 else 'Fail'}")
print(f"Probabilities → Fail: {proba[0]*100:.1f}%, Pass: {proba[1]*100:.1f}%")
Training accuracy: 100.0%
Test accuracy: 100.0%

=== Test Predictions ===
Name           Actual   Predicted   Result
-------------------------------------------
Jack          Fail    Fail       ✅ Correct
Liam          Fail    Fail       ✅ Correct
Alice         Pass    Pass       ✅ Correct

=== Feature Importance (Logistic Regression) ===
Feature               Coefficient
----------------------------------
english             0.813
attendance_pct      0.522
study_hours_per_day 0.484
pe                  0.475
math                0.438
science             0.323
history             0.263

=== New Student Prediction ===
Predicted: Pass
Probabilities → Fail: 9.2%, Pass: 90.8%
