# ১. প্রয়োজনীয় লাইব্রেরি ইমপোর্ট করা
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ২. ডেটাসেট রিড করা (ফাইলটি একই ফোল্ডারে থাকতে হবে)
df = pd.read_csv("diabetes.csv")

# ৩. ফিচার (X) এবং টার্গেট (y) আলাদা করা
X = df.drop(columns=["Outcome"]) 
y = df["Outcome"]

# ৪. ডেটা স্প্লিট করা (Training 80%, Testing 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ৫. তিনটি মডেল তৈরি করা
models = {
    "Random Forest Classifier": RandomForestClassifier(random_state=42),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "KNeighborsClassifier": KNeighborsClassifier()
}

best_acc = 0
best_model_name = ""

# ৬. লুপ চালিয়ে প্রতিটি মডেল ট্রেন ও ইভালুয়েট করা
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # মেটরিক্স ক্যালকুলেশন
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # সেরা মডেল ট্র্যাক করা
    if accuracy > best_acc:
        best_acc = accuracy
        best_model_name = name
    
    # রেজাল্ট টেবিল তৈরি
    result_table = pd.DataFrame({
        "Model": [name],
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1]
    })
    
    # হুবহু Word ফাইলের মতো আউটপুট স্টাইল তৈরি করা
    table_str = result_table.to_string(index=True)
    lines = table_str.split('\n')
    
    # হেডারের শুরু এবং এলাইনমেন্ট ঠিক করা
    print("..." + lines[0][3:])
    # ভ্যালুগুলো প্রিন্ট করা
    print(lines[1])
    
    # Confusion Matrix-এর আগে স্পেস (Indentation) দেওয়া হচ্ছে
    print("\n    Confusion Matrix:")
    for row in str(conf_matrix).split('\n'):
        print("    " + row)
    
    # দুটি মডেলের আউটপুটের মাঝে ফাঁকা জায়গা রাখার জন্য
    print("\n")

# ৭. একদম শেষে অটোমেটিক কনক্লুশন লেখা
best_acc_percent = int(round(best_acc * 100, 0))
print(f"\tFrom the above result tables of different Classification model we see that, the highest accuracy is {best_acc_percent}% which is predicted by the {best_model_name}. So, we can say that the best model is {best_model_name}.")