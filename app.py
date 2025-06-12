import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__)

# Baca dataset dan persiapan
data = pd.read_csv ('proba/personality_dataset.csv')
data = data.dropna()

# Label encoding kolom kategorikal
encoders = {}
for col in data.columns:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

# Fitur dan target
features = [
    'Time_spent_Alone',
    'Stage_fear',
    'Social_event_attendance',
    'Going_outside',
    'Drained_after_socializing',
    'Friends_circle_size',
    'Post_frequency'
]
target = 'Personality'

X = data[features]
y = data[target]     
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model
X = data[features]
y = data[target]
model = GaussianNB()
model.fit(X, y)

y_pred = model.predict(X_test)

print("Akurasi:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Route untuk halaman utama
@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/form")
def index():
    return render_template("index.html", prediction=None)

# Route untuk prediksi (method POST)
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("index"))
    # Ambil data dari form
    time_alone = int(request.form["time_alone"])
    stage_fear = 1 if request.form["stage_fear"] == "Yes" else 0
    social_event = int(request.form["social_event"])
    going_outside = int(request.form["going_outside"])
    drained = 1 if request.form["drained"] == "Yes" else 0
    friends_circle = int(request.form["friends_circle"])
    post_freq = int(request.form["post_freq"])

    # Prediksi
    input_data = [[
        time_alone, stage_fear, social_event,
        going_outside, drained, friends_circle, post_freq
    ]]
    pred_encoded = model.predict(input_data)[0]

    # Konversi hasil ke label asli jika pakai encoder
    if 'Personality' in encoders:
        prediction = encoders['Personality'].inverse_transform([pred_encoded])[0]
    else:
        prediction = pred_encoded
        
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)