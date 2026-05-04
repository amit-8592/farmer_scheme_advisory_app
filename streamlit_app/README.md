# 🌱 Smart Agriculture Scheme Advisor

AI-powered Streamlit web application that recommends the most suitable
Government agriculture scheme for a farmer based on their profile.
Uses a pre-trained **CatBoost** classifier (`model.pkl`) with column
alignment (`columns.pkl`) and decoded labels (`label_encoder.pkl`).

---

## 📂 File Structure

```
.
├── app.py
├── model.pkl
├── columns.pkl
├── label_encoder.pkl
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the app

```bash
streamlit run app.py
```

Streamlit will open the app automatically in your browser at
`http://localhost:8501`.

---

## 🧑‍🌾 Input Features

| Field         | Options                                           |
|---------------|---------------------------------------------------|
| Land Size     | Small / Medium / Large                            |
| Income Level  | Low / Medium / High                               |
| Crop Type     | Rice / Wheat / Cotton / Maize / Soybean           |
| Irrigation    | Yes / No                                          |
| Soil Type     | Sandy / Clay / Loamy / Black                      |
| Loan Status   | Yes / No                                          |
| Weather Risk  | Low / Medium / High                               |
| Experience    | Beginner / Intermediate / Expert                  |
| State         | Punjab / UP / MP / MH / Bihar                     |

---

## 📈 Output

* **Top-1** Recommended scheme (with confidence %)
* **Top-3** Alternative schemes (with confidence %)
* **Probability distribution** bar chart for all schemes
* **Submitted profile** view for verification

---

## 🛠 Tech Stack

* Streamlit
* scikit-learn
* CatBoost
* pandas / numpy / joblib

---

Built with 🌾 for Indian farmers.
