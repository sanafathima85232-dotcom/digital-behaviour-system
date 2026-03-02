import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="Digital Behaviour Fingerprint System", layout="wide")

# ===============================
# CYBER THEME STYLING
# ===============================
st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

/* Title */
.title-text {
    color: #8b5cf6;
    font-size: 34px;
    font-weight: bold;
}

/* Force Equal Height Columns */
div[data-testid="column"] {
    display: flex;
    flex-direction: column;
}

/* Main Metric Boxes */
.metric-box {
    padding: 20px;
    border-radius: 14px;
    border: 2px solid #6366f1;
    background: linear-gradient(145deg, #1e293b, #0f172a);
    text-align: center;
    height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 0 15px rgba(99,102,241,0.3);
}

/* Small Indicator Boxes */
.small-box {
    padding: 15px;
    border-radius: 12px;
    border: 1.5px solid #334155;
    background: linear-gradient(145deg, #111827, #0f172a);
    height: 100px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 10px rgba(148,163,184,0.2);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown("<div class='title-text'>🧠 Digital Behaviour Fingerprint System</div>", unsafe_allow_html=True)


st.write("---")

# ===============================
# SIDEBAR INPUTS
# ===============================
st.sidebar.header("User Behaviour Inputs")

typing_speed = st.sidebar.slider("Typing Speed (WPM)", 20, 120, 60)
mouse_speed = st.sidebar.slider("Mouse Movement Speed", 10, 100, 50)
login_time = st.sidebar.slider("Login Time (Hour)", 0, 23, 14)
pressure = st.sidebar.slider("Key Pressure Level", 1, 10, 5)

input_data = np.array([[typing_speed, mouse_speed, login_time, pressure]])

# ===============================
# HYBRID MODEL (RF + MLP)
# ===============================
rf = RandomForestClassifier()
mlp = MLPClassifier(max_iter=300)

# Dummy Training Dataset
X_train = np.random.randint(1, 100, (200, 4))
y_train = np.random.randint(0, 2, 200)

rf.fit(X_train, y_train)
mlp.fit(X_train, y_train)

rf_prob = rf.predict_proba(input_data)[0][1]
mlp_prob = mlp.predict_proba(input_data)[0][1]

# Hybrid Probability
genuine_prob = (rf_prob + mlp_prob) / 2
imposter_prob = 1 - genuine_prob
risk_score = imposter_prob * 100
deviation_index = abs(0.5 - genuine_prob)

# ===============================
# RISK CLASSIFICATION
# ===============================
if risk_score < 30:
    threat = "Low Risk"
    monitor_mode = "Passive"
elif risk_score < 60:
    threat = "Medium Risk"
    monitor_mode = "Active"
else:
    threat = "High Risk"
    monitor_mode = "High Alert"

# ===============================
# AUTHENTICATION RESULT
# ===============================
st.write("## 🔐 Authentication Result")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class='metric-box'>
        <h3 style='color:#22c55e;'>Genuine Probability</h3>
        <h1 style='color:white;'>{genuine_prob*100:.2f}%</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-box'>
        <h3 style='color:#ef4444;'>Risk Score</h3>
        <h1 style='color:white;'>{risk_score:.2f}%</h1>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-box'>
        <h3 style='color:#a78bfa;'>Threat Level</h3>
        <h1 style='color:white;'>{threat}</h1>
    </div>
    """, unsafe_allow_html=True)

st.write(" ")

# ===============================
# ADDITIONAL INDICATORS
# ===============================
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown(f"""
    <div class='small-box'>
        <b style='color:#38bdf8;'>Imposter Probability:</b> {imposter_prob*100:.2f}%
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class='small-box'>
        <b style='color:#facc15;'>Behaviour Deviation Index:</b> {deviation_index:.2f}
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class='small-box'>
        <b style='color:#f472b6;'>Monitoring Mode:</b> {monitor_mode}
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# ===============================
# GRAPH DASHBOARD
# ===============================
st.subheader("📊 Behaviour Probability Dashboard")

fig, ax = plt.subplots()

bars = ax.bar(
    ["Genuine", "Imposter"],
    [genuine_prob * 100, imposter_prob * 100],
    color=["#22c55e", "#ef4444"]
)

ax.set_ylabel("Probability (%)", color="white")
ax.set_ylim(0, 100)
ax.set_facecolor("#0f172a")
fig.patch.set_facecolor("#0f172a")
ax.tick_params(colors="white")
ax.yaxis.label.set_color("white")

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 2,
            f'{height:.1f}%',
            ha='center', color='white')

st.pyplot(fig)