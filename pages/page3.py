import streamlit as st
import pickle
import numpy as np

# Load the models
with open('model_klasifikasi_anxiety.pkl', 'rb') as f:
    model_anxiety = pickle.load(f)
with open('model_klasifikasi_depression.pkl', 'rb') as f:
    model_depression = pickle.load(f)
with open('model_klasifikasi_stress.pkl', 'rb') as f:
    model_stress = pickle.load(f)

# Define the questions
questions = [
    "Aku tidak menikmati apapun", "Aku tidak bisa berhenti merasa sedih", 
    "Tidak ada hal menyenangkan yang bisa aku nantikan", "Aku membenci hidupku", 
    "Aku merasa hidupku sangat buruk", "Aku merasa seperti tidak berguna", 
    "Aku benci diriku sendiri", "Aku bisa merasakan jantungku berdetak sangat kencang, meskipun tidak melakukan olahraga berat", 
    "Aku mengalami kesulitan bernapas (misalnya, bernafas cepat), bahkan ketika aku tidak sedang berolahraga dan tidak sedang sakit", 
    "Tanganku terasa bergetar", "Aku merasa seperti akan panik", "Aku merasa ketakutan", 
    "Aku merasa takut tanpa alasan yang jelas", "Aku merasa pusing, seperti mau pingsan.", 
    "Aku merasa sulit untuk rileks", "Aku marah karena hal-hal kecil", 
    "Aku stres tentang banyak hal", "Aku mendapati diriku bereaksi berlebihan terhadap situasi", 
    "Aku mudah tersinggung", "Aku mudah terganggu", 
    "Aku merasa kesal ketika orang lain menggangguku"
]

# Streamlit app
st.title("Penilaian Tingkat Anxiety, Depression, dan Stress")

st.write("Jawab pertanyaan-pertanyaan berikut dengan skala 0-3, di mana:")
st.write("0: Tidak pernah")
st.write("1: Kadang-kadang")
st.write("2: Sering")
st.write("3: Selalu")

# Collect answers
answers = []
for i, question in enumerate(questions):
    answer = st.radio(f"{i+1}. {question}", options=[0, 1, 2, 3], index=0, key=f"q{i+1}")
    answers.append(answer)

# Predict button
if st.button("Submit"):
    # Reshape answers for prediction
    answers = np.array(answers).reshape(1, -1)

    # Predict using each model
    anxiety_result = model_anxiety.predict(answers)[0]
    depression_result = model_depression.predict(answers)[0]
    stress_result = model_stress.predict(answers)[0]

    # Display results
    st.write("### Hasil Penilaian Anda")
    st.write(f"**Anxiety:** {anxiety_result}")
    st.write(f"**Depression:** {depression_result}")
    st.write(f"**Stress:** {stress_result}")
