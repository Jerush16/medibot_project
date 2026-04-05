from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from pyngrok import ngrok
from datetime import datetime
import google.generativeai as genai
import pickle
import requests
import pandas as pd
import string
import random
import os

app = Flask(__name__)

# -------------------------- NGROK SETUP --------------------------
ngrok.set_auth_token("36Ctkd2LjWJhfAFYiWT1z3hKZs6_3NcvDk1ZDCCLzPgJLv4Lf")
try:
    public_url = ngrok.connect(5000)
    print("Public URL:", public_url)
except Exception as e:
    print("Ngrok connection error:", e)

# -------------------------- DAILY LOGGER --------------------------
def write_log(sender, user_msg, bot_reply):
    if not os.path.exists("logs"):
        os.makedirs("logs")
    filename = f"logs/{datetime.now().strftime('%d-%m-%Y')}.log"
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    entry = f"{timestamp} | From: {sender} | User: \"{user_msg}\" | Bot: \"{bot_reply}\"\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(entry)

# -------------------------- ONLINE AI (GEMINI) --------------------------
genai.configure(api_key="YOUR_GOOGLE_API_KEY")  # Replace with your Google API Key
online_model = genai.GenerativeModel("gemini-pro")

def ask_gemini(prompt):
    try:
        context = (
            "You are a friendly healthcare assistant. "
            "Give basic health suggestions only. "
            "Do NOT give medical diagnosis. "
            "If symptom is emergency-level, tell user to visit doctor immediately."
        )
        response = online_model.generate_content(
            f"{context}\nUser: {prompt}",
            temperature=0.6,
            max_output_tokens=250
        )
        return response.text
    except:
        return None

# -------------------------- LOAD OFFLINE MODELS --------------------------
def load_model(model_file, label_file, feature_file=None):
    model, label_encoder, features = None, None, []
    try:
        model = pickle.load(open(model_file, "rb"))
        label_encoder = pickle.load(open(label_file, "rb"))
        if feature_file:
            features = pd.read_csv(feature_file, nrows=1).columns.tolist()
    except Exception as e:
        print(f"Error loading {model_file}: {e}")
    return model, label_encoder, features

symptom_model, symptom_label_encoder, symptom_features = load_model(
    "model/symptom_model.pkl",
    "model/symptom_label_encoder.pkl",
    "model/preprocessed_features.csv"
)

blood_model, blood_label_encoder, _ = load_model(
    "model/blood_model.pkl",
    "model/blood_label_encoder.pkl"
)

disease_model, disease_label_encoder, disease_features = load_model(
    "model/disease_model.pkl",
    "model/disease_label_encoder.pkl",
    "model/disease_feature_cols.csv"
)

# Load disease precautions (like symptom model)
try:
    disease_precautions = pickle.load(open("model/disease_precautions.pkl", "rb"))
except:
    disease_precautions = {}

# -------------------------- UTILITY FUNCTIONS --------------------------
def internet_available():
    try:
        requests.get("https://www.google.com", timeout=1)
        return True
    except:
        return False

def preprocess_message(msg):
    return msg.lower().translate(str.maketrans("", "", string.punctuation))

# -------------------------- FAQ RESPONSES --------------------------
faq_greetings = [
    "Hey! 👋 How can I help you today?",
    "Hello! Tell me your symptoms.",
    "Hi there! Ready to assist with your health concerns."
]
faq_followup = [
    "Do you have any other symptoms?",
    "Can you explain your symptoms a bit more?",
    "Anything else bothering you?"
]
faq_thanks = [
    "Thanks for using our service!",
    "Glad to help! Stay healthy!",
    "You're welcome! 😄"
]
faq_wish = [
    "I'm fully charged and ready to help! 🔋 How are you feeling today?",
    "I'm fine! How about you? Any health issues bothering you?",
    "Doing great! How are you today? Any symptoms I should know about?"
]

def faq_response(message):
    msg = message.lower()
    if any(x in msg for x in ["hi", "hello", "hii", "hey"]):
        return random.choice(faq_greetings)
    if any(x in msg for x in ["how are you", "how r u", "how r you"]):
        return random.choice(faq_wish)
    if "thank" in msg:
        return random.choice(faq_thanks)
    if any(x in msg for x in ["yes", "other symptom", "more"]):
        return random.choice(faq_followup)
    return None

# -------------------------- OFFLINE ML PREDICTIONS --------------------------
def ml_predict(message, model, label_encoder, features, precautions_map=None):
    if not model or not label_encoder or not features:
        return None
    msg_clean = preprocess_message(message)
    input_vector = pd.DataFrame([[1 if f in msg_clean else 0 for f in features]], columns=features)
    try:
        pred_idx = model.predict(input_vector)[0]
        prediction = label_encoder.inverse_transform([pred_idx])[0]
        # Add precautions if available
        if precautions_map and prediction.lower() in precautions_map:
            p_text = "\n".join([f"- {p}" for p in precautions_map[prediction.lower()]])
            return f"Predicted disease: {prediction}\nPrecautions:\n{p_text}"
        return prediction
    except:
        return None

def blood_predict(message):
    if not blood_model or not blood_label_encoder:
        return None
    try:
        data = {}
        for item in message.split(","):
            if ":" not in item:
                continue
            k, v = item.split(":")
            data[k.strip().lower()] = float(v.strip())
        df = pd.DataFrame([data])
        pred_idx = blood_model.predict(df)[0]
        prediction = blood_label_encoder.inverse_transform([pred_idx])[0]
        return f"Blood test prediction: {prediction}"
    except:
        return None

# -------------------------- MAIN PROCESSOR --------------------------
def process_message(message):
    # Check FAQ first
    faq_ans = faq_response(message)
    if faq_ans:
        return faq_ans

    responses = []

    # Multi-sentence support: split by punctuation
    sentences = [s.strip() for s in message.replace("?", ".").split(".") if s.strip()]

    for sent in sentences:
        # Symptom-based prediction
        symptom_pred = ml_predict(sent, symptom_model, symptom_label_encoder, symptom_features)
        if symptom_pred:
            responses.append(f"Symptom-based prediction: {symptom_pred}")

        # Blood prediction
        if ":" in sent and "," in sent:
            blood_pred = blood_predict(sent)
            if blood_pred:
                responses.append(blood_pred)

        # Disease model prediction with precautions
        disease_pred = ml_predict(sent, disease_model, disease_label_encoder, disease_features, disease_precautions)
        if disease_pred:
            responses.append(f"Disease model prediction: {disease_pred}")

    # Online fallback
    if not responses and internet_available():
        g = ask_gemini(message)
        if g:
            responses.append(g)

    if not responses:
        responses.append("I couldn't understand that. Can you please repeat again?")

    return "\n\n".join(responses)

# -------------------------- FLASK ROUTES --------------------------
@app.route("/sms", methods=["GET", "POST"])
def sms_reply():
    incoming_msg = request.values.get("Body", "")
    sender = request.values.get("From", "Unknown")

    resp = MessagingResponse()
    if incoming_msg:
        reply = process_message(incoming_msg)
        write_log(sender, incoming_msg, reply)
        resp.message(reply)
    else:
        resp.message("Hello! What can I help you with?")

    return str(resp)

@app.route("/status", methods=["POST"])
def status_callback():
    return "OK"

if __name__ == "__main__":
    app.run(port=5000)
