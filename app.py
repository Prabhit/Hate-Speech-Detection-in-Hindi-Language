from flask import Flask, render_template, request
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

app = Flask(__name__)
prediction = []
accuracy = []
# Load the saved model
model_path = r"C:\Project\Flask App\model"
tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/hindi-abusive-MuRIL")
model = TFAutoModelForSequenceClassification.from_pretrained(model_path)

def classify(predicted_class):
    if predicted_class == 0:
        return "Non-hate"
    if predicted_class == 1:
        return "Hate"        
@app.route('/', methods=["GET", "POST"])
def index():
    global prediction , accuracy
    if request.method == "POST":
        entry_content = request.form.get("content")
        text = [entry_content]  # Declare text inside the function
        inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
        input_dict = {key: tf.convert_to_tensor(value) for key, value in inputs.items()}
        predictions = model.predict(input_dict)
        logits = predictions['logits'][0]
        probabilities = tf.nn.softmax(logits, axis=-1)
        accuracy=probabilities.numpy()
        predicted_class = tf.argmax(probabilities, axis=-1).numpy()
        prediction=classify(predicted_class)   
         

    return render_template("index.html", word=prediction, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
