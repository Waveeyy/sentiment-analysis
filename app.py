from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)


def load_model_and_tokenizer(model_path, tokenizer_path):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer('C:\\Users\\lilbl\\OneDrive\\Desktop\\Sentiment Analysis on Movie Reviews//bert_sentiment_analysis.pth',
                                            'C:\\Users\\lilbl\\OneDrive\\Desktop\\Sentiment Analysis on Movie Reviews//bert_sentiment_analysis_tokenizer')


def predict_sentiment(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)

    logits = outputs[0]
    predicted_class_id = logits.argmax().item()
    return "Positive" if predicted_class_id == 1 else "Negative"


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ""
    if request.method == 'POST':
        review_text = request.form.get('review')
        prediction = predict_sentiment(review_text)
    return render_template('index.html', prediction=prediction)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json['review']
    prediction = predict_sentiment(data)
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)
