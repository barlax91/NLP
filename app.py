import gradio as gr
import pickle

# Load the pre-trained model
with open("NLP/monVecteur.pkl", "rb") as file:
    model = pickle.load(file)

# Define the input and output interfaces
inputs = gr.inputs.Textbox(lines=5, label="Enter text here:")
outputs = gr.outputs.Label(num_top_classes=1, label="Predicted class:")

# Define the function that runs the model and returns the output
def predict(text):
    prediction = model.predict([text])
    return prediction

# Create the Gradio app
gr.Interface(predict, inputs, outputs, title="NLP Stackoverflow analyse de questions").launch()
