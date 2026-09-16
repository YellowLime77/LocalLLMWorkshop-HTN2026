from flask import Flask, render_template, request, jsonify
import os
# TODO: Import the OpenAI client

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    project_info = data.get('project_info', '')
    style = data.get('style', '')
    instructions = data.get('instructions', '')

    # TODO: Initialize the LLM client
    
    # TODO: Create the prompt using the default Devpost sections
    # (Inspiration, What it does, How we built it, Challenges, Accomplishments, What we learned, What's next)

    # TODO: Call the LLM to generate the Devpost submission in Markdown
    
    # Dummy placeholder response
    generated_text = "This is a placeholder. Complete the TODOs to generate the real submission!"

    return jsonify({"result": generated_text})

if __name__ == '__main__':
    app.run(debug=True, port=5000)