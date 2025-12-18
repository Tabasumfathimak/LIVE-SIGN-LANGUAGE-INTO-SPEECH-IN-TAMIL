from flask import Flask, render_template, send_from_directory, request
import pandas as pd

app = Flask(__name__)

# Route to serve the admin.html
@app.route('/')
def home():
    df = pd.read_csv('gesture_data.csv', header=None)
    gestures = df[0].values.tolist()  # Only get the 'gesture' column values
    return render_template('admin.html', gestures=gestures)

# Route to serve the gesture_data.csv file
@app.route('/gesture_data.csv')
def serve_csv():
    return send_from_directory('.', 'gesture_data.csv')

# Delete gesture name route
@app.route('/delete_gesture/<name>', methods=['DELETE'])
def delete_gesture(name):
    df = pd.read_csv('gesture_data.csv', header=None)
    df = df[df[0] != name]
    df.to_csv('gesture_data.csv', index=False, header=False)
    return '', 204

# Edit gesture name route
@app.route('/edit_gesture/<old_name>', methods=['POST'])
def edit_gesture(old_name):
    new_name = request.json.get('new_name')
    df = pd.read_csv('gesture_data.csv', header=None)
    df.loc[df[0] == old_name, 0] = new_name
    df.to_csv('gesture_data.csv', index=False, header=False)
    return '', 204

if __name__ == '__main__':
    app.run(debug=True, port=5002)
