import signal

from flask import Flask, request, jsonify, render_template
import subprocess
import os
import requests
from flask_cors import CORS
import yaml

app = Flask(__name__, static_folder='static')
CORS(app)

# 设置绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FOLDER = os.path.abspath(os.path.join(BASE_DIR, '..', 'configs'))
BACKEND_SRC_FOLDER = os.path.abspath(os.path.join(BASE_DIR, '..', 'backend', 'src'))
CONTROLLER_ADDRESS = "http://localhost:5555"


@app.route('/get_configs', methods=['GET'])
def get_configs():
    """
    返回 configs 文件夹下的所有配置文件。
    """
    configs = [f for f in os.listdir(CONFIG_FOLDER) if os.path.isfile(os.path.join(CONFIG_FOLDER, f))]
    return jsonify(configs)


@app.route('/read_config', methods=['GET'])
def read_config():
    """
    读取指定的配置文件并返回其内容。
    """
    config_filename = request.args.get('filename')  # 从查询参数获取配置文件名
    config_path = os.path.join(CONFIG_FOLDER, config_filename)
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            try:
                config_data = yaml.safe_load(file)  # 读取YAML文件内容
                return jsonify(config_data), 200
            except yaml.YAMLError as exc:
                return jsonify({'error': 'Error parsing YAML file', 'details': str(exc)}), 500
    else:
        return jsonify({'error': 'File not found'}), 404


@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    global simulation_process
    config_filename = request.json.get('config_file')
    config_path = os.path.join(CONFIG_FOLDER, config_filename)
    simulation_command = [
        "python", "-m", "xinhai.arena.simulation",
        "--config_path", config_path,
        "--debug"
    ]
    try:
        # 切换到 backend/src 目录
        os.chdir(BACKEND_SRC_FOLDER)
        simulation_process = subprocess.Popen(simulation_command)
        return jsonify({"message": "Simulation started successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/stop_simulation', methods=['POST'])
def stop_simulation():
    global simulation_process
    try:
        if simulation_process:
            simulation_process.terminate()
            simulation_process.wait()
            return jsonify({"message": "Simulation stopped successfully"}), 200
        else:
            return jsonify({"error": "No simulation process running"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/fetch_memory/<int:agent_id>', methods=['GET'])
def fetch_memory(agent_id):
    storage_key = f"xinhai_cbt_simulation_chillway-{agent_id}"
    fetch_url = f"{CONTROLLER_ADDRESS}/api/storage/fetch-memory"
    fetch_request = {"storage_key": storage_key}
    try:
        response = requests.post(fetch_url, json=fetch_request)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    """
    主页面。
    """
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=51623)
