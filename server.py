from flask import Flask
from flask import jsonify
from flask import request

import message_generator

app = Flask(__name__)

# Load our models into markov for our endpoint
message_generator.init_markovify()


@app.route("/message/markov")
def generate_message_markov():
    bot_id = request.args.get('botId')
    size = int(request.args.get('size'))
    seed = request.args.get('seed')

    return jsonify(message_generator.generate_message_markov(bot_id, size, seed))


@app.route("/message/tf")
def generate_message_tf():
    size = int(request.args.get('size'))
    seed = request.args.get('seed')

    return jsonify(message_generator.generate_message_tf(size, seed))
