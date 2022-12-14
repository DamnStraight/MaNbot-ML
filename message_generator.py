import random

import tensorflow as tf
import markovify

alphabet = "abcdefghijklmnopqrstuvwxyz"

bot_selection: dict[str, str] = {
    "122512846041907203": "DamnStraight",
    "111815854492061696": "Renbot",
    "169633774257176577": "CloudRunner",
    "ALL": "All"
}

models: list[str] = ["122512846041907203", "111815854492061696", "169633774257176577"]

#  Load our pretrained model
one_step_reloaded = tf.saved_model.load('one_step')
markov_models: dict[str, markovify.NewlineText] = {}


def init_markovify() -> None:
    """
    Build the dict containing our user associated markov models
    TODO Use markovify's compressed model format
    """
    for botId in models:
        with open(f"datasets/{botId}.txt") as f:
            text = f.read()
            text_model = markovify.NewlineText(text, state_size=2)
            markov_models[botId] = text_model
            f.close()

    # Combination of all avaialble user datasets
    markov_models["ALL"] = markovify.combine(list(markov_models.values()))
    print("Models loaded")


def generate_message_tf(length: int = 200, seed: str = "") -> str:
    """
    Generate a random message using our trained chat user model
    :param seed: The start of the generated message
    :param length: The length of the message to be generated
    :return:
    """
    states = None
    next_char = tf.constant([seed if seed != "" else random.choice(list(alphabet))])
    result = [next_char]

    for n in range(length):
        next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
        result.append(next_char)

    return f"Nucloud: {tf.strings.join(result)[0].numpy().decode('utf-8')}"


def generate_message_markov(bot_id: str, length: int = 200, seed: str = "") -> str:
    if seed != "":
        try:
            result = markov_models[bot_id].make_sentence_with_start(beginning=seed, size=length, strict=False)
        except markovify.text.ParamError:
            return f"Could not generate sentence beginning with: {seed}"
    else:
        result = markov_models[bot_id].make_short_sentence(length)

    return f"**{bot_selection[bot_id]}**: {result}"
