import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import pickle
from keras.models import load_model
import urllib.request
from flask import Flask, jsonify, request, render_template
import random
import os

# Obter lista de todas as imagens

batch_size = 16

img_width = 250
img_height = 60

downsample_factor = 4

# ================= Previsão =============


infile = open("max_length", 'rb')
max_length = pickle.load(infile)
infile.close()

import string

chars = string.printable
chars = chars[:-6]
characters = [c for c in chars]

# ====================== PRÉ PROCESSAMENTO =================
char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), mask_token=None
)

# Mapeando inteiros de volta aos caracteres originais
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

prediction_model = load_model("modelo_treinado.h5")


# prediction_model.summary()


def encode_single_sample(img_path, label):
    # 1. Leia a imagem
    img = tf.io.read_file(img_path)

    # 2. Decodifique e converta para tons de cinza
    img = tf.io.decode_png(img, channels=1)

    # 3. Converta para float32 no intervalo [0, 1]
    img = tf.image.convert_image_dtype(img, tf.float32)

    # 4. Redimensione para o tamanho desejado
    img = tf.image.resize(img, [img_height, img_width])

    # 5. Transponha a imagem porque queremos tempo
    # dimensão para corresponder à largura da imagem
    img = tf.transpose(img, perm=[1, 0, 2])

    # 7. Retorne um dict, pois nosso modelo está esperando duas entradas
    return {"image": img}


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use uma busca gananciosa. Para tarefas complexas, você pode usar a pesquisa de feixe
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
              :, :max_length
              ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def resolver_captcha(url):
    # print(url)

    # response = requests.get(url)
    nome = random.randint(0, 999999)

    urllib.request.urlretrieve(url, f"{nome}.png")

    test_img_path = [f'./{nome}.png']

    validation_dataset = tf.data.Dataset.from_tensor_slices((test_img_path[0:1], ['']))
    validation_dataset = (
        validation_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    for batch in validation_dataset.take(1):
        # print(batch['image'])

        preds = prediction_model(batch['image'])  # reconstructed_model is saved trained model
        pred_texts = decode_batch_predictions(preds)
        # pred_json = {"captcha": pred_text}
        os.remove(f"{nome}.png")
        print(pred_texts[0])
        return pred_texts[0]


app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template("my-form.html")


def main():
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


@app.route('/', methods=['POST', 'GET'])
def my_form_post():
    text1 = request.form['text']
    codigo = resolver_captcha(text1)

    arquivos = {"captcha": f'{codigo}'}
    resposta = jsonify(arquivos)

    return resposta


@app.route('/gay')
def gay():
    return "Você é gay"


if __name__ == "__main__":
    main()
# Rodar API
