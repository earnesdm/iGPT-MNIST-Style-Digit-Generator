from flask import Flask, render_template, request, redirect, url_for, jsonify
from model import ImageGPT, sample_model

import torch
import matplotlib.pyplot as plt

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
H, W = 28, 28

model = ImageGPT().to(device=device)
model.load_state_dict(torch.load("iGPT_MNIST_JUN.pt", weights_only=True, map_location=device))
model.eval()


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        output = sample_model(model, torch.zeros((1, H * W, 1)).to(device, dtype=torch.float))
        output = output.cpu().detach().numpy().reshape(H, W, 1)

        plt.imshow(output, cmap='grey')
        plt.axis('off')
        plt.savefig('static/images/MNIST_temp.png', bbox_inches='tight')

        return render_template("index.html", image='MNIST_temp.png')
    else:
        return render_template("index.html", image='MNIST_sample.png')


if __name__ == '__main__':
    app.run(debug=True)
