Scribe: Realist Handwriting in Tensorflow
=======
See [blog post](https://greydanus.github.io/2016/08/21/handwriting/)

About
--------
This model is trained on the IAM handwriting dataset and was inspired by the model described by the famous 2014 Alex Graves [paper](https://arxiv.org/abs/1308.0850). It consists of a three-layer recurrent neural network (LSTM cells) with a Gaussian Mixture Density Network (MDN) cap on top. I have also implemented the attention mechanism from the paper which allows the network to 'focus' on character at a time in a sequence as it draws them.

The model at one time step looks like this: 
![overview](static/model_rolled.png?raw=true)

Unrolling in time, we get
![overview](static/model_unrolled.png?raw=true)

I've implemented the attention mechanism from the paper:
![Attention mechanism](static/diag_window.png?raw=true)

iPython Notebooks
--------
For an easy intro to the code (along with equations and explanations) check out these iPython notebooks:
* [inspecting the data](https://nbviewer.jupyter.org/github/greydanus/scribe/blob/master/dataloader.ipynb)
* [sampling from the model](https://nbviewer.jupyter.org/github/greydanus/scribe/blob/master/sample.ipynb)

Dependencies
--------
* All code is written in python 2.7. You will need:
 * Numpy
 * Matplotlib
 * [TensorFlow](https://www.tensorflow.org/versions/master/get_started/os_setup.html#pip_install)
