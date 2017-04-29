VGG16_Keras_TensorFlow
====  
This is a image classification by VGG16 pre-trained model.
# Envoriment #
  Python3.5<br />
  Keras2.0<br />
  TensorFlow<br />
  Windows10<br />
  Most py-package (such like numpy or cv2...) you can install by pip install xx.whl from [here.](http://www.lfd.uci.edu/~gohlke/pythonlibs/ "pythonlibs")  

# Used Files #
  _model/synset_words.txt_ : map class id (which is equal to line id) to synset_id and human words.
                           For example, If you get the prediction id is 281, then you should out put the 281th content in                 synset_words.txt.
  _model/vgg16_weights_tf_dim_ordering_tf_kernels.h5_: special VGG16 pre-trained weight file for Keras2.

# Hint #
  Mind the path which is used in windows. If you are UNIX player, should change the format.<br />
  Any other errors happen you can check [my blog](http://www.cnblogs.com/luntai/p/6786500.html "轮胎的博客").<br />
  You should be careful with the Keras's backend (th:theano or tf:tensorflow), because they have different mat size format.<br />
  
# Resource #
  About model you can find in [Github-Glist.](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)<br />
  About VGG16 network structure find in [here.](http://ethereon.github.io/netscope/#/gist/dc5003de6943ea5a6b8b)<br />

# #### #
  If it helps you, please give me a *star*. Thks.
