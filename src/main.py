########################################################################
#
# The style-transfer function for TensorFlow.
#
# This model seems to produce better-looking images in Style Transfer
# than the Inception 5h model that otherwise works well for DeepDream.
#
# Implemented in Python 2.7 with TensorFlow v0.11.0rc0
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/zjucx/RosePrisma.git
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2017 by ZjuCx
#
########################################################################

import numpy as np
import tensorflow as tf
import vgg16

def mean_squared_error(a, b):
    return tf.reduce_mean(tf.square(a - b))

def content_img_loss(sess, model, content_img, layer_ids):
    feed_dict = model.create_feed_dict(image=content_img)
    # get layer tensor of certain layer_ids that calculate last time
    layers = model.get_layer_tensors(layer_ids)
    # update feed_dict and calculate layer tensor again
    values = sess.run(layers, feed_dict=feed_dict)

    with model.graph.as_default():
        losses = []
        for layer, value in zip(layers, values):
            loss = mean_squared_error(tf.constant(layer), tf.constant(value))
            losses.append(loss)
        # loss function
        total_loss = tf.reduce_mean(losses)
    return total_loss


# matrix of dot-products for the vectors output by the style-layers
def gram_matrix(tensor):
    shape = tensor.get_shape()
    
    # Get the number of feature channels for the input tensor,
    # which is assumed to be from a convolutional layer with 4-dim.
    num_channels = int(shape[3])

    # Reshape the tensor so it is a 2-dim matrix. This essentially
    # flattens the contents of each feature-channel.
    matrix = tf.reshape(tensor, shape=[-1, num_channels])
    
    # Calculate the Gram-matrix as the matrix-product of
    # the 2-dim matrix with itself. This calculates the
    # dot-products of all combinations of the feature-channels.
    gram = tf.matmul(tf.transpose(matrix), matrix)

    return gram

def style_img_loss(sess, model, style_img, layer_ids):
    feed_dict = model.create_feed_dict(image=style_img)
    # get layer tensor of certain layer_ids that calculate last time
    layers = model.get_layer_tensors(layer_ids)


    with model.graph.as_default():
        gram_layers = [gram_matrix(layer) for layer in layers]
        # update feed_dict and calculate layer tensor again
        values = sess.run(gram_layers, feed_dict=feed_dict)
        losses = []
        for gram_layer, value in zip(gram_layers, values):
            loss = mean_squared_error(tf.constant(gram_layer), tf.constant(value))
            losses.append(loss)
        # loss function
        total_loss = tf.reduce_mean(losses)
    return total_loss

# reference: https://en.wikipedia.org/wiki/Total_variation_denoising
def create_denoise_loss(model):
    loss = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) + \
           tf.reduce_sum(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))
    return loss

########################################################################
#    Use gradient descent to find an image that minimizes the
#    loss-functions of the content-layers and style-layers. This
#    should result in a mixed-image that resembles the contours
#    of the content-image, and resembles the colours and textures
#    of the style-image.
    
#    Parameters:
#    content_image: Numpy 3-dim float-array with the content-image.
#    style_image: Numpy 3-dim float-array with the style-image.
#    content_layer_ids: List of integers identifying the content-layers.
#    style_layer_ids: List of integers identifying the style-layers.
#    weight_content: Weight for the content-loss-function.
#    weight_style: Weight for the style-loss-function.
#    weight_denoise: Weight for the denoising-loss-function.
#    num_iterations: Number of optimization iterations to perform.
#    step_size: Step-size for the gradient in each iteration.
########################################################################

def style_transfer(content_image, style_image,
                   content_layer_ids, style_layer_ids,
                   weight_content=1.5, weight_style=10.0,
                   weight_denoise=0.3,
                   num_iterations=120, step_size=10.0):
    model = vgg16.VGG16()

    sess = tf.InteractiveSession(graph=model.graph)
    # Print the names of the content-layers.
    print("Content layers:")
    print(model.get_layer_names(content_layer_ids))
    print()

    # Print the names of the style-layers.
    print("Style layers:")
    print(model.get_layer_names(style_layer_ids))
    print()

    # Create the loss-function for the content-layers and -image.
    loss_content = content_img_loss(sess=sess,
                                       model=model,
                                       content_img=content_image,
                                       layer_ids=content_layer_ids)

    # Create the loss-function for the style-layers and -image.
    loss_style = style_img_loss(sess=sess,
                                   model=model,
                                   style_img=style_image,
                                   layer_ids=style_layer_ids)    

    # Create the loss-function for the denoising of the mixed-image.
    loss_denoise = create_denoise_loss(model)

    adj_content = tf.Variable(1e-10, name='adj_content')
    adj_style = tf.Variable(1e-10, name='adj_style')
    adj_denoise = tf.Variable(1e-10, name='adj_denoise')

    # Initialize the adjustment values for the loss-functions.
    sess.run([adj_content.initializer,
                 adj_style.initializer,
                 adj_denoise.initializer])

    update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
    update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
    update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))
