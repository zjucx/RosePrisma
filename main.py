########################################################################
#
# The IMG help function for RosePrisma.
#
# This file is the mian function for this project
#
# Implemented in Python 2.7 with TensorFlow v0.11.0rc0
#
########################################################################
#
# This file is part of the RosePrisma project at:
#
# https://github.com/zjucx/RosePrisma.git
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2017 by ZjuCx
#
########################################################################

import src.style_transfer as rose_prisma
import src.image as img

if __name__=="__main__":

    # content image filename
    content_filename = 'image/content02.JPG'
    content_image = img.load_img(content_filename, max_size=None)

    # style image filename
    style_filename = 'image/style08.jpg'
    style_image = img.load_img(style_filename, max_size=None)

    # the 5th layer in vgg16 model
    content_layer_ids = [4]

    # the style we choose the 1 2 3 4th layer in vgg16 model
    style_layer_ids = [1, 2, 3, 4]

    img = rose_prisma.style_transfer(content_image=content_image,
                    style_image=style_image,
                    content_layer_ids=content_layer_ids,
                    style_layer_ids=style_layer_ids,
                    weight_content=1.5,
                    weight_style=10.0,
                    weight_denoise=0.3,
                    num_iter=60,
                    step_size=10.0)