import argparse
import time

from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import cv2
import numpy as np
import tensorflow as tf

from models.nets import vnect_model_bn_folded as vnect_model
import utils.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='gpu')
parser.add_argument('--model_file', default='models/weights/vnect_tf')
parser.add_argument('--test_img', default='test_imgs/yuniko.jpg')
parser.add_argument('--input_size', default=368)
parser.add_argument('--num_of_joints', default=21)
parser.add_argument('--pool_scale', default=8)
args = parser.parse_args()

joint_color_code = [[139, 53, 255],
                    [0, 56, 255],
                    [43, 140, 237],
                    [37, 168, 36],
                    [147, 147, 0],
                    [70, 17, 145]]

# Limb parents of each joint
limb_parents = [1, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]

# input scales
scales = [1.0, 0.7]

# Use gpu or cpu
gpu_count = {'GPU':1} if args.device == 'gpu' else {'GPU':0}


# == CREATION ==
# Create model
model_tf = vnect_model.VNect(args.input_size)

# Create session
sess_config = tf.ConfigProto(device_count=gpu_count)
sess = tf.Session(config=sess_config)

# Restore weights
saver = tf.train.Saver()
saver.restore(sess, args.model_file)

# Joints placeholder
joints_2d = np.zeros(shape=(args.num_of_joints, 2), dtype=np.int32)
joints_3d = np.zeros(shape=(args.num_of_joints, 3), dtype=np.float32)

#cam = cv2.VideoCapture(0)



def demo_webcam():
    
   # # Create model
   # model_tf = vnect_model.VNect(args.input_size)
#
   # # Create session
   # sess_config = tf.ConfigProto(device_count=gpu_count)
   # sess = tf.Session(config=sess_config)
#
   # # Restore weights
   # saver = tf.train.Saver()
   # saver.restore(sess, args.model_file)
#
   # # Joints placeholder
   # joints_2d = np.zeros(shape=(args.num_of_joints, 2), dtype=np.int32)
   # joints_3d = np.zeros(shape=(args.num_of_joints, 3), dtype=np.float32)
#
   # cam = cv2.VideoCapture(0)

    while True:
        t1 = time.time()
        input_batch = []

        cam_img = utils.read_square_image('', cam, args.input_size, 'WEBCAM')
        print(type(cam_img))
        orig_size_input = cam_img.astype(np.float32)

        # Create multi-scale inputs
        for scale in scales:
            resized_img = utils.resize_pad_img(orig_size_input, scale, args.input_size)
            input_batch.append(resized_img)

        input_batch = np.asarray(input_batch, dtype=np.float32)
        input_batch /= 255.0
        input_batch -= 0.4

        # Inference
        [hm, x_hm, y_hm, z_hm] = sess.run(
            [model_tf.heapmap, model_tf.x_heatmap, model_tf.y_heatmap, model_tf.z_heatmap],
            feed_dict={model_tf.input_holder: input_batch})

        # Average scale outputs
        hm_size = args.input_size // args.pool_scale
        hm_avg = np.zeros(shape=(hm_size, hm_size, args.num_of_joints))
        x_hm_avg = np.zeros(shape=(hm_size, hm_size, args.num_of_joints))
        y_hm_avg = np.zeros(shape=(hm_size, hm_size, args.num_of_joints))
        z_hm_avg = np.zeros(shape=(hm_size, hm_size, args.num_of_joints))
        for i in range(len(scales)):
            rescale = 1.0 / scales[i]
            scaled_hm = cv2.resize(hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
            scaled_x_hm = cv2.resize(x_hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
            scaled_y_hm = cv2.resize(y_hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
            scaled_z_hm = cv2.resize(z_hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
            mid = [scaled_hm.shape[0] // 2, scaled_hm.shape[1] // 2]
            hm_avg += scaled_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                      mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
            x_hm_avg += scaled_x_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                        mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
            y_hm_avg += scaled_y_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                        mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
            z_hm_avg += scaled_z_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                        mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
        hm_avg /= len(scales)
        x_hm_avg /= len(scales)
        y_hm_avg /= len(scales)
        z_hm_avg /= len(scales)

        # Get 2d joints
        utils.extract_2d_joint_from_heatmap(hm_avg, args.input_size, joints_2d)

        # Get 3d joints
        utils.extract_3d_joints_from_heatmap(joints_2d, x_hm_avg, y_hm_avg, z_hm_avg, args.input_size, joints_3d)


        # Plot 2d joint location
        joint_map = np.zeros(shape=(args.input_size, args.input_size, 3))
        for joint_num in range(joints_2d.shape[0]):
            cv2.circle(joint_map, center=(joints_2d[joint_num][1], joints_2d[joint_num][0]), radius=3,
                        color=(255, 0, 0), thickness=-1)
        # Draw 2d limbs
        utils.draw_limbs_2d(cam_img, joints_2d, limb_parents)

        print('FPS: {:>2.2f}'.format(1 / (time.time() - t1)))

    
        # Display 2d results
        concat_img = np.concatenate((cam_img, joint_map), axis=1)
        cv2.imshow('2D', concat_img.astype(np.uint8))
        if cv2.waitKey(1) == ord('q'): break



class KivyCamera(Image):
    def __init__(self, cam, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.cam = cam
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        # get frame
        ret, frame = self.cam.read()
        # get vnect joint images
        frame = demo_webcam_kivy(frame)
        # trasfer to type
        frame = frame.astype(np.uint8)
        print(frame.shape)
        
        # convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        image_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.texture = image_texture
        
        """
        if ret:
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture
        """


class CamApp(App):
    def build(self):
        self.cam = cv2.VideoCapture(0)
        self.my_camera = KivyCamera(cam=self.cam, fps=30)
        return self.my_camera

    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.cam.release()
        
        
def demo_webcam_kivy(frame):
    

    t1 = time.time()
    input_batch = []

    #cam_img = utils.read_square_image('', cam, args.input_size, 'WEBCAM')
    #_, frame = cam.read()
    cam_img = utils.read_square_webcam(frame, args.input_size, 'WEBCAM')
    orig_size_input = cam_img.astype(np.float32)

    # Create multi-scale inputs
    for scale in scales:
        resized_img = utils.resize_pad_img(orig_size_input, scale, args.input_size)
        input_batch.append(resized_img)

    input_batch = np.asarray(input_batch, dtype=np.float32)
    input_batch /= 255.0
    input_batch -= 0.4

    # Inference
    [hm, x_hm, y_hm, z_hm] = sess.run(
        [model_tf.heapmap, model_tf.x_heatmap, model_tf.y_heatmap, model_tf.z_heatmap],
        feed_dict={model_tf.input_holder: input_batch})

    # Average scale outputs
    hm_size = args.input_size // args.pool_scale
    hm_avg = np.zeros(shape=(hm_size, hm_size, args.num_of_joints))
    x_hm_avg = np.zeros(shape=(hm_size, hm_size, args.num_of_joints))
    y_hm_avg = np.zeros(shape=(hm_size, hm_size, args.num_of_joints))
    z_hm_avg = np.zeros(shape=(hm_size, hm_size, args.num_of_joints))
    for i in range(len(scales)):
        rescale = 1.0 / scales[i]
        scaled_hm = cv2.resize(hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
        scaled_x_hm = cv2.resize(x_hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
        scaled_y_hm = cv2.resize(y_hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
        scaled_z_hm = cv2.resize(z_hm[i, :, :, :], (0, 0), fx=rescale, fy=rescale, interpolation=cv2.INTER_LINEAR)
        mid = [scaled_hm.shape[0] // 2, scaled_hm.shape[1] // 2]
        hm_avg += scaled_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                  mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
        x_hm_avg += scaled_x_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                    mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
        y_hm_avg += scaled_y_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                    mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
        z_hm_avg += scaled_z_hm[mid[0] - hm_size // 2: mid[0] + hm_size // 2,
                    mid[1] - hm_size // 2: mid[1] + hm_size // 2, :]
    hm_avg /= len(scales)
    x_hm_avg /= len(scales)
    y_hm_avg /= len(scales)
    z_hm_avg /= len(scales)

    # Get 2d joints
    utils.extract_2d_joint_from_heatmap(hm_avg, args.input_size, joints_2d)

    # Get 3d joints
    utils.extract_3d_joints_from_heatmap(joints_2d, x_hm_avg, y_hm_avg, z_hm_avg, args.input_size, joints_3d)


    # Plot 2d joint location
    joint_map = np.zeros(shape=(args.input_size, args.input_size, 3))
    for joint_num in range(joints_2d.shape[0]):
        cv2.circle(joint_map, center=(joints_2d[joint_num][1], joints_2d[joint_num][0]), radius=3,
                    color=(255, 0, 0), thickness=-1)
    # Draw 2d limbs
    utils.draw_limbs_2d(cam_img, joints_2d, limb_parents)

    print('FPS: {:>2.2f}'.format(1 / (time.time() - t1)))


    # Display 2d results
    concat_img = np.concatenate((cam_img, joint_map), axis=1)
    print(concat_img)
    return concat_img
    #cv2.imshow('2D', concat_img.astype(np.uint8))
    #if cv2.waitKey(1) == ord('q'): break
       


if __name__ == '__main__':
    
    CamApp().run()

    #demo_webcam()
    
    #cam = cv2.VideoCapture(0) 
    #while True:
    #    concat_img = demo_webcam_kivy(cam)
    #    cv2.imshow('2D', concat_img.astype(np.uint8))
    #    if cv2.waitKey(1) == ord('q'): break
