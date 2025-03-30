# denoiser


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random as r
from datetime import datetime
from tqdm import tqdm
import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=0"

r.seed(datetime.now().timestamp())
global window_size
window_size = 128
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:")
for gpu in gpus:
    print(gpu)
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7680)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
with tf.device('/device:CPU:0'):


    def uniform_gen(min,max):
        rand_max = int(1e30)
        return ((max-min)*(r.randint(0,rand_max-1)/(1+rand_max))+min)

    def gaussian_gen(uniform1,uniform2,st_dev,mean):
        gaussian = st_dev*np.sqrt(-2*np.log(uniform1))*np.cos(2*np.pi*uniform2)+mean
        return gaussian

    def gauss(time,sd,amp,time0):
        return amp*np.exp(-((time-time0)**2)/sd**2)

    def squaresin(time,amp,freq,time0):
        return amp*np.sin(np.pi*2*freq*(time-time0))**2

    def sin(time,amp,freq,time0):
        return amp*np.sin(np.pi*2*freq*(time-time0))



    # ===================================++++++++++++++++++++++++++++++++++++++++===============================
    #
    #                                       noisy signal & clean signal
    #
    # ===================================++++++++++++++++++++++++++++++++++++++++===============================

    def training_data(bs):
        global window_size
        
        data_clear=np.zeros(shape=(bs,window_size,1),dtype=np.float32)
        for batch in range(bs):
        
        
            offset=uniform_gen(-10,10)
            data_clear+=offset
            
            no_sin=r.randint(0,3)
            no_squaresin=r.randint(0,3)
            no_gauss=r.randint(0,25)

            
            for _ in range(no_sin):
                amp=uniform_gen(-1,1)
                freq=uniform_gen(-64,64)
                time0=uniform_gen(-128,128)
                for j in range(window_size):
                    data_clear[batch,j,0]+=sin(j,amp,freq,time0)
            
            for _ in range(no_squaresin):
                amp=uniform_gen(-1,1)
                freq=uniform_gen(-64,64)
                time0=uniform_gen(-128,128)
                for j in range(window_size):
                    data_clear[batch,j,0]+=squaresin(j,amp,freq,time0)
            
            for _ in range(no_gauss):
                amp=uniform_gen(-10,10)
                time0=uniform_gen(-128,128)
                sd=uniform_gen(0,2)
                for j in range(window_size):
                    data_clear[batch,j,0]+=gauss(j,sd,amp,time0)    
            
            # for _ in range(no_lognorm):
            #     amp=uniform_gen(-1,1)
            #     mean=(0,128)
            #     # time0=uniform_gen(-1,128)
            #     sd=uniform_gen(0,5)
            #     for j in range(window_size):
            #         data_clear+=lognorm(j,mean,amp,sd)  
                    
                    
            # noise addition
            
            data_noise=np.copy(data_clear)
            
            for _ in tqdm(range(window_size)):
                uni_noise1=uniform_gen(0,1)
                uni_noise2=uniform_gen(0,1)
                noise_uni=uniform_gen(-3,3)
                gauss_noise_mean=uniform_gen(-3,3)
                gauss_noise_sd=uniform_gen(0,4)
                
                gaussian_noise=gaussian_gen(uni_noise1,uni_noise2,gauss_noise_sd,gauss_noise_mean)
                
                data_noise[batch,_,0]+=0.5*noise_uni+0.5*gaussian_noise
            data_max=np.max(data_noise)
            data_min=np.min(data_noise)
            data_clear=(data_clear-data_min)/(data_max-data_min)
            data_noise=(data_noise-data_min)/(data_max-data_min)
                
            return [data_clear,data_noise]   
        

    # ===================================++++++++++++++++++++++++++++++++++++++++===============================
    #
    #                                       the model
    #
    # ===================================++++++++++++++++++++++++++++++++++++++++===============================

    def model(n,shape):
        input=tf.keras.Input(shape)
        
        x1=tf.keras.layers.Dense(2*n,"elu")(input)
        x2=tf.keras.layers.Dense(2*n,"elu")(input)
        x3=tf.keras.layers.Dense(2*n,"elu")(input)
        
        neck1=tf.keras.layers.Dense(n//2,"elu")(x1)
        neck2=tf.keras.layers.Dense(n//2,"elu")(x2)
        neck3=tf.keras.layers.Dense(n//2,"elu")(x3)
        
        x1=tf.keras.layers.Dense(2*n,"elu")(neck2)
        x2=tf.keras.layers.Dense(2*n,"elu")(neck1)
        x3=tf.keras.layers.Dense(2*n,"elu")(neck3)
        
        mix=tf.keras.layers.Add()([x1,x2,x3])
        
        out=tf.keras.layers.Dense(1,"elu")(mix)
        
        return tf.keras.Model(input,out)

    net=model(300,(window_size,1))

    net.summary()


    opti=tf.keras.optimizers.Adam(learning_rate=0.003)
    net.compile(optimizer=opti, loss=tf.keras.losses.MeanSquaredError(),metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC()])
    # ===================================++++++++++++++++++++++++++++++++++++++++===============================
    #
    #                                        training loop
    #
    # ===================================++++++++++++++++++++++++++++++++++++++++===============================
    @tf.function
    def train_step(inp,target):
        huber_loss = tf.keras.losses.MeanSquaredError()


        with tf.GradientTape() as tape:
            preds = net(inp)
            loss = huber_loss(target, preds)
        grads=tape.gradient(loss,net.trainable_variables)
        opti.apply_gradients(zip(grads,net.trainable_variables))
        return float(tf.math.reduce_mean(loss))
        
    no_epochs=2000
    total_loss=0.0

    for epoch in range(no_epochs):
        
        
        
        if epoch==0 or epoch%10==0:
            input,target=training_data(1500)
        
        loss=train_step(input,target)
        total_loss+=loss
        
        print(f"Epoch: {epoch+1}, loss: {loss:.10f}")
        net.save(f"/home/tk/Desktop/denoisier.keras")
