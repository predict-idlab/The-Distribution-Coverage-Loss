import tensorflow as tf
import numpy as np
from scipy.special import erfinv
from math import pi
from math import sqrt, erf
from sklearn import metrics as me


def classification_report(y_true,y_pred):
    print(me.classification_report(y_true,y_pred))

def tf_calculate_sigma(q_up,q_low,distribution='normal',alpha=0.9):
    q = alpha
    if distribution == 'normal':
        mult = tf.math.sqrt(2.0)*( tf.math.erfinv(q)-tf.math.erfinv(-q)) #q = 2*p-1, so if q = 0.9 this means predicting the p=0.95 interval
        #just calculate the standard deviation
        nom = tf.subtract(q_up,q_low)#*tf.math.maximum(0.0,tf.sign(q_up-q_low))
        
    elif distribution == 'log_normal':
        mult =  tf.math.sqrt(2.0)*( tf.math.erfinv(q)-tf.math.erfinv(-q))
        nom = tf.subtract(tf.math.log(q_up),tf.math.log(q_low))  #make the NN directly predict the log instead of the actual intervals, as this creates numeric problems
        
    elif distribution == 'logistic':
        mult = tf.math.log(tf.math.square(q)/tf.math.square(1-q))
        nom = tf.subtract(q_up,q_low)
        
    elif distribution == 'shifted_rayleigh':
        p=q/2+0.5 #for q = 0.9, this results in p=0.95
        mult = tf.math.subtract(tf.math.sqrt(-2.0*tf.math.log(1-p)),tf.math.sqrt(-2.0*tf.math.log(p))) #Q here is the quantile, so q=0.9 gives the p=0.9 quantile, while the q=0.1 gives 0.1 quantile
        nom = tf.subtract(q_up,q_low)
    
    return tf.math.abs(tf.divide(nom,mult))

def tf_calculate_mean(q_up,q_low,sigma,distribution='normal',alpha=0.9):
    q = alpha
    if distribution == 'normal':
        mean_up=tf.subtract(q_up,tf.multiply(sigma,tf.math.sqrt(2.0)*tf.math.erfinv(q)))#tf.subtract(q_up,sigma)#
        mean_low=tf.subtract(q_low,tf.multiply(sigma,tf.math.sqrt(2.0)*tf.math.erfinv(-q)))#tf.subtract(q_low,sigma)#
        
    elif distribution == 'log_normal': 
        mu_up=tf.subtract(tf.math.log(q_up),tf.multiply(sigma,tf.math.sqrt(2.0)*tf.math.erfinv(q))) 
        mean_up=tf.math.exp(mu_up+tf.math.square(sigma)/2.0) #make the NN directly predict the log instead of the actual intervals, as this creates numeric problems
        
        mu_low=tf.add(tf.math.log(q_low),tf.multiply(sigma,tf.math.sqrt(2.0)*tf.math.erfinv(q))) 
        mean_low=tf.math.exp(mu_low+tf.math.square(sigma)/2.0) #make the NN directly predict the log instead of the actual intervals, as this creates numeric problems
        
    elif distribution == 'logistic':
        mean_up=q_up-sigma*tf.math.log(q/(1-q))
        mean_low=q_low-sigma*tf.math.log((1-q)/q)
        
    elif distribution == 'shifted_rayleigh':
        p=q/2+0.5 #for q = 0.9, this results in p=0.95
        beta_up=q_up-sigma*tf.math.sqrt(-2.0*tf.math.log(1-p))
        beta_low=q_low-sigma*tf.math.sqrt(-2.0*tf.math.log(p))
        
        mean_up = sigma+beta_up #mode assumption - beta_up+sigma*tf.math.sqrt(0.5*math.pi)
        mean_low = sigma+beta_low #mode assumption - beta_low+sigma*tf.math.sqrt(0.5*math.pi)
    
    return (mean_up+mean_low)/2,tf.math.reduce_mean(tf.math.abs(mean_up-mean_low))#(mean_low+mean_up)/2
    

def tf_Soft_PICP(upper,lower,true_val,s=160.0):
    return tf.reduce_mean(tf.multiply(tf.math.sigmoid(s*(true_val-lower)),tf.math.sigmoid(s*(upper-true_val))),axis=1)

def tf_Soft_PICP_one_dim(upper,lower,true_val,s=160.0):
    return tf.reduce_mean(tf.multiply(tf.math.sigmoid(s*(upper-true_val)),tf.math.sigmoid(s*(true_val-lower))))

def tf_PINAW(Y,upper,lower):
    y_max = tf.math.reduce_max(Y)
    y_min = tf.math.reduce_min(Y)
    y_range = y_max-y_min
    return tf.reduce_mean(tf.math.abs(upper-lower),axis=1)/(y_range)

def tf_PINAW_one_dim(Y,upper,lower):
    y_max = tf.math.reduce_max(Y)
    y_min = tf.math.reduce_min(Y)
    y_range = y_max-y_min
    return tf.reduce_mean(tf.math.abs(upper-lower))/(y_range)
                         
def tf_nmpiw_capt(Y,upper,lower):
    y_max = tf.math.reduce_max(Y)
    y_min = tf.math.reduce_min(Y)
    y_range = y_max-y_min
    return tf_mpiw_capt(Y,upper,lower)/y_range
                         
def tf_mpiw_capt(Y,upper,lower,s=160):
    k = tf.math.sigmoid(s*(Y-lower))*tf.math.sigmoid(s*(upper-Y))
    c = tf.reduce_sum(k)
    return tf.reduce_sum(tf.math.abs(upper-lower)*k)/c 

def tf_pinafd(Y,upper,lower,s=160):
    k = 1-tf.math.sigmoid(s*(Y-lower))*tf.math.sigmoid(s*(upper-Y))
    c = tf.reduce_sum(k)
    epsilon = 1e-10
    y_max = tf.math.reduce_max(Y)
    y_min = tf.math.reduce_min(Y)
    y_range = y_max-y_min
    mins_tf = tf.math.minimum(tf.math.abs(Y-upper),tf.math.abs(lower-Y))
    return tf.reduce_sum(tf.multiply(k,mins_tf))/(y_range*c+epsilon)
    
def tf_ECE(Y,upper,lower,distribution='normal',RMSE_mult=1,CE_mult=1,mpiw_mult=0.1,alpha=0.9):

    sigm = tf_calculate_sigma(upper,lower,distribution,alpha)
    Y_pred,pred_diff = tf_calculate_mean(upper,lower,sigm,distribution,alpha)

    Y_range = tf.math.reduce_max(Y)-tf.math.reduce_min(Y)

    max_points = 1000.0

    c_values = tf.range(0.01,0.99,0.98/1000.0)
    
    c_up =  tf.reshape(0.5+c_values/2.0,[len(c_values),1])
    c_low = tf.reshape(0.5-c_values/2.0,[len(c_values),1])
    
    if distribution == 'normal':
        Y_pred_tiled = tf.tile(tf.transpose(tf.reshape(Y_pred,[len(Y_pred),1])), [1000,1])
        
        c_val_low = tf.matmul(tf.math.erfinv(2.0*c_low-1.0)*tf.math.sqrt(2.0),tf.transpose(tf.reshape(sigm,[len(sigm),1])))
        c_val_high = tf.matmul(tf.math.erfinv(2.0*c_up-1.0)*tf.math.sqrt(2.0),tf.transpose(tf.reshape(sigm,[len(sigm),1])))  

        out_low = Y_pred_tiled+c_val_low
        out_high = Y_pred_tiled+c_val_high
    elif distribution == 'log_normal':
        
        mu = tf.math.log(Y_pred)-tf.math.square(sigm)/2.0
        
        mu_tiled = tf.tile(tf.transpose(tf.reshape(mu,[len(mu),1])), [1000,1])

        c_val_low = tf.matmul(tf.math.erfinv(2.0*c_low-1.0)*tf.math.sqrt(2.0),tf.transpose(tf.reshape(sigm,[len(sigm),1])))
        c_val_high = tf.matmul(tf.math.erfinv(2.0*c_up-1.0)*tf.math.sqrt(2.0),tf.transpose(tf.reshape(sigm,[len(sigm),1])))

        out_low = tf.math.exp(mu+c_val_low)
        out_high = tf.math.exp(mu+c_val_high)

    elif distribution == 'logistic':
        Y_pred_tiled = tf.tile(tf.transpose(tf.reshape(Y_pred,[len(Y_pred),1])), [1000,1])
        
        c_val_low = tf.matmul(tf.math.log(tf.divide(c_low,1.0-c_low)),tf.transpose(tf.reshape(sigm,[len(sigm),1])))
        c_val_high =  tf.matmul(tf.math.log(tf.divide(c_up,1.0-c_up)),tf.transpose(tf.reshape(sigm,[len(sigm),1])))

        out_low = Y_pred_tiled+c_val_low
        out_high = Y_pred_tiled+c_val_high

        
    elif distribution == 'shifted_rayleigh':
        beta=Y_pred-sigm#mode assumption
        
        beta_tiled = tf.tile(tf.transpose(tf.reshape(beta,[len(beta),1])), [1000,1])

        c_val_low = tf.matmul(tf.math.sqrt(-2.0*tf.math.log(1-c_low)),tf.transpose(tf.reshape(sigm,[len(sigm),1])))
        c_val_high =  tf.matmul(tf.math.sqrt(-2.0*tf.math.log(1-c_up)),tf.transpose(tf.reshape(sigm,[len(sigm),1])))

        out_low = beta_tiled+c_val_low
        out_high = beta_tiled+c_val_high
    
    debug_bool = False
    if debug_bool:
        print("sigma: "+str(sigm[0]))
        print("lower pred: "+str(lower[0]))
        print("low: " + str(out_low[900,0]))
        print("high: " + str(out_high[900,0]))
        print("higher pred: "+str(upper[0]))
        print(20*"-")
    
    Y_tiled = tf.tile(tf.transpose(tf.reshape(Y,[len(Y),1])), [1000,1])
    
    
    result=tf_Soft_PICP(out_high,out_low,Y_tiled,s=1000)
    #result=tf_Soft_PICP_one_dim(upper,lower,Y,s=1000)

    RMSE = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y, Y_pred))))
    
    #pinaw = tf.reduce_mean(tf_PINAW(Y_tiled,out_high,out_low))
    #pinaw = tf_PINAW_one_dim(Y,upper,lower)
    # hard uses sign step fn
    K_HU = tf.maximum(0.,tf.sign(upper - Y_tiled))
    K_HL = tf.maximum(0.,tf.sign(Y_tiled - lower))
    K_H = tf.multiply(K_HU, K_HL)
    
    MPIW_c = tf.divide(tf.reduce_sum(tf.abs(upper - lower)*K_H), tf.reduce_sum(K_H)+0.001)#/Y_range #+0.001 to avoid numerical errors when sum KH = 0
    
    
    cov_exp = RMSE_mult*(RMSE+pred_diff)+CE_mult*tf.math.reduce_mean(tf.math.square(tf.minimum(0.,result-c_values)))+mpiw_mult*MPIW_c
    
    return cov_exp

def tf_ce_plot(Y,upper,lower,distribution='normal',alpha=0.9,max_points=1000.0):

    sigm = tf_calculate_sigma(upper,lower,distribution,alpha)
    Y_pred,diff = tf_calculate_mean(upper,lower,sigm,distribution,alpha)

    c_values = tf.range(0.01,0.99,0.98/max_points)
    
    DCE_width = 2*(tf.math.reduce_std(Y)*tf.math.sqrt(2.0)*tf.math.erfinv(c_values))/(tf.math.reduce_max(Y)-tf.math.reduce_min(Y))
    
    c_up =  tf.reshape(0.5+c_values/2.0,[len(c_values),1])
    c_low = tf.reshape(0.5-c_values/2.0,[len(c_values),1])
    
    if distribution == 'normal':
        Y_pred_tiled = tf.tile(tf.transpose(tf.reshape(Y_pred,[len(Y_pred),1])), [int(max_points),1])
        
        c_val_low = tf.matmul(tf.math.erfinv(2.0*c_low-1.0)*tf.math.sqrt(2.0),tf.transpose(tf.reshape(sigm,[len(sigm),1])))
        c_val_high = tf.matmul(tf.math.erfinv(2.0*c_up-1.0)*tf.math.sqrt(2.0),tf.transpose(tf.reshape(sigm,[len(sigm),1])))  

        out_low = Y_pred_tiled+c_val_low
        out_high = Y_pred_tiled+c_val_high

        DCE_width = 2*(tf.math.reduce_std(Y)*tf.math.sqrt(2.0)*tf.math.erfinv(c_values))/(tf.math.reduce_max(Y)-tf.math.reduce_min(Y))


    elif distribution == 'log_normal':
        mu = tf.math.log(Y_pred)-tf.math.square(sigm)/2.0
        
        mu_tiled = tf.tile(tf.transpose(tf.reshape(mu,[len(mu),1])), [int(max_points),1])

        c_val_low = tf.matmul(tf.math.erfinv(2.0*c_low-1.0)*tf.math.sqrt(2.0),tf.transpose(tf.reshape(sigm,[len(sigm),1])))
        c_val_high = tf.matmul(tf.math.erfinv(2.0*c_up-1.0)*tf.math.sqrt(2.0),tf.transpose(tf.reshape(sigm,[len(sigm),1])))

        out_low = tf.math.exp(mu+c_val_low)
        out_high = tf.math.exp(mu+c_val_high)
        
    elif distribution == 'logistic':
        Y_pred_tiled = tf.tile(tf.transpose(tf.reshape(Y_pred,[len(Y_pred),1])), [int(max_points),1])
        
        c_val_low = tf.matmul(tf.math.log(tf.divide(c_low,1.0-c_low)),tf.transpose(tf.reshape(sigm,[len(sigm),1])))
        c_val_high =  tf.matmul(tf.math.log(tf.divide(c_up,1.0-c_up)),tf.transpose(tf.reshape(sigm,[len(sigm),1])))

        out_low = Y_pred_tiled+c_val_low
        out_high = Y_pred_tiled+c_val_high

    elif distribution == 'shifted_rayleigh':
        beta=Y_pred-sigm#mode assumption
        
        beta_tiled = tf.tile(tf.transpose(tf.reshape(beta,[len(beta),1])), [int(max_points),1])
        
        c_val_low = tf.matmul(tf.math.sqrt(-2.0*tf.math.log(1-c_low)),tf.transpose(tf.reshape(sigm,[len(sigm),1])))
        c_val_high =  tf.matmul(tf.math.sqrt(-2.0*tf.math.log(1-c_up)),tf.transpose(tf.reshape(sigm,[len(sigm),1])))

        out_low = beta_tiled+c_val_low
        out_high = beta_tiled+c_val_high

    Y_tiled = tf.tile(tf.transpose(tf.reshape(Y,[len(Y),1])), [int(max_points),1])

    result=tf_Soft_PICP(out_high,out_low,Y_tiled,s=1000)
    pinaw = tf_PINAW(Y_tiled,out_high,out_low)
    DCE_below_width = tf.reduce_mean(tf.math.sigmoid(50*(pinaw[200:]-DCE_width[200:])))


    cov_exp = 2*tf.math.reduce_sum(tf.math.abs(result-c_values))/(max_points)
    width_cov_exp = 2*tf.math.reduce_sum(tf.math.abs(result-c_values))/(max_points)+DCE_below_width

    return result,c_values,cov_exp,pinaw,width_cov_exp,DCE_width

#Source = paper "High-Quality Prediction Intervals for Deep Learning : A Distribution-Free, Ensembled Approach"
def tf_qd(Y_l,upper_l,lower_l,alpha_=0.1,lambda_=0.001):

    soften_=160.0
    
    y_true = Y_l
    y_u = upper_l
    y_l = lower_l
    N_=tf.cast(tf.size(y_true),tf.float32)
    
    K_HU = tf.maximum(0.,tf.sign(y_u - y_true))
    K_HL = tf.maximum(0.,tf.sign(y_true - y_l))
    K_H = tf.multiply(K_HU, K_HL)
    
    K_SU = tf.sigmoid(soften_ * (y_u - y_true))
    K_SL = tf.sigmoid(soften_ * (y_true - y_l))
    K_S = tf.multiply(K_SU, K_SL)
    

    MPIW_c = tf.divide(tf.reduce_sum(tf.abs(y_u - y_l)*K_H), tf.reduce_sum(K_H)+0.001)
    PICP_H = tf.reduce_mean(K_H)
    PICP_S = tf.reduce_mean(K_S)


    Loss_S = MPIW_c+ lambda_*tf.sqrt(N_)* tf.square(tf.maximum(0., (1. - alpha_) - PICP_S))
    
    return Loss_S