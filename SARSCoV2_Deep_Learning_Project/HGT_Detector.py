import sys

import tkinter as tk
from tkinter import filedialog
import random
from tensorflow.keras.models import save_model, load_model
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate   
import os
import subprocess

from kivy.clock import Clock

from sklearn import metrics
from kivy.lang import Builder

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput

from kivy.uix.progressbar import ProgressBar

from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.base import runTouchApp


import tensorflow as tf
#tf.compat.v1.disable_v2_behavior()
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, concatenate, SpatialDropout1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Embedding, \
    SeparableConv1D, Add, BatchNormalization, Activation, LeakyReLU, Flatten
from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate, Lambda, Multiply, LSTM, Bidirectional, PReLU, MaxPooling1D, add
from tensorflow.keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy
from tensorflow.keras.models import Model, Sequential, save_model, load_model
from tensorflow.keras.applications.nasnet import  preprocess_input
from tensorflow.keras.optimizers import Adam, SGD
import math
from tensorflow.keras.preprocessing import sequence
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score


from scipy.stats import gaussian_kde

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import pandas as pd
import re


import scipy.cluster.hierarchy as sch 
from sklearn.cluster import AgglomerativeClustering 





from lime import lime_text
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from lime.lime_text import IndexedString,IndexedCharacters
from lime.lime_base import LimeBase
from sklearn.linear_model import Ridge, lars_path
from lime.lime_text import explanation
from functools import partial
import scipy as sp
from sklearn.utils import check_random_state
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import TransformerMixin

import shap
import dalex as dx


import copy

import math

import pickle

import pandas as pd

import numpy as np
from numpy import mean
from numpy import std
import seaborn as sns; sns.set_theme()
from matplotlib import rcParams




class HGT_Detector(App):
        
    def build(self):
        self.window = GridLayout()
        self.window.cols = 1
        self.window.size_hint = (0.6, 0.7)
        self.window.pos_hint = {"center_x": 0.5, "center_y":0.5}

   
        self.instructions = Label(
                        text= "Upload any file containing a single genome",
                        font_size= 18,
                        color= (100,100,100)
                        )
        self.window.add_widget(self.instructions)

        self.pb = ProgressBar(max=1000)
   
        self.button = Button(
                      text= "Upload File",
                      font_size= 28,
                      size_hint= (1,.5),
                      bold= True,
                      background_color ='#455d6b',
                      #remove darker overlay of background colour
                       background_normal = ""
                      )
        self.window.add_widget(self.button)

        self.button.bind(on_release=self.select_file)
        self.dropdown =  DropDown()
        f = ['Virus,' 'Bacteria']
        btn = Button(text='Virus', size_hint_y=None, height=44)
        btn.bind(on_release=lambda btn: self.dropdown.select(btn.text))
        self.dropdown.add_widget(btn)
        btn2 = Button(text='Bacteria', size_hint_y=None, height=44)
        btn2.bind(on_release=lambda btn: self.dropdown.select(btn2.text))
        self.dropdown.add_widget(btn2)
        self.mainbutton = Button(text='Virus or Bacteria', size_hint=(None, None))
        self.mainbutton.bind(on_release=self.dropdown.open)
        self.dropdown.bind(on_select=lambda instance, x: setattr(self.mainbutton, 'text', x))
        self.window.add_widget(self.mainbutton)

       
        return self.window
    


    def make_complement_strand(self,DNA):
        complement=[]
        rules_for_complement={"1":"4","4":"1","2":"3","3":"2"}
        for letter in DNA:
            complement.append(rules_for_complement[letter])
        return(complement)

    def is_this_a_palindrome(self,DNA): 
        DNA=list(DNA)
        if DNA!=(self.make_complement_strand(DNA)[::-1]):     
            return False
        else:                             
            return True


  
    def select_file(self,instance):


        self.pb.value+=100

        self.root = tk.Tk()
        self.root.withdraw()

        self.selected_file = filedialog.askopenfilenames()
        self.total_genome_count = 0

        if self.selected_file != '':
            self.prepare_data()


    def prepare_data(self):
        print('preparing data')
        self.LSTM_Model = load_model('C:\python_work\Covid Project\LSTM_Final', compile = True)
        self.GRU_Model = load_model('C:\python_work\Covid Project\GRU_Final', compile = True)
        self.Dense_Model = load_model('C:\python_work\Covid Project\Dense_Final', compile = True)
        self.Hybrid_Model = load_model('C:\python_work\Covid Project\Hybrid_Final', compile = True)
        self.recombination_rate_list = []
        self.label_list = []
        genome_length_list = []
        total_genome_list = []
     



        for line in self.selected_file:
            if '1' in line:
              self.label_list.append(1)
            elif '0' in line:
              self.label_list.append(0)
        self.selected_file = str(self.selected_file)
        self.selected_file = self.selected_file[2:-3]

        fi = open(self.selected_file) 
        x=0
        self.sample_list = []
        self.deletion_list = []

        self.date_list = []

        self.mult_genome_list = []
        self.selected_genome_list = [] 
        self.genome_count = 0
      

        for line in fi:
            x+=1
            self.sample_list.append(line)
        self.sample_list = [s.replace("\n", "") for s in self.sample_list]
        self.sample_list = [s.replace(" ", "") for s in self.sample_list]
        self.sample_list = [s.replace("0", "") for s in self.sample_list]
        self.sample_list = [s.replace(",", "") for s in self.sample_list]
        self.sample_list = [s.replace("1", "") for s in self.sample_list]


        print(self.sample_list)

        string = ':_.,?!@#$%^&1234567890QqWwEeRrYyUuIiOoPpSsDdFfHhJjKkLlZzXxVvBbMm>|<'
        for i in range(0,len(self.sample_list)):
            matched_list = [characters in self.sample_list[i] for characters in string]
            for r in range(len(matched_list)):
                if matched_list[r] == True:
                    if i not in self.deletion_list:
                        self.deletion_list.append(i)
        for i in range(0, len(self.deletion_list)):
            self.deletion_list[i] -= i

            del self.sample_list[self.deletion_list[i]]
        print(self.deletion_list)



        self.sample_list = ''.join(self.sample_list)
        print(self.sample_list)
   
        self.encoded_list = []
        self.sliding_window_size = 5

        for x in range(0,len(self.sample_list), self.sliding_window_size):
            self.encoded_list.append(self.sample_list[x:x+120])
  
        self.sample_list = self.encoded_list
      
        self.palindrome_list = []

        
        self.encoded_list = []

        for g in range(0,len(self.sample_list)):
            self.encoded_list.append('')
  
        print(self.sample_list)
        self.integer_encoding()
        max_length = 120

        self.x_train = np.zeros((len(self.encoded_list),max_length))
        for x in range(len(self.encoded_list)):
            for i in range(len(self.encoded_list[x])):
                if i <max_length:
                    self.x_train[x][i] = int(self.encoded_list[x][i])
                else:
                    x+=1
                    i=0

        self.x_train = self.x_train.reshape((len(self.encoded_list),max_length))
        print(self.encoded_list)
        print(self.x_train)
        print(len(self.x_train))
        print(self.encoded_list[15])
     
        self.predict()

   
    def predict(self):
        print('loading model')
        self.text_displayed = 'Loading Model...'



            
        model = load_model('C:\python_work\Covid Project\my_model8(81% Accurate)', compile = True)
        ll = model.layers[28].output
        ll = Dropout(.3, name = 'special_name_01')(ll)
        ll = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu',name = 'special_name_07')(ll)
        shortcut = ll
        ll = BatchNormalization(name = 'special_name_10')(ll)
        ll = add([shortcut, ll])
        ll = LeakyReLU(name = 'special_name_14')(ll)
        ll = Dropout(.2,name = 'special_name_09')(ll)
        shortcut = ll
        ll = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu',name = 'special_name_17')(ll)
        ll = BatchNormalization(name = 'special_name_11')(ll)
        ll = LeakyReLU(name = 'special_name_15')(ll)
        ll = Dropout(.4 ,name = 'special_name_12')(ll)
        ll = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu',name = 'special_name_18')(ll)
        ll = BatchNormalization(name = 'special_name_08')(ll)
        ll = add([shortcut, ll])
        ll = LeakyReLU(name = 'special_name_16')(ll)
        ll = LSTM(50,return_sequences=True,name = 'special_name_06')(ll)
        ll = Dropout(.3, name = 'special_name_02')(ll)
        ll = LSTM(30,name = 'special_name_05')(ll)
        ll = Flatten()(ll)

        ll = Dense(32,activation="relu")(ll)
  
        ll = Dense(1,activation="sigmoid")(ll)

        new_model = Model(inputs=model.input,outputs=ll)

  

        new_model.load_weights("checkpoint(covidlooksgood).hdf5")


        
        f = []
        print('Forming Predictions')
        self.text_displayed = 'Forming Predictions...'



        self.probabilities_LSTM = self.LSTM_Model.predict(self.x_train)
        self.probabilities_GRU = self.GRU_Model.predict(self.x_train)
        self.probabilities_Dense = self.Dense_Model.predict(self.x_train)

        self.probabilities_Hybrid = self.Hybrid_Model.predict(self.x_train)





        self.probabilities = []

        for x in range(0,len(self.probabilities_LSTM)):
        
            avrg = (self.probabilities_LSTM[x]+self.probabilities_GRU[x]+self.probabilities_Dense[x]+self.probabilities_Hybrid[x])/4

            self.probabilities.append(avrg)
      
        self.visualize_data()






    def visualize_data(self):
        print('Visualizing Data..')
        self.text_displayed = 'Visualizing Data...'
        self.bar_y_axis = []
        self.scatter_y_axis = []
        self.x_axis = []
        self.total_count = 0

        for x in range(0,len(self.probabilities)):
            self.bar_y_axis.append(self.probabilities[x][0])
            if self.probabilities[x][0] > .5:
                self.scatter_y_axis.append(1)
                self.total_count +=1 
            else:
                self.scatter_y_axis.append(0)
        print(self.bar_y_axis)

        self.correct_predictions = 0
        self.incorrect_predictions = 0
        
        for x in range(0,(len(self.bar_y_axis) * self.sliding_window_size), self.sliding_window_size):
            self.x_axis.append(x)

        table_dic = zip(self.x_axis, self.bar_y_axis)
        data_table = tabulate(table_dic, headers = ['Genome Position', 'Probability'], tablefmt = 'fancy_grid')
        values = []
        for g in range(0, self.x_axis[-1], 5):
            values.append(g)
        values.append(660)
     
        with open(f'Data_Table.txt', 'w', encoding='utf-8') as f:
            f.write(str(data_table))
        with plt.style.context('seaborn'):
            plt.bar(self.x_axis,self.bar_y_axis, width = 20, alpha = .6, color = 'black')
            plt.xticks(np.arange(10, max(self.x_axis), 100.0))


            plt.title(f'Recombination Rate ({self.selected_file})')
            plt.xlabel('Genome Position')
            plt.ylabel('Probability')
        dot_size = 1250/len(self.x_axis)
        if dot_size > 8:
            dot_size = 7

        self.positive_list = []
        for r in range(0,len(self.x_axis)):
            if self.scatter_y_axis[r] == 1:
                self.positive_list.append(self.x_axis[r])
        print(self.positive_list)
    

        self.cluster_array  = [[]] * len(self.positive_list)
        for x in range(0,len(self.cluster_array)):
            self.cluster_array[x] = [self.positive_list[x], 1]
            
        self.cluster_array = np.array(self.cluster_array)
        print(self.cluster_array)
       

        
        xy = np.vstack([self.x_axis,self.scatter_y_axis])
        z = gaussian_kde(xy)(xy)

   
        self.text_displayed = ''
        figure = plt.gcf()
        figure.set_size_inches(12, 8)
        plt.savefig('graph.png', bbox_inches='tight', dpi = 100)
        process = subprocess.Popen("Data_Table.txt", shell=True)
        process1 = subprocess.Popen("graph.png", shell=True)    

        plt.close()
        return self.window


    def explain(self):
        pipeline = make_pipeline(new_model)

        pipeline.fit(self.x_train, self.scatter_y_axis)
        y_preds = pipeline.predict(self.x_train)
        idx = 11
        text_sample = self.x_train[idx]
        class_names = ['negative', 'positive']

        print('Sample {}: last 80 words (only part used by the model)'.format(idx))
        print('-'*50)
        print(" ".join(text_sample.split()[-80:]))
        print('-'*50)
        print('Probability(positive) =', pipeline.predict_proba([text_sample])[0,1])
        print('True class: %s' % class_names[self.scatter_y_axis[idx]])

        class_names = {0: 'non-recom', 1:'recom'}
        LIME_explainer = LimeTextExplainer(class_names=class_names)
        print(self.x_train[10])



        explanation = LIME_explainer.explain_instance(
         self.x_train[2], 
         new_model.predict
            )
        print(explanation)


    def integer_encoding(self):
        random_value = ['1', '2', '3', '4']
        for x in range(len(self.sample_list)):
            for i in range(len(self.sample_list[x])):
                if self.sample_list[x][i] == 'A':
                    self.encoded_list[x] += '1'
                elif self.sample_list[x][i] == 'C':
                    self.encoded_list[x] += '2'
                elif self.sample_list[x][i] == 'G':
                    self.encoded_list[x] += '3'
                elif self.sample_list[x][i] == 'T':
                    self.encoded_list[x] += '4'
                else:
                    self.encoded_list[x] += random_value[random.randint(0,3)]


    def dbscan(self, data):
        db = DBSCAN(eps=240, min_samples=5).fit(data)
        labels = db.labels_
        no_clusters = len(np.unique(labels) )
        no_noise = np.sum(np.array(labels) == -1, axis=0)

        print('Estimated no. of clusters: %d' % no_clusters)
        print('Estimated no. of noise points: %d' % no_noise)
        self.one_list = []
        for x in range(0,len(self.positive_list)-1):
            self.one_list.append(1)
        self.one_list.append(0)
        xy = np.vstack([self.positive_list,self.one_list])
        z = gaussian_kde(xy)(xy)
        plt.scatter(data[:,0], data[:,1], c=z, marker="o", picker=True)
        plt.title('Two clusters with data')
        plt.xlabel('Axis X[0]')
        plt.ylabel('Axis X[1]')
        plt.show()
      

if __name__=='__main__':

    HGT_Detector().run()