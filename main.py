import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from tkinter import *
from tkinter import ttk
import tkinter as tk


#################  read data from files
def read_file():
    data_df = pd.read_csv('penguins.csv')
    data = data_df
    data = pd.get_dummies(data['species'], dummy_na=False)

    ############# data preprocessing
    le = LabelEncoder()
    gender_encoded = le.fit_transform(data_df['gender'])
    data_df['gender'] = gender_encoded
    ############# data preprocessing

    ############# division data
    speciesCol = data
    data_df = data_df.drop(columns=['species'])
    for column in data_df.columns:
        data_df[column] = data_df[column] / data_df[column].abs().max()
    data = pd.concat([speciesCol, data_df], axis=1)
    featuresX=data_df
    print("the whole data",data)
    ############# division data

    return featuresX , speciesCol

def bias_choice(bias_value):
    if bias_value == 1:
        bias = np.random.random_sample()
    else:
        bias = 0
    return bias

def split_data(features , species , bias):
    y = species
    x = features
    ############ bias step
    bias_values = bias_choice(bias)
    x.insert(0, "bias", bias_values)
    ############ bias step
    ############ train test split
    x_train,x_test,y_train,y_test =train_test_split(x,y ,test_size=0.4,shuffle=True)
    ############ train test split
    return x_train, y_train, x_test, y_test

def activationFunction(net,type):
    sigmoid = (type == 1)
    tanh = (type == 2)
    ############# forward sigmoid
    if sigmoid:
        forward_activation = 1 / (1 + np.exp(-1 * net))
    ############ forward tanh
    elif tanh:
        forward_activation = (1 - np.exp(-1 * net)) / (1 + np.exp(-1 * net))
    return forward_activation

def BackActivationFunction(net, function):
    sigmoid = (function == 1)
    tanh = (function == 2)
    ########## back sigmoid
    if sigmoid:
        back_activation = net * (1 - net)
    ######### back tanh
    elif tanh:
        back_activation = (1 - net) * (1 + net)
    return back_activation

def create_array(m, n):
    weights_array = []
    ######### number of neurons
    i=0
    while(i< m):
        col_of_weight = []
        j=0
        ######### number of features
        while(j < n):
            # weights in range -1 , 1
            col_of_weight.append(np.random.uniform(-1, 1))
            # increasing counter
            j+=1
        weights_array.append(col_of_weight)
        # increasing counter
        i+=1
    return weights_array

def weights(layers_num , neurons_num):
        # define index
        ind = 0
        # define number of neurons
        m = neurons_num[ind]
        # define features
        n = 5 + 1
        # define weights
        weights = []
        # define index
        i=0
        ############## number of layers
        while(i < (layers_num + 1)):
            ########### looping for neurons and features
            arr = create_array(m, n)
            ########## weight array
            weights.append(arr)
            condition_layer = (i == layers_num)
            if condition_layer:
                break
            # increasing counters
            ind = ind + 1
            # increasing counters
            m = neurons_num[ind]
            # increasing counters
            n = neurons_num[ind - 1] + 1
            # increasing counters
            i+=1
        return weights

def forward_propgation(NumberOfLayers,row,weights,bias,typeAF):
    arrayOfnet = []
    bias_values = bias_choice(bias)
    i = 0
    while(i < (NumberOfLayers+1)):
        # weight converted array
        weight = np.array(weights[i])
        # net value
        net = np.dot(weight,row)
        # activation function result
        result = activationFunction(net, typeAF)
        # array of nets
        arrayOfnet.append(result)
        # increase row
        row = np.insert(result, 0, bias_values,axis=0)
        # counter increasing
        i += 1
    return arrayOfnet

def back(arrayOfnetarrays, ActivFunction, weights, TrainY, n_Layers):
    # define counter
    counter = n_Layers
    # define arrays
    segmArray = []
    new_segma = []
    # loop for layers
    while (True):  # for layers
        # condition index
        if (counter > -1):
            net = arrayOfnetarrays[counter]
            # define number of nets
            n_neorns = len(net)
            # sigma array condition
            condition1 = (segmArray != [])
            # for output layer
            if condition1:
                # output layer
                old_segma = segmArray[len(segmArray) - 1]
            #loop for neurons
            i=0
            while(i < n_neorns):  # for nurons
                # reult of back activation
                result = BackActivationFunction(arrayOfnetarrays[counter][i], ActivFunction)
                # condition layers
                condition2= (counter == n_Layers)
                if (condition2):
                    # calculater error
                    error = TrainY[i] - arrayOfnetarrays[counter][i]
                    # calculate sigma
                    segma_net = (error) * result
                else:
                    j=0
                    len_old_sigma=len(old_segma)
                    # loop old sigma
                    while( j < len_old_sigma):  # for sum old sigma
                        # calculate value of sum
                        x = old_segma[j] * weights[counter + 1][j][i + 1]
                        # making sum
                        segma_net += x
                        j+=1
                    segma_net *= result
                i += 1
                new_segma.append(segma_net)
                segma_net = 0
            # decrease counter
            counter -= 1
            # sigma array
            segmArray.append(new_segma)
            # make new sigma
            new_segma = []
        else:
            break
    # return sigma array
    return segmArray


def updatewights(segmArray, arrayOfnetarrays, weights, eta, bias, row):
    weight_len=len(weights)
    i=0
    # looping in weight length
    while(i <weight_len):  # layers
        j=0
        w1=len(weights[i])
        while(j < w1):  # neron index
            # weight index eli daa5el flneron index
            k=0
            w2=len(weights[i][j])
            while(k < w2):
                x = x_cases(i, k, row, bias, arrayOfnetarrays)
                # sigma
                s = segmArray[i][j]
                # weight update
                weights[i][j][k] = weights[i][j][k] + (eta * s * x)
                # increase counter
                k+=1
            # increase counter
            j+=1
        # increase counter
        i+=1
    return weights


def x_cases(i, k, row, bias, arrayOfnetarrays):
    # declare boolean variables
    bias_true=(bias==1)
    bias_false=(bias==0)
    k_zero=(k==0)
    # declare boolean variables
    if i == 0:
        case_x = row[k]  # x=input
    elif (bias_true and k_zero):
        case_x = 1
    elif (bias_false == 0 and k_zero):
        case_x = 0
    else:
        # i-1 3shan elnet bybtdi mn awl layer b3d elinput, fah index ellayer hyb2a 1 eli hwa i bs elindex flnet 0
        case_x = arrayOfnetarrays[i - 1][k - 1]

    return case_x

def equality(i,actual,cm):
    # class1
    if (actual[i] == [1, 0, 0]):
        cm[0][0] += 1
    # class2
    if (actual[i] == [0, 1, 0]):
        cm[1][1] += 1
    #class3
    if (actual[i] == [0, 0, 1]):
        cm[2][2] += 1

def non_equality(i,actual,predicted,cm):
    # declare error classes
    class_1_2=(actual[i] == [1, 0, 0] and predicted[i] == [0, 1, 0])
    class_1_3=(actual[i] == [1, 0, 0] and predicted[i] == [0, 0, 1])
    class_2_1=(actual[i] == [0, 1, 0] and predicted[i] == [1, 0, 0])
    class_2_3=(actual[i] == [0, 1, 0] and predicted[i] == [0, 0, 1])
    class_3_1=(actual[i] == [0, 0, 1] and predicted[i] == [1, 0, 0])
    class_3_2=(actual[i] == [0, 0, 1] and predicted[i] == [0, 1, 0])
    # declare error classes
    # error value in class2
    if class_1_2:
        cm[0][1] += 1
    # error value in class3
    elif class_1_3:
        cm[0][2] += 1
    # error value in class1
    elif class_2_1:
        cm[1][0] += 1
    # error value in class3
    elif class_2_3:
        cm[1][2] += 1
    # error value in class1
    elif class_3_1:
        cm[2][0] += 1
    # error value in class2
    elif class_3_2:
        cm[2][1] += 1

def CreateConf(actual,predicted):
    cm = [[0] * 3 for _ in range(3)]
    for i in range(len(actual)):
        equal=(actual[i] == predicted[i])
        not_equal=(actual[i] != predicted[i])
        # actual value equal predicted value
        if equal:
            equality(i,actual,cm)
        # actual value not equal predicted value
        if not_equal:
            non_equality(i,actual,predicted,cm)
    return cm


def get_data():
    ##################### learning rate
    print(learningRateTB_entry.get())
    learning_rate = float(learningRateTB_entry.get())
    print("Learning rate:", learning_rate)
    ##################### learning rate

    ##################### epochs
    print(epotxt_entry.get())
    epochs = int(epotxt_entry.get())
    print("Epochs:", epochs)
    ##################### epochs

    #########################bias
    print(biasValue.get())
    if biasValue.get() == 'yes':
        bias = 1
    elif biasValue.get() == 'no':
        bias = 0
    print("Bias:", bias)
    #########################bias

    ##################### function
    print(functionTB_entry.get())
    function = functionTB_entry.get()
    print("Function:", function)
    if function == 'Sigmoid':
        fun_type = 1
    elif function == 'Tanh':
        fun_type = 2
    ##################### function

    ##################### layers
    print(layersTB_entry.get())
    layers = int(layersTB_entry.get())
    print("Layers:", layers)
    ##################### layers

    ##################### neuron
    neuron_num =[]
    neurons = neuronTB_entry.get()
    print(neuronTB_entry.get())
    for i in neurons.split(','):
        neuron_num.append(int(i))
    neuron_num.append(3)
    #neurons = int(neurons)
    print("Neuron:", neuron_num)#2,2
    ##################### neuron

    ##########calling
    features , species = read_file()
    # split data
    xTrain, yTrain, xTest,yTest = split_data(features , species,bias)
    # convert from list to array
    xTrain = np.array(xTrain)
    # convert from list to array
    yTrain = np.array(yTrain)
    # convert from list to array
    xTest = np.array(xTest)
    # convert from list to array

    yTest = np.array(yTest)
    correct = 0
    predArr = []
    ######## train
    # original weight values
    weight = weights(layers,neuron_num)
    # epochs array
    epoch=0
    while(epoch <epochs):
        # features array
        for i, row in enumerate(xTrain):
            # forward net values
            forward_net = forward_propgation(layers, row, weight, bias, fun_type)
            # backword fix
            back_array=back(forward_net, fun_type, weight,yTrain[i], layers)
            back_array = back_array[::-1]
            # update good weights
            result_updated_w = updatewights(back_array, forward_net, weight, learning_rate, bias, row)
            # increase counter
            epoch+=1
    ####### train

    ####### test
    for i, row in enumerate(xTest):
        # forward net values
        arrayOfnetarrays = forward_propgation(
            layers, row, weight, bias, fun_type)
        # get the class
        result = np.where(arrayOfnetarrays[layers] == np.amax(
            arrayOfnetarrays[layers]))
        # put chosen class 1
        arrayOfnetarrays[layers][result] = 1
        # put zeros to another classes
        newArr = not_exist_class(arrayOfnetarrays[layers])
        # calculate correct reslt
        if np.array_equal(newArr, yTest[i]):
            correct += 1
        # predicted array
        predArr.append(newArr.tolist())
    ####### test
    # calculate accuracy
    accuracy = (correct / len(xTest)) * 100
    # convert array to list
    testY = yTest.tolist()
    # print accuracy
    print("accuracy:" , accuracy)
    matrix = CreateConf(testY,predArr)
    print("------matrix of confusion-------")
    for i in range(len(matrix)):
        print("\t",matrix[i])
    ##########calling


def not_exist_class(array_net_predict):
    i = 0
    # len predicted array
    len_arr = len(array_net_predict)
    while( i < len_arr):
        # if the class not exist
        condition = array_net_predict[i] != 1
        if (condition):
            # not exist
            array_net_predict[i] = 0
        i+=1
    # return value
    return array_net_predict

if __name__ == "__main__":
    #######GUI
    ########### master window
    master = Tk()
    master.geometry('700x200')  # Size of the window
    master.title("Task1 Form")
    ########### master window

    ########################## layers
    # ----------layers label----------
    layersLB = Label(master, text="Layers :", font=('Times New Roman', '12'))
    layersLB.place(x=10, y=15)
    # ----------layers input----------
    layersTB = StringVar()
    layersTB_entry = Entry(master, textvariable=layersTB, width="20")
    layersTB_entry.place(x=100, y=20)
    ########################## layers

    ########################## neuron
    # ----------neuron label----------
    neuronLB = Label(master, text="Neuron :", font=('Times New Roman', '12'))
    neuronLB.place(x=250, y=15)
    # ----------neuron input----------
    neuronTB = StringVar()
    neuronTB_entry = Entry(master, textvariable=neuronTB, width="20")
    neuronTB_entry.place(x=340, y=20)
    ########################## neuron

    ########################## function
    # ----------function label----------
    functionLB = Label(master, text="Function :", font=('Times New Roman', '12'))
    functionLB.place(x=500, y=15)
    # ----------function input----------
    functionTB = tk.StringVar()

    functionTB_entry = ttk.Combobox(master, textvariable=functionTB, width="15")
    functionTB_entry['values'] = ('Sigmoid' , 'Tanh')
    functionTB_entry.current(0)
    functionTB_entry.place(x=570, y=20)
    ########################## function

    ########################## learning rate
    # ----------learning rate label----------
    learningRateLB = Label(master, text="learning rate :", font=('Times New Roman', '12'))
    learningRateLB.place(x=10, y=70)
    # ----------learning rate input----------
    learningRateTB = StringVar()
    learningRateTB_entry = Entry(master, textvariable=learningRateTB, width="20")
    learningRateTB_entry.place(x=100, y=75)
    ########################## learning rate

    ########################## epochs
    # ----------number of epochs label----------
    epo = Label(master, text="Epochs num :", font=('Times New Roman', '12'))
    epo.place(x=250, y=70)
    # ----------number of epochs input---------
    epotxt = StringVar()
    epotxt_entry = Entry(master, textvariable=epotxt, width="20")
    epotxt_entry.place(x=340, y=75)
    ########################## epochs

    ########################## BIAS
    # -----------bias label-------------
    bias = Label(master, text="bias :", font=('Times New Roman', '12'))
    bias.place(x=500, y=70)
    # -----------bias input-------------
    bias_arr = ('yes', 'no')
    biasValue = StringVar()
    biasDropMenu = OptionMenu(master, biasValue, *bias_arr).place(x=550, y=70)
    ########################## BIAS
    # ----------Submit button----------
    Submit = Button(master, text="Submit", width="20", command=get_data)
    Submit.place(x=260, y=135)

    master.mainloop()

