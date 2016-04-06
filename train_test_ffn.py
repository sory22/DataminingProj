from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.structure import TanhLayer
import argparse
import csv
import numpy as np
import matplotlib.pylab  as pl

#take csv string for each dataset and convert it to float list and normalize the attributes
def training_normalization (fileName,new_min,new_max ):
    '''
    Input Parameters:fileName, min normalized value,max normalized value
    Output: multidimensional array containing normalized attributes          
    '''
    data = []
    attrib_array = []
    global normal_array,max_array,min_array,range_array
 
    with open(fileName,'rb') as f:
        reader=csv.reader(f)
        data = list(reader)

    #normalize dataset arrays,i = #rows, j = #columns
    #5 inputs 3 outputs
    for j in range(0,8):
        attrib_array = []
        #convert from string to float list and find min/max for each attribute
        #using len(data)-1 for closing price offset  
        for i in range(2,len(data)-1):
            attrib_array.insert(i-2,float(data[i][j+1]))
        
        orig_range = max(attrib_array) - min(attrib_array)
        max_array[j] = max(attrib_array);
        min_array[j] = min(attrib_array);
        range_array[j] = orig_range

        #normalize attribute array
        for i in range(len(attrib_array)):
            temp = (attrib_array[i] - min(attrib_array))/(orig_range)*(float(new_max)-float(new_min))+(float(new_min))
            normal_array[j].insert(i,round(temp,2))
#            print round(attrib_array[i],2),"\t",normal_array[j][i]

#need to normalize test inputs
#orig_range value can be different for both test and train data because of two seperate datasets 
#if the new data has the new max then should the entire training be redone? or should we take the data in training as the highest value 
def test_normalize(input_array,new_min,new_max):
    global normalized_input
    global range_array
    global min_array
    global max_array

#    print len(input_array)
    #normalize attribute array
    for i in range(len(input_array)):
        # this is done for the case when the test data has min/max beyond min/max for data seen in training 
        if (input_array[i]>max_array[i]):
            input_array[i] = max_array[i]
        elif(input_array[i]<min_array[i]):
            input_array[i] = min_array[i]
        temp = (input_array[i] - min_array[i])/(range_array[i])*(float(new_max)-float(new_min))+(float(new_min))
        normalized_input.insert(i,round(temp,2))

def test_denormalize(input_val,new_min,new_max):
    global denorm_output
    global range_array
    global min_array
#    print len(input_val)
    for i in range(len(input_val)):
        denorm_output =  (input_val[i] - float(new_min))/(float(new_max)-float(new_min))*(range_array[i+5])+(min_array[i+5])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock Price Prediction')
    parser.add_argument('-f1', type=str,help="Location of file with training data.", required=True)
    parser.add_argument('-f2', type=str,help="Location of file with test data.",required=True)
    parser.add_argument("-min", type=str, help="Specifiy min normalization value",required=True)
    parser.add_argument("-max", type=str, help="Specifiy max normalization value", required=True)

#fixme change this to dynamic array and update using insert function
    normal_array           = [[],[],[],[],[],[],[],[]]
    normalized_input  = [[],[],[],[],[],[],[],[]]
    max_array                 = [[],[],[],[],[],[],[],[]]
    min_array                  = [[],[],[],[],[],[],[],[]]   
    range_array              = [[],[],[],[],[],[],[],[]]
    args = parser.parse_args()
#    denorm_output = [[],[],[]]
    denorm_output = []
#    prediction = [[],[],[]]
    prediction = []
   
    training_normalization(args.f1,args.min,args.max)
    #initialize dataset for neural network with 5 input + bias and 3 target 
    DS = SupervisedDataSet(5,1)

    #adding datasets to the network
    for i in range (0,len(normal_array[0])):
#       DS.addSample([normal_array[0][i],normal_array[1][i],normal_array[2][i],normal_array[3][i],normal_array[4][i]],[normal_array[5][i],normal_array[6][i],normal_array[7][i]])
       DS.addSample([normal_array[0][i],normal_array[1][i],normal_array[2][i],normal_array[3][i],normal_array[4][i]],[normal_array[5][i]])

#    NN = buildNetwork(5,4,3,bias =True,hiddenclass=TanhLayer)
    NN = buildNetwork(DS.indim,5,DS.outdim,bias = True,hiddenclass=TanhLayer)
    TRAINER = BackpropTrainer(NN,dataset=DS,learningrate = 0.01,momentum = 0.99)

    print 'MSE before',TRAINER.testOnData(DS)
    TRAINER.trainOnDataset(DS,500)
    print 'MSE after',TRAINER.testOnData(DS)

# testing 
#clearing arrays
    normal_array           = [[],[],[],[],[],[],[],[]]
    normalized_input  = [[],[],[],[],[]]
    max_array                 = [[],[],[],[],[],[],[],[]]
    min_array                  = [[],[],[],[],[],[],[],[]]   
    range_array              = [[],[],[],[],[],[],[],[]]
    x_axis                          = []
    pred_arr           = []
    act_arr                  = []
    training_normalization(args.f2,args.min,args.max)
    for i in range (len(normal_array[0])):
        prediction= NN.activate([normal_array[0][i],normal_array[1][i],normal_array[2][i],normal_array[3][i],normal_array[4][i]])
        test_denormalize(prediction,args.min,args.max)
        prediction = round(denorm_output,2)
        test_denormalize([normal_array[5][i],normal_array[6][i],normal_array[7][i]] ,args.min,args.max)     
        error = ((denorm_output-prediction)/denorm_output  )*100 
        x_axis.insert(i,i)
        pred_arr.insert(i,prediction)
        act_arr.insert(i,denorm_output)
       # print 'predicted, actual and error', prediction, denorm_output, error
       
    plot1 = pl.plot(x_axis,pred_arr,'r',label = 'Predicted')
    plot2 = pl.plot(x_axis,act_arr,'g',label = 'Actual')
    pl.xlabel('time(days)')
    pl.ylabel('price')
    pl.title('Predicted vs Actual INTC Price')
    legend = pl.legend(loc='upper left', shadow=True)     
    pl.show()   
     

