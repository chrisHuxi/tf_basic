# -*- coding: UTF-8 -*-

from numpy import *
import csv
from copy import *
import time

#======================================Part1.����׼��==================================#
####��csv�ļ��е�������ȡ��list�У��ֱ����Label��1ά����Feature_Vector��2ά����####
def perpare_data(): 
	csvfile = file('train.csv', 'rb')
	Feature_Vectors = []
	Labels = []
	i = 1
	reader = csv.reader(csvfile)
	for line in reader:
		if i == 1:
			i = i + 1
			continue
		Labels.extend(line[0])
		Feature_Number_List = ConvertPixls2Float(List_StringTONumber(line[1:len(line)]))
		Feature_Vectors.append(Feature_Number_List)
		#Feature_Vectors.append(line[1:len(line)])
	Labels =List_StringTONumber(Labels)
	csvfile.close()
	return Feature_Vectors,Labels


####���ַ��б�lists=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']ת��Ϊ�����б�####
def List_StringTONumber(lists):  	
	lists= map(int, lists)
	temp_lists = []
	for i in lists:
		temp_lists.append(i)
	return temp_lists
	
####��0~255��pixlsֵת��-1.0~1.0��float������	
def ConvertPixls2Float(lists):
	FloatNumber_list = []
	for everyeum in lists:
		FloatNumber = (everyeum/255.0)*2 - 1.0
		FloatNumber_list.append(FloatNumber)
	return FloatNumber_list
#========================================================================================#	




#======================================Part2.����������==================================#
####���������粢��ɳ�ʼ��####
def BuildingNN(InputLayer_List,OutputLayer_List):
	InputLayer = matrix(InputLayer_List)	#42000*784
	OutputLayer = Num2Vector(OutputLayer_List)	#42000*10
	NumInput = size(InputLayer[0])	#�����������ÿ��ͼƬ���������� 784
	NumHidden = 25 #25	#25���м�ڵ㣬���Ը�
	NumOutput = 10	#��0��9ʮ��������10ά������ʾ
	alpha = 0.2		#�����޸ĵ�ϵ������������Ҫʹ����JС��1.5����ѡ��3���µ���
	InitialTheta_ItoH = matrix(random.random((NumHidden,NumInput+1)))	# 25 * 784+1
	InitialTheta_HtoO = matrix(random.random((NumOutput,NumHidden+1)))	# 10 * 25+1
	J,Theta_ItoH_grad,Theta_HtoO_grad = BackPropagate(InputLayer_List,OutputLayer,InitialTheta_ItoH,InitialTheta_HtoO)
	#��һ�����Theta_ItoH_grad,Theta_HtoO_grad�ĺ���
	TempTheta_ItoH = deepcopy(InitialTheta_ItoH)# 25 * 784+1
	TempTheta_HtoO = deepcopy(InitialTheta_HtoO)# 10 * 25+1
	result_J = deepcopy(J)
	result_ItoH_grad = deepcopy(Theta_ItoH_grad)# 25 * 784+1
	result_HtoO_grad = deepcopy(Theta_HtoO_grad)# 10 * 25+1
	i = 0
	#result_J_before = deepcopy(result_J)
	while (1):
		i = i+1
		result_J_before = deepcopy(result_J)
		ResultTheta_ItoH = TempTheta_ItoH - alpha*result_ItoH_grad	#
		ResultTheta_HtoO = TempTheta_HtoO - alpha*result_HtoO_grad
		result_J,result_ItoH_grad,result_HtoO_grad = BackPropagate(InputLayer_List,OutputLayer,ResultTheta_ItoH,ResultTheta_HtoO)
		TempTheta_ItoH = deepcopy(ResultTheta_ItoH)
		TempTheta_HtoO = deepcopy(ResultTheta_HtoO)
		
		print "cost:%f ,%d th"%(result_J,i)
		#����������J<1.0֮����ܱ�֤�������ȷ��
		if(i>=5000 or result_J <= 0.8):
			break
	#--------------------------------���ϼ������Theta_ItoH,Theta_HtoO---------------------------#
	return ResultTheta_ItoH,ResultTheta_HtoO		#25*784+1 	10*25+1
	
	
####���򴫲�####
def BackPropagate(InputLayer_List,OutputLayer,Theta_ItoH_Mat,Theta_HtoO_Mat):	#42000*784 42000*10
	m = size(InputLayer_List,0)	#42000
	#print m
	Lambda = 3			#Lambda �ͷ�ϵ�� �����޸�
	a1 = AddBiasNode(InputLayer_List)	#42000*784+1
	#print a1
	z2 = a1*Theta_ItoH_Mat.T 	#42000*784+1	 784+1*25
	#print z2
	a2 = AddBiasNode(Sigmoid(z2))	#42000*25+1
	#print Sigmoid(z2)
	#print a2
	z3 = a2*Theta_HtoO_Mat.T	#42000*25+1 25+1*10
	#print z3
	a3 = Sigmoid(z3)	#42000*10
	#print a3
	#-------����ȫ��������------#
	
	#---------------------------------------------------------------------------------------------------------
	#-----------------����grad��-------------------#
	delta3 = a3 - OutputLayer	#�������������ƫ�� #42000*10
	d3_temp1 = delta3*Theta_HtoO_Mat[:,1:size(Theta_HtoO_Mat,1)] 	#42000*10 10*25
	d3_temp2 = SigmoidGradient(z2)		#42000*25
	delta2 = multiply(d3_temp1,d3_temp2)	#42000*10 10*26-1 
	Delta1 = delta2.T*a1	#42000*25 42000*784+1
	Delta2 = delta3.T*a2	#42000*10 42000*25+1
	unregulered_Theta1_grad = Delta1/(0.0 + m)	#25*784+1
	unregulered_Theta2_grad = Delta2/(0.0 + m)	#10*25+1
	Theta1_grad = Regulerize_grad(unregulered_Theta1_grad,Theta_ItoH_Mat,m,Lambda)	#25*784+1
	Theta2_grad = Regulerize_grad(unregulered_Theta2_grad,Theta_HtoO_Mat,m,Lambda)	#10*25+1
	#-------------------------------------------------#
	#-------------------grad�������------------------#
	
	#--------------------J�������-------------------#
	#-----------------����J��-------------------#
	J_temp1 = multiply(OutputLayer,log(a3))*(-1.0)
	J_temp2_1 = CreateOnesMatrix(size(OutputLayer,0),size(OutputLayer,1)) - OutputLayer
	J_temp2_2 = log(CreateOnesMatrix(size(a3,0),size(a3,1)) - a3 + 0.0)
	#a = CreateOnesMatrix(size(a3,0),size(a3,1)) - a3 + 0.0
	#print a
	J_temp2 = multiply(J_temp2_1,J_temp2_2)
	unregulered_J = sum(J_temp1 - J_temp2)/m	#42000*10==>value
	J = Regulerize_J(unregulered_J,Theta_ItoH_Mat,Theta_HtoO_Mat,m,Lambda)
	#-------------------------------------------#

	return J,Theta1_grad,Theta2_grad
	#---------------------------------------------------------------------------------------------------------
	#������Ҫ�ص��飺�����ϣ����ؽ����ȷ
	

####��grad��ӹ����####
def Regulerize_grad(unregulered_Theta_grad,Theta,m,Lambda):
	i = 0
	#Lambda = 1
	temp_Theta = deepcopy(Theta)	#ע�������������ȸ�ֵ�������ı䴫��Ĳ���
	for everyrow in temp_Theta:
		temp_Theta[i,0] = 0
		i = i+1
	#print temp_Theta
	regulared_Theta_grad = unregulered_Theta_grad + Lambda*temp_Theta/(m+0.0)
	#�ں������0.0��Ϊ��ת�ɸ���������
	return regulared_Theta_grad
	
	
####��J����ӹ����####	
def Regulerize_J(unregulered_J,Theta1,Theta2,m,Lambda):
	Theta1_square = multiply(Theta1,Theta1)+0.0
	Theta2_square = multiply(Theta2,Theta2)+0.0
	#print m
	regulared_term = sum(Theta1_square[:,1:size(Theta1_square,1)]) + sum(Theta2_square[:,1:size(Theta2_square,1)])
	J = unregulered_J + regulared_term*Lambda/(2.0*m)
	return J
	
	
####���bias unit####
def AddBiasNode(InputLayer):
	Input_matrix = mat(deepcopy(InputLayer))
	i = 0
	row_num = size(Input_matrix,0)
	Bias_column = CreateOnesMatrix(row_num,1)
	Added_Matrix = column_stack((Bias_column, Input_matrix))
	
	#for everyrow in InputLayer:
		#Add_matrix[i] = column_stack((a, everyrow))
		#print "======="
		#print Add_matrix[i]
		#i = i+1
	#return matrix(Add_matrix)
	return Added_Matrix
	
####sigmoid����####	
def	Sigmoid(X):	
	return 1.0 / (1.0 + exp(-X))
	
	
####sigmoid�����ĵ���####	
def SigmoidGradient(Z):		 #Z�����Ƕ�ά��
	OnesMatrix = CreateOnesMatrix(size(Z,0),size(Z,1))
	temp_result = OnesMatrix - Sigmoid(Z) + 0.0
	result = multiply(Sigmoid(Z),temp_result)
	return result

	
####�������ַ�Labelת��Ϊһ��10ά��������0��9####
def	Num2Vector(OutputLayer_List):
	ZeroArray= array([0]*len(OutputLayer_List)*10)
	OutputLayer_Vec = ZeroArray.reshape(len(OutputLayer_List),10) #42000*10
	i = 0
	for items in OutputLayer_List:
		OutputLayer_Vec[i,items] = 1
		i = i+1
	#print i #�����Ƿ����е�label��ת����vector
	return OutputLayer_Vec	#42000*10
	
####����ȫ1����####	
def CreateOnesMatrix(row_num,column_num):
	OnesMatrix = array([1]*row_num*column_num).reshape(row_num,column_num)
	return OnesMatrix
#========================================================================================#	
	

	
	
	
#======================================Part3.��������������Ԥ��==================================#
def PredictFunction(test_Mat,Theta_ItoH,Theta_HtoO): 	#28000*784	25*784+1  10*25+1
	#print test_Mat
	#for everyeum in test_Mat[0]:
		#print everyeum
	#---------------------����Ӧ����������---------------------#
	test_InputLayer = AddBiasNode(test_Mat) #28000*785

	test_HiddenLayer_temp = Sigmoid((test_InputLayer*Theta_ItoH.T))
	#print (test_InputLayer*Theta_ItoH.T)
	#print "/////////////////////"
	#for everyeum in test_HiddenLayer_temp[0]:
		#print everyeum
		
	test_HiddenLayer = AddBiasNode(test_HiddenLayer_temp)
	test_OutputLayer = array(Sigmoid(test_HiddenLayer*Theta_HtoO.T))	#28000*10 
	print test_OutputLayer
	print "======"
	print size(test_InputLayer,0)
	print size(test_InputLayer,1)
	print "+"
	print size(test_HiddenLayer,0)
	print size(test_HiddenLayer,1)
	print "+"
	print size(test_OutputLayer,0)
	print size(test_OutputLayer,1)
	#array����ҵ����Ϊ������ļ�����mat�����޷���������ת��array
	#print test_OutputLayer
	test_Output_Num = []
	for everyrow in test_OutputLayer:
		flag = 0
		i = 0
		for everycolumn in everyrow:
			if everycolumn==max(everyrow):
				test_Output_Num.append(i)
				break
			i = i+1
	print size(test_OutputLayer)
	return test_Output_Num
#---------------test-------------#	
#========================================================================================#




	
#=================================Part4.С��ģtest���ݼ�����===============================#
def Test_SmallDataSet():		#ѡȡtrain��ǰ200��
	time1 = time.clock()
	All_Feature,All_Label = perpare_data()
	Train_Feature = All_Feature[0:200]
	#print size(Train_Feature,0)
	Test_Feature = All_Feature[201:231]
	
	Train_Label = All_Label[0:200]
	Test_Label = All_Label[201:231]
	Theta1,Theta2 = BuildingNN(Train_Feature,Train_Label)
	test_result = PredictFunction(mat(Test_Feature),Theta1,Theta2)
	index_test_result = 0
	right_num = 0
	for everynumber in test_result:
		if everynumber == Test_Label[index_test_result]:
			right_num = right_num + 1
		print "the perdict answer:%f  the right answer:%f"%(everynumber,Test_Label[index_test_result])
		index_test_result = index_test_result + 1
		
	Accuracy_Rate = right_num/(size(Test_Label)+0.0) * 100.0
	print "accuracy rate:%f %%"%(Accuracy_Rate)
	print test_result
	print Test_Label
	time2 = time.clock()
	UsingTime = time2 - time1
	print UsingTime
#========================================================================================#




#==============================Part5.��鷴�򴫲����صĽ���Ƿ���ȷ=============================#
def Check_Gradient():
	All_Feature,All_Label = perpare_data()
	Train_Feature = All_Feature[0:500]
	Train_Label = All_Label[0:500]
	
	InputLayer = matrix(Train_Feature)	#42000*784
	OutputLayer = Num2Vector(Train_Label)	#42000*10
	NumInput = size(InputLayer,1)	#�����������ÿ��ͼƬ���������� 784
	NumHidden = 25 #25	#25���м�ڵ㣬���Ը�
	NumOutput = 10	#��0��9ʮ��������10ά������ʾ
	alpha = 0.1	#�����޸ĵ�ϵ��
	InitialTheta_ItoH = matrix(random.random((NumHidden,NumInput+1)))	# 25 * 784+1
	InitialTheta_HtoO = matrix(random.random((NumOutput,NumHidden+1)))	# 10 * 25+1
	J_back,Theta1_grad_back,Theta2_grad_back = BackPropagate(Train_Feature,OutputLayer,InitialTheta_ItoH,InitialTheta_HtoO)
	
	#numgrad = CreateOnesMatrix(size(InitialTheta_ItoH)) - 1
	#perturb = CreateOnesMatrix(size(InitialTheta_ItoH)) - 1
	#print numgrad
	epsilon = 0.01
	
	#--------------------����û����-------------------#
	J_temp1,Theta1_grad_back_temp,Theta2_grad_back_temp = BackPropagate(Train_Feature,OutputLayer,InitialTheta_ItoH - epsilon,InitialTheta_HtoO)
	J_temp2,Theta1_grad_back_temp,Theta2_grad_back_temp = BackPropagate(Train_Feature,OutputLayer,InitialTheta_ItoH + epsilon,InitialTheta_HtoO)
	print (J_temp2 - J_temp1)
	numgrad_Theta1_grad = (J_temp2 - J_temp1)/(2.0*epsilon)
	#-------------------------------------------------------------------------------
	print sum(Theta1_grad_back)
	print "===================="
	print numgrad_Theta1_grad
	
	J_temp3,Theta1_grad_back_temp,Theta2_grad_back_temp = BackPropagate(Train_Feature,OutputLayer,InitialTheta_ItoH,InitialTheta_HtoO - epsilon)
	J_temp4,Theta1_grad_back_temp,Theta2_grad_back_temp = BackPropagate(Train_Feature,OutputLayer,InitialTheta_ItoH,InitialTheta_HtoO + epsilon)
	print (J_temp4 - J_temp3)
	numgrad_Theta2_grad = (J_temp4 - J_temp3)/(2.0*epsilon)
	#-------------------------------------------------------------------------------
	print sum(Theta2_grad_back)
	print "===================="
	print numgrad_Theta2_grad	
#========================================================================================#	

#==============================Part6.ʹ��PCAѹ������=============================#
'''
ע��������ɣ�����
'''
def PCA_algorithm(dataMat,k):	#42000*784
	'''�㷨���裺
		��������Ϊm*n�ľ���
		1.Ԥ��������ÿһ��������ֵ��ƽ��ֵ������feature scaling
		2.����Э�������(covariance matrix)
		3.����þ������������
		4.������ֵ�Ӵ�С����
		5.��������k������ֵ������Щ����ֵ��m*k����ԭ���ݣ�m*n����ת�óˣ��õ���ѹ�������ݣ�m*k��
		
		ע����Ng�Ŀγ���ʹ�õ���SVD�����ǣ���˵����PCAʵ������һ�����������ϲ�����ͳ��PCA�㷨
		'''
	meanOfDataMat_allRow = mean(dataMat,axis = 0)	#1*784 list
	scaledDataMat = dataMat - meanOfDataMat_allRow	#42000*784 		list
	covMatrix = cov(scaledDataMat,rowvar = 0)	#list
	eigenValues,eigenVector= linalg.eig(mat(covMatrix))	#��֮ǰ����list������ſ�ʼ��mat����
	eigenValuesIndex = argsort(-eigenValues)
	choosedEigenValuesIndex = eigenValuesIndex[0:k]
	choosedEigenVectors = eigenVector[:,choosedEigenValuesIndex]	#784*k
	kDimensionDataMat = scaledDataMat *choosedEigenVectors 	#42000*k
	reconstructedDataMat = kDimensionDataMat*choosedEigenVectors.T	# 42000*k 784*k.T
	#==================����ͨ��=====================
	return scaledDataMat,kDimensionDataMat,reconstructedDataMat
	
###ʹ�ö��ַ����ұȽϺðɣ���ͷ����###
'''������ѡ��K�ķ�����Ч��̫����'''
def ChooseK(dataMat):
	m = size(dataMat,1)
	for i in range(m-1,1,-1):
		pressedDataMat, reconstructedDataMat = PCA_algorithm(dataMat,i)
		pressedDataArray = array(pressedDataMat)
		reconstructedDataArray = array(reconstructedDataMat)
		differencePercent = sum((array(dataMat) - reconstructedDataArray)**2)/sum((array(dataMat))**2)
		print differencePercent
		if differencePercent > 0.05:	#����95%������
			break
	print pressedDataMat
	return i+1,pressedDataMat
	
###PCA�㷨����ͨ��###
'''��ʹ��2000��ѵ����������ѹ����ʱ�򣬽�784������ѹ����100����������7.9%������'''
def Test_PCA():
	All_Feature,All_Label = perpare_data()
	Train_Feature = All_Feature[0:2000]
	#print size(Train_Feature,0)
	Test_Feature = All_Feature[201:231]
	
	Train_Label = All_Label[0:2000]
	Test_Label = All_Label[201:231]
	dataMat = Train_Feature	#�����dataMat��������listҲ���ԣ�����
	
	k = 100
	#dataMat = mat([[1,2,4,6],[2,5,7,1],[4,9,1,3]])
	m = size(dataMat,0)
	print "======================"
	scaledDataMat,pressedDataMat, reconstructedDataMat = PCA_algorithm(dataMat ,k)
	pressedDataArray = array(pressedDataMat)
	reconstructedDataArray = array(reconstructedDataMat)
	differencePercent_1 = 0
	differencePercent_2 = 0
	for j in range(0,m):
		differencePercent_1 = differencePercent_1 + (linalg.norm(array(scaledDataMat[j]) - reconstructedDataArray[j]))**2
		differencePercent_2 = differencePercent_2 + (linalg.norm(array(scaledDataMat[j])))**2
		#print j
	#print differencePercent_1
	#print differencePercent_2
	differencePercent = differencePercent_1/differencePercent_2
	print j
	print differencePercent
	
	#print "the best dimension is: %f" %(k)
	print "======================"
#========================================================================================#	


#===============���ű�===============#
#if __name__ =='__main__':

#ȡ������2100��
All_Feature,All_Label = perpare_data()
UsedSample = All_Feature[0:5100]

k = 150		#0.04~0.05
m = size(UsedSample,0)
#ѹ������
scaledDataMat,kDimensionDataMat,reconstructedDataMat = PCA_algorithm(UsedSample,k)
print "======================"
pressedDataArray = array(kDimensionDataMat)
reconstructedDataArray = array(reconstructedDataMat)
differencePercent_1 = 0
differencePercent_2 = 0
for j in range(0,m):
	differencePercent_1 = differencePercent_1 + (linalg.norm(array(scaledDataMat[j]) - reconstructedDataArray[j]))**2
	differencePercent_2 = differencePercent_2 + (linalg.norm(array(scaledDataMat[j])))**2
differencePercent = differencePercent_1/differencePercent_2

print "the lost feature rate is: %f" %(differencePercent)
print "======================"


Train_Feature = kDimensionDataMat[0:5000]
Test_Feature = kDimensionDataMat[5000:5100]
Train_Label = All_Label[0:5000]
Test_Label = All_Label[5000:5100]

time1 = time.clock()
Theta1,Theta2 = BuildingNN(Train_Feature,Train_Label)
test_result = PredictFunction(mat(Test_Feature),Theta1,Theta2)
index_test_result = 0
right_num = 0

print size(Test_Label)
print size(test_result)

for everynumber in test_result:
	if everynumber == Test_Label[index_test_result]:
		right_num = right_num + 1
	print "the perdict answer:%f  the right answer:%f"%(everynumber,Test_Label[index_test_result])
	index_test_result = index_test_result + 1
	
Accuracy_Rate = right_num/(size(Test_Label)+0.0) * 100.0
print "accuracy rate:%f %%"%(Accuracy_Rate)
print test_result
print Test_Label
time2 = time.clock()
UsingTime = time2 - time1
print UsingTime
