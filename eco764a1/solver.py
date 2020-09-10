import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random as rand
from datetime import datetime
import sys


eps = 1
r_f = 0.5
if(len(sys.argv)!=2 or (sys.argv[1] == "help")):
	# Inp format
	print("python3 solver.py help : For help")
	print("python3 solver.py plot : Generate plots")
	print("python3 solver.py appendix : Generate tables")
	exit(1)

def checkifnan(num):
	if num is np.nan:
		return 0
	else:
		return num

def calculateWeightedParam(companies,listofcompanies,x):
	totalsum = 0.0
	capsum = 0.0
	for company_name in listofcompanies:
		totalsum += companies[company_name]['CAP'][x]*companies[company_name]['PIreturn'][x]
		capsum += companies[company_name]['CAP'][x]
	if capsum == 0:
		return 0
	return (totalsum/(capsum))
	


df = pd.read_excel('CE_Europe.xlsx')
columns = list(df.loc[2])[1:]
row_count = df.shape[0] - 5
companies = dict()
j = 0
company_name = ""
prev_err = False

date=[]
l=df.iloc[4:(4+row_count),0].tolist()
for x in range(row_count):
	date.append(str(l[x]).split()[0])


for i,col in enumerate(columns):
	idx = i + 1
	if(col == "#ERROR"):
		prev_err = True
		j = (j+1)%3
		continue
	elif(j == 0):
		company_name = col.split('-')[0].strip()
		companies[company_name] = dict()
		# CAP
		a = np.nan_to_num(np.array(df.iloc[4:(4+row_count),idx]).astype(float))
		companies[company_name]['CAP'] = a
		
	elif(j == 1):
		# ROI
		if(prev_err == True):
			company_name = col.split('-')[0].strip()
			if(company_name not in companies):
				companies[company_name] = dict()
		a = np.nan_to_num(np.array(df.iloc[4:(4+row_count),idx]).astype(float))
		companies[company_name]['ROI'] = a
	elif(j == 2):
		# PI
		if(prev_err == True):
			company_name = col.split('-')[0].strip()
			if(company_name not in companies):
				companies[company_name] = dict()
		a = np.nan_to_num(np.array(df.iloc[4:(4+row_count),idx]).astype(float))
		companies[company_name]['PI'] = a

	j = (j+1)%3
	prev_err = False
for company_name in companies:
	if 'PI' not in companies[company_name]:
		companies[company_name]['PI'] = np.array([0.]*row_count)
	if 'ROI' not in companies[company_name]:
		companies[company_name]['ROI'] = np.array([0.]*row_count)
	if 'CAP' not in companies[company_name]:
		# print(company_name)
		companies[company_name]['CAP'] = np.array([0.]*row_count)

marketcaparr = []
priceindexarr = []
growtharr = []
booktomarketarr = []
companylist =[]
smblist = []
hmllist = []
flist=[]
momlist=[]
t=[]
y_r=[]

for company_name  in companies:
	companies[company_name]['YR'] = []
	for x in range(row_count):
		companies[company_name]['YR'].append(0.0)

for x in range(0,row_count,12):
	for company_name in companies:
		# companies[company_name]['YR'] = []
		# for  iters in range(0,row_count):
		# 	companies[company_name]['YR'].append(0.0)
		for y in range(12):
			if y == 0:
				if x == 0:
					if companies[company_name]['PI'][x] > 0:
						companies[company_name]['YR'][x+y] = 100.00*(companies[company_name]['PI'][x+11]-companies[company_name]['PI'][x])/(companies[company_name]['PI'][x])
				elif x > 0:
					if companies[company_name]['PI'][x-11] > 0:
						companies[company_name]['YR'][x+y] = 100.00*(companies[company_name]['PI'][x]-companies[company_name]['PI'][x-11])/(companies[company_name]['PI'][x-11])
			elif y > 0:
				if x == 0:
					if companies[company_name]['PI'][x] > 0:
						companies[company_name]['YR'][x+y] = 100.00*(companies[company_name]['PI'][x+11]-companies[company_name]['PI'][x])/(companies[company_name]['PI'][x])
				elif x > 0:
					if companies[company_name]['PI'][x-11] > 0:
						companies[company_name]['YR'][x+y] = 100.00*(companies[company_name]['PI'][x]-companies[company_name]['PI'][x-11])/(companies[company_name]['PI'][x-11])



for company_name in companies:
	companylist.append(company_name)

three_factors = []
four_factors = []
new_mom = []
capm=[]
# print(row_count)

for x in range(row_count):
	marketcap = []
	priceindex = []
	growth = []
	booktomarket = []
	pireturnarr = []
	new_mom_arr = []

	for company_name in companies:
		if(x == 0):
			companies[company_name]['PIreturn'] = []
			companies[company_name]['PIreturn'].append(0.0)

		marketcap.append(companies[company_name]['CAP'][x])
		growth.append(companies[company_name]['ROI'][x])
		if x > 0:
			if companies[company_name]['PI'][x-1] ==0:
				pireturn=0
			else:
				pireturn = 100.00*(companies[company_name]['PI'][x]-companies[company_name]['PI'][x-1])/(companies[company_name]['PI'][x-1])
			companies[company_name]['PIreturn'].append(pireturn)
		pireturnarr.append(companies[company_name]['PIreturn'][x])
		booktomarket.append(companies[company_name]['ROI'][x])
		new_mom_arr.append(companies[company_name]['YR'][x])
	

	high = np.percentile(booktomarket,70)
	low = np.percentile(booktomarket,30)
	small = np.percentile(marketcap,30)
	big = np.percentile(marketcap,70)

	win = np.percentile(new_mom_arr,70)
	lose = np.percentile(new_mom_arr,30)

	valuefirm = []
	growthfirm = []
	neutralfirm = []
	smallfirm = []
	bigfirm = []
	winnerfirm = []
	loserfirm = []

	for y in range(len(companylist)):
		if marketcap[y] > big:
			bigfirm.append(companylist[y])
		elif marketcap[y] < small:
			smallfirm.append(companylist[y])
		if booktomarket[y] < low:
			growthfirm.append(companylist[y])
		elif booktomarket[y] > low and booktomarket[y] < high:
			neutralfirm.append(companylist[y])
		elif booktomarket[y] > high:
			valuefirm.append(companylist[y])
		if new_mom_arr[y] > win:
			winnerfirm.append(companylist[y])
		elif new_mom_arr[y] < lose:
			loserfirm.append(companylist[y])


	BV = list(set(bigfirm) & set(valuefirm))
	BN = list(set(bigfirm) & set(neutralfirm))
	BG = list(set(bigfirm) & set(growthfirm))
	SV = list(set(smallfirm) & set(valuefirm))
	SN = list(set(smallfirm) & set(neutralfirm))
	SG = list(set(smallfirm) & set(growthfirm))
	WB = list(set(winnerfirm) & set(bigfirm))
	WS = list(set(winnerfirm) & set(smallfirm))
	LB = list(set(loserfirm) & set(bigfirm))
	LS = list(set(loserfirm) & set(smallfirm))

	if len(BV)> 0:
		bvreturn = calculateWeightedParam(companies,BV,x)
	else:
		bvreturn = 0

	if len(BN)> 0:
		bnreturn = calculateWeightedParam(companies,BN,x)
	else:
		bnreturn = 0

	if len(BG)> 0:
		bgreturn = calculateWeightedParam(companies,BG,x)
	else:
		bgreturn = 0


	if len(SV)> 0:
		svreturn = calculateWeightedParam(companies,SV,x)
	else:
		svreturn = 0

	if len(SN)> 0:
		snreturn = calculateWeightedParam(companies,SN,x)
	else:
		snreturn = 0

	if len(SG)> 0:
		sgreturn = calculateWeightedParam(companies,SG,x)
	else:
		sgreturn = 0

	if len(WB)>0:
		wbreturn = calculateWeightedParam(companies,WB,x)
	else:
		wbreturn = 0

	if len(WS)>0:
		wsreturn = calculateWeightedParam(companies,WS,x)
	else:
		wsreturn = 0

	if len(LB)>0:
		lbreturn = calculateWeightedParam(companies,LB,x)
	else:
		lbreturn = 0

	if len(LS)>0:
		lsreturn = calculateWeightedParam(companies,LS,x)
	else:
		lsreturn = 0

	smb = (svreturn+snreturn+sgreturn)/3 - (bvreturn+bnreturn+bgreturn)/3 
	smblist.append(smb)

	hml = (svreturn+bvreturn)/2 - (sgreturn+bgreturn)/2
	hmllist.append(hml)

	f = calculateWeightedParam(companies,companylist,x)
	flist.append(f)

	mom = (wsreturn+wbreturn)/2 - (lsreturn+lbreturn)/2
	momlist.append(mom)

	three_factors.append([f-r_f,hml,smb])
	capm.append([f-r_f])
	four_factors.append([f-r_f,hml,smb,mom])

	t.append(x)


C = np.array(capm, np.float32)
Z = np.array(four_factors, np.float32)
X = np.array(three_factors, np.float32)

smblist[0]=1
for i in range(1,len(smblist)):
	smblist[i]=smblist[i]/100+smblist[i-1]

hmllist[0]=1
for i in range(1,len(hmllist)):
	hmllist[i]=hmllist[i]/100+hmllist[i-1]

flist[0]=1
for i in range(1,len(flist)):
	flist[i]=flist[i]/100+flist[i-1]

momlist[0]=1
for i in range(1,len(momlist)):
	momlist[i]=momlist[i]/100+momlist[i-1]

for i in range(len(date)):
	date[i] = datetime.strptime(date[i], '%Y-%m-%d')
#print(date)
if(sys.argv[1] == "plot"):
	plt.plot(date,smblist,label='SMB')
	plt.plot(date,hmllist,label='HML')
	plt.plot(date,flist,label='F')
	plt.title('SMB+HML+F vs date')
	plt.xlabel('Date')
	plt.ylabel('Commulative return')
	plt.locator_params(axis='x', nbins=10)
	plt.legend(loc=2)
	plt.show()

if(sys.argv[1] == "appendix"):
	file_w = open("FF3factor.txt", "w")
	for company_name in companies:
		y = np.array(companies[company_name]['PIreturn'])
		y = y - r_f
		# print(y)
		model = LinearRegression().fit(X, y)
		r_sq = model.score(X, y)
		file_w.write(company_name+" & "+str(round(model.intercept_,3))+" & "+str(round(model.coef_[0],3))+" & "+str(round(model.coef_[1],3))+" & "+str(round(model.coef_[2],3))+" \\\\ \n")

	print("Latex Table stored for FamaFench 3 factor model in FF3factor.txt ...")
	file_w.close()
	file_w = open("C4factor.txt", "w")
	for company_name in companies:
		y = np.array(companies[company_name]['PIreturn'])
		y = y - r_f
		model = LinearRegression().fit(Z, y)
		r_sq = model.score(Z, y)
		file_w.write(company_name+" & "+str(round(model.intercept_,3))+" & "+str(round(model.coef_[0],3))+" & "+str(round(model.coef_[1],3))+" & "+str(round(model.coef_[2],3))+" & "+str(round(model.coef_[3],3))+" \\\\ \n")

	print("Latex Table stored for Carhart 4 factor model in C4factor.txt ...")
	file_w.close()
	file_w = open("CAPM1factor.txt", "w")
	for company_name in companies:
		y = np.array(companies[company_name]['PIreturn'])
		y = y - r_f
		# print(y)
		model = LinearRegression().fit(C, y)
		r_sq = model.score(C, y)
		file_w.write(company_name+" & "+str(round(model.intercept_,3))+" & "+str(round(model.coef_[0],3))+" \\\\ \n")
	print("Latex Table stored for CAPM 1 factor model in CAPM1factor.txt ...")
	file_w.close()
