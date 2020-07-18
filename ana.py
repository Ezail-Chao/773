

# Geant4 xml analysis


import xml.etree.ElementTree as ET



root = ET.parse('kabenasi5cm/detector_loc_000.xml').getroot()



for child in root.findall('histogram1d'):
    if child.attrib['name'] == 'h1:0':
        break
print(child.attrib['title'])
xml_axis = child.find('axis')
print(xml_axis.attrib)



# get bin 
for it in child.findall('data1d/bin1d'):
    print(it.attrib)


# Get 0.662 MeV bin


assert xml_axis.attrib['numberOfBins'] == '1'
assert float(xml_axis.attrib['min']) > 0.6
assert float(xml_axis.attrib['max']) < 0.7



count = 0
for it in child.findall('data1d/bin1d'):
    if it.attrib['binNum'] == '0':
        count = int(it.attrib['entries'])
        break
print('count:', count)


# Function to extract count


import xml.etree.ElementTree as ET
def get_count_xml(fname, target_mev=0.662, debug=False):
    # parse xml
    root = ET.parse(fname).getroot()
    for child in root.findall('histogram1d'):
        if child.attrib['name'] == 'h1:0':
            break
            
    xml_axis = child.find('axis')
    if debug:
        print(child.attrib['title'])
        print(xml_axis.attrib)
    
    # check target MeV
    assert xml_axis.attrib['numberOfBins'] == '1'
    assert float(xml_axis.attrib['min']) > (target_mev-0.1)
    assert float(xml_axis.attrib['max']) < (target_mev+0.1)
    
    # get count from bin
    count = 0
    for it in child.findall('data1d/bin1d'):
        if it.attrib['binNum'] == '0':
            count = int(it.attrib['entries'])
            break
    if debug:
        print('count:', count)
    return count



get_count_xml('kabenasi5cm/detector_loc_000.xml')



get_count_xml('kabenasi5cm/detector_loc_001.xml', debug=True)


# Data visualization


import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline



#width_len = 11
width_len = 11
data = []
for i in range(width_len*width_len):
    fname = f'kabenasi5cm/detector_loc_{i:03}.xml'
    if i%10==0:
        print(fname)
    cnt = get_count_xml(fname)
    data.append(cnt)



print(data)



viz = np.array(data).reshape(width_len,width_len)
viz = viz.astype(np.float)/viz.max()
plt.imshow(viz)


# Localization
## define variables


b_measure = np.array(data)
x_b = np.genfromtxt('kabenasi5cm/detector_position.txt', delimiter=',')
assert x_b.shape[0] == len(b_measure), 'Number of measument should be equal to number of position'



# radiation source candidate
c_q = []

z_q = []
#MAX_I = 25
MAX_I=26
# MAX_I=21
MAX_J=26
# MAX_J = 21
# MAX_J=21
factor=10
#factor = 0.9
# factor=12.5
for j in range(MAX_J):
    for i in range(MAX_I):
        x_pos = factor * (i - (MAX_I - 1) / 2);
        z_pos = factor * (j - (MAX_J - 1) / 2);
        c_q.append([x_pos, 0, z_pos])
        
MAX_IY=26
MAX_IZ=26
factor_YZ=10
for iy in range(MAX_IY):
    for jz in range(MAX_IZ):
        y_pos1 = factor_YZ * iy;
        z_pos1 = factor_YZ * (jz - (MAX_IZ - 1) / 2);
        
        z_q.append([-126, y_pos1, z_pos1])

z_q = np.array(z_q)
        
        



# x_q = np.array(c_q)


x_q = np.concatenate((z_q,c_q), axis = 0) 
# x_q=np.array(c_q)
# distance matrix
radiation_factor = 100
dist_mat = (x_b[:, np.newaxis]-x_q[np.newaxis])
dist_mat = np.linalg.norm(dist_mat, axis=2)
A = radiation_factor/dist_mat



# # save for MATLAB
# import numpy as np
# import scipy.io
# scipy.io.savemat('data_for_matlab.mat', dict(A=A, b_measure=b_measure, x_q=x_q, x_b=x_b))


## Optimization


# visualize square image
# def imshow(src):
#     l = int(np.sqrt(src.shape))
#     plt.imshow(src.reshape(l,l))
    
def imshow(src):
    src1,src2 = np.split(src, 2, axis = 0)
    l_1 = int(np.sqrt(src1.shape))
    l_2 = int(np.sqrt(src2.shape))
    #plt.imshow(src1.reshape(l_1,l_1))
    plt.imshow(src2.reshape(l_2,l_2))
    plt.colorbar()


# initial guess for radiation distribution
M = x_q.shape[0]
q_max = 10
q_init = 1/q_max
q = np.array([q_init]*M)



# imshow(q)



b_ave = A.dot(q)
# imshow(b_ave)



def score_func(q):
    global A, b_measure
    b_ave = A.dot(q)
    score = np.sum(b_measure*np.log(b_ave))-np.sum(b_ave)
    return score

def grad_func(q):
    global A, b_measure
    b_ave = A.dot(q)
    grad_tmp = (b_measure/b_ave)[:,np.newaxis]*A
    grad = grad_tmp.sum(axis=0) - A.sum(axis=0)
    return grad


### Gradient decent


optim_factor = 0.0005
init_score = score_func(q)
print(f'initial score:{init_score}')



for i in range(1, 50):
    q_diff = grad_func(q)
    q += optim_factor*q_diff
    
#     bound > 0
    q[q<0]=0.0000001
    
    score = score_func(q)
    if i%2==0:
        plt.figure()
        plt.title(f'iter:{i} score:{score} q_max:{q.max():.3}, q_min:{q.min():.3}')
        imshow(q)



max_idx = np.argmax(q)
print(f'max intensity location: {x_q[max_idx]}')



# SLSQP
#Sequential (least-squares) quadratic programming (SQP)


# from scipy import optimize



# # initial guess for radiation distribution
# M = x_q.shape[0]
# q_max = 10
# q_init = 1/q_max
# q = np.array([q_init]*M)

# # bound
# lb = [0.000001]*M
# ub = [np.inf]*M
# bounds = optimize.Bounds(lb, ub)

# method
# method='SLSQP'
# options={'disp': True, 'iprint':2}
# # method='L-BFGS-B'
# # options={'disp': True, 'iprint':101, 'maxfun': 150000}

# # method='trust-constr'
# # options={'disp': True, 'verbose':2}

# # Formulated as error function
# inv_score_fun = lambda x:-score_func(x)
# inv_grad_fun = lambda x:-grad_func(x)



# #gradient check
# for _ in range(3):
#     rand_q = np.random.rand(M)
#     err = optimize.check_grad(inv_score_fun, inv_grad_fun, rand_q)
#     err/=inv_grad_fun(rand_q).mean()
#     print(f'Error ratio:{err:.4}')



# res = optimize.minimize(inv_score_fun, q, method=method, bounds=bounds, options=options)
# print(res.message)



# print(f'Final score:{-res.fun}')
# imshow(res.x)



# thresh = res.x.max()/10
# idx = np.where(res.x>thresh)
# print(f'high intensity location:\n {x_q[idx[0]]}')
# #imshow(res.x>thresh)
# imshow(res.x)
# print(x_q[idx[0]])

# [array([112.5, 100. , 112.5, -25. , -25. ]),
#  array([-25. , -12.5, -12.5,  75. ,  87.5])]
# list([x_q[idx[0], 0], x_q[idx[0], 2]])
# true_value = list([[100,100,100, -25, -25], [100,100,100, -25, -25]])

# fig, ax = plt.subplots()

# ax.scatter(x_q[idx[0], 0], x_q[idx[0], 2])
# ax.scatter(true_value[0], true_value[1])
# plt.show()

# plt.scatter(x_q[idx[0], 0], x_q[idx[0], 2]).

# list([[100, -25], [100, -10], [100, -10], [-25, 75], [-25, 75]])


 # true_value = list([[100,100,100, -25, -25], [100,100,100, -25, -25]])

# plt.scatter(true_value[0], true_value[1])

# Pattern1 = “200cm121x125”
# Pattern2 = “399cm121x125”
# Root = ET.perse(pattern1 + “/detector_loc_000.xml”).getroot()
# Filename = pattern1 + “/detector_loc_000.xml”
# “200cm121x125”, “200cm121x125”}
# Pattern[0] +  “/sssss.xml”
