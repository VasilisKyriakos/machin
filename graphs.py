import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sam_f = 4

def graph(filenames):
    data=[]
    for fl in filenames:
        with open(fl,"rb") as f:
            data.append(pickle.load(f))

    all_evalz = [d["evaluations"] for d in data]
    expects = [[np.mean(x) for x in y ]for y in all_evalz]
    expectz = np.mean(expects,axis=0)
    all_evz=np.array(all_evalz).flatten()
    q1=np.quantile(all_evz,0.25)
    q2=np.quantile(all_evz,0.5)
    q3=np.quantile(all_evz,0.75)
    ll = len(expectz)

    durs = [d["dur"] for d in data]
    fig , axis = plt.subplots(1,2)

    axis[0].plot(np.array(range(ll))*sam_f+1,expectz,color='green',label='Expected Return')
    axis[0].hlines(q1,1,(ll-1)*sam_f+1,color='blue',label='First Quartile')
    axis[0].hlines(q2,1,(ll-1)*sam_f+1,color='red',label='Median')
    axis[0].hlines(q3,1,(ll-1)*sam_f+1,color='black',label='Third Quirtile')
    axis[1].boxplot(durs)
    axis[0].legend(loc='center right', bbox_to_anchor=(0, 0.5))

    plt.show()
    

    return expects
    
    
