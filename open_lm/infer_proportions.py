import torch
import numpy as np

from extra_funcs import train_classifier, proj_simplex, round_preserving_sum, sample_and_rename_files, inference, del_dir

def comparison(x, xcandidate):

    list1 = round_preserving_sum(x.tolist())
    list2 = round_preserving_sum(xcandidate.tolist())
    list = [list1, list2]

    sample_and_rename_files(list)

    train_classifier()

    result = inference()

    del_dir("/workspace/youssef/lrz/logs/RedPajama/prop")
    del_dir("/workspace/youssef/lrz/datasets/prop/train")

    return result


def gradientless_descent(N=6, num_iter=200, radius = 0.2, alpha=0.5):

    #For measuring error
    xorig = np.array([0.0325,0.1575,0.6775,0.0525,0.0275,0.0525])
    
    # initialize x with equal probability 
    x = np.ones(N)/N
    
    error = []
    prop = []
    
    for i in range(num_iter):
        
        stepsize = 1/(i+1)**alpha
        # choose random direction with radius R
        dir = np.random.randn(N)
        dir = dir/np.linalg.norm(dir)*radius*stepsize
        
        xcandidate = proj_simplex( x + dir )
        
        # compare x with x+dir and update x
        if comparison(x, xcandidate) == 1:
            x = xcandidate

        print(i, np.linalg.norm(x-xorig), x)
        error.append(np.linalg.norm(x-xorig))
        prop.append(x)

        torch.save(error, "error.pt")
        torch.save(prop, "prop.pt")
    return x

if __name__ == "__main__":
    gradientless_descent()
