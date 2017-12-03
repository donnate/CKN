import numpy as np
import sklearn as sk
import copy
import torch

###############################################################################
###### FUNCTIONS DEFINED IN THE MATLAB CODE FOR PREPROCESSING PATCHES #########
###############################################################################
def centering(X,n_channels):
	size_channel=X.shape[1]/n_channels;
	Y=np.zeros(X.shape)
	for ii in range(n_channels):
   		XX=X[(ii-1)*size_channel+1:ii*size_channel,:]
   		Y[(ii-1)*size_channel+1:ii*size_channel,:]=XX-np.mean(XX)
	return Y
	

def contrast_normalize(X):
	nrm=np.sqrt(sum(X**2,0));
	def trim(x): return np.max([x,0.00001])
	trim_v=np.vectorize(trim)
	nrm=trim_v(nrm)
	return nrm.dot(X)
	
def contrast_normalize_median(X):
	nrm=np.sqrt(sum(X**2,0));
	med=np.max([np.median(nrm),0.00001])
	return 1.0/med*X



def get_zeromap(inpt, type_zerolayer):
    if not dtype(inpt)==float:
        inpt=double(inpt)/255
    sx=inpt.size[0]
    sy=inpt.size[1]
    if 3*sx == sy: #assume this means rgb 
        if type_zerolayer==3:
            out=inpt[:,sx+1:2*sx]; # get the green channel
        else:   
            out=inpt.reshape([sx, sx, 3])
    else:
        out=inpt
    return out



###############################################################################
##############    UTILITIES FOR PATCH EXTRACTION AND POOLING  #################
###############################################################################
def rel_distance_patch(size_patch):
    '''
    Function for computing the weights associated to each patch in line  of 
    the algorithm (Pooling step)
    
    INPUT
    ===========================================================================
    size_patch :  size of the patch.
    
    OUTPUT
    ===========================================================================
    model_patch: tensor of size size_patch x size_patch with the corresponding 
                 decay weights
    '''

    model_patch=np.zeros((size_patch,size_patch))
    cntr_x,cntr_y=[size_patch//2+size_patch%2-1,size_patch//2+size_patch%2-1]
    for ii in range(size_patch):
        for jj in range(size_patch):
             model_patch[ii,jj]=(ii-cntr_x)**2+(jj-cntr_y)**2
    return torch.Tensor(model_patch)
    
    
    
    
    
    
    
    

def extract_patches_from_image(data,size_patch, zero_padding=True):
    '''
    Function for extracting patches from an image dataset
    
    INPUT
    ===========================================================================
    data           :     tensor of size [n_d , dim_x, dim_y] where n_d is the size
                         of the dataset, dim_x and dim_y represent the size of the
                         images
    size_patch     :     size of the patch (int)
    zero_padding   :     borders dealt with using zero padding (alternative (?)
                         not yet implemented)
                            
    OUTPUT
    ===========================================================================
    patches        :    tensor of  patches of size [n_p , n_d, size_patch**2 ] 
    '''
    size_image=data.size()[1:]
    a=size_patch+size_image[0]
    b=size_patch+size_image[1]
    padded_image=torch.Tensor(np.zeros((data.size()[0],a,b)))
    padded_image[:,size_patch//2:(size_patch//2)+size_image[0],size_patch//2:(size_patch//2)+size_image[1]]=data
    nx,ny=padded_image.size()[1:]
    patches=torch.Tensor(np.array([padded_image[:,ii:ii+size_patch,jj:jj+size_patch].numpy() for ii in np.arange(0,nx-size_patch,1) for jj in np.arange(0,nx-size_patch,1) ]))
    return patches
    
def extract_patches_from_vector(X,size_patch, zero_padding=True):
    '''
    Function for extracting patches from a patch dataset
    
    INPUT
    ===========================================================================
    data           :     tensor of size [n_p, n_d , p_dim] where n_p is the
                         number of patches per image, n_d is the size of the dataset
                         and p_dim is the size of each filter.
    size_patch     :     size of the patch (int)
    zero_padding   :     borders dealt with using zero padding (alternative (?)
                         not yet implemented)
                            
    OUTPUT
    ===========================================================================
    patches        :     tensor of size size_patch x size_patch with the corresponding 
                         decay weights
    '''
    nb_locations=X.size()[0]
    x_dim=int(np.sqrt(X.size()[0]))
    nb_data_points=X.size()[1]
    D=X.size()[2]
    
    #####
    size_image=[x_dim,x_dim]
    a=2*np.max([size_patch//2,1])+size_image[0]
    b=2*np.max([size_patch//2,1])+size_image[1]
    padded_image=np.zeros((a,b))
    padded_image[np.max([size_patch//2,1]):np.max([size_patch//2,1])+size_image[0],np.max([size_patch//2,1]):(np.max([size_patch//2,1]))+size_image[1]]=1
    nx,ny=padded_image.shape
    test=torch.Tensor([create_BW_mask(size_image,size_patch,ii,jj)[np.max([size_patch//2,1]):np.max([size_patch//2,1])+size_image[0],np.max([size_patch//2,1]):(np.max([size_patch//2,1]))+size_image[1]] \
        for ii in np.arange(0,nx-2*np.max([size_patch//2,1])) \
        for jj in np.arange(0,ny-2*np.max([size_patch//2,1])) ])
    test=test.view(test.size()[0],-1)
    #for i in xrange(0,batch_size*(data.size()[1]//batch_size),batch_size):
    test2=torch.stack([torch.cat([X[u,:,:].t() for u in np.where(test[j,:].numpy()!=0)[0]]\
                +[torch.Tensor(np.zeros((D,nb_data_points)))]*(size_patch**2-len(np.where(test[j,:].numpy()!=0)[0]))) for j in range(nb_locations)])
    
    return test2.permute(0,2,1)
 
 
def extract_selected_patches(data,id_patch,size_patch, zero_padding=True):
    '''
    Function for extracting the pairs of patches necessary for training the kernel
    
    INPUT
    ===========================================================================
    data           :     tensor of size [n_p,n_d , p_dim] where n_d is the size
                         of the dataset, dim_x and dim_y represent the size of the
                         images
    id_patch       :     list of the patches to extract. Each entry is a tuple t
                         of size 3 where t[0]: id of the image, t[1] (resp t[2])
                         is the id of the 1st (resp the 2nd) patch in image t[0]
    size_patch :  size of the patch
    
    OUTPUT
    ===========================================================================
    selected_patches:     tensor of size [len(id_patch),2,dim_patch ] where each
                         entry (w.r.t the first dimension)
                         corresponds to a pair of patches
    '''

    selected_patches=torch.Tensor(len(id_patch),2,size_patch**2)
    patches=extract_patches(data,size_patch, zero_padding=True)
    for j in range(len(id_patch)):

            while torch.sum(patches[id_patch[j][1],id_patch[j][0],:])==0 and torch.sum(patches[id_patch[j][2],id_patch[j][0],:])==0:
                nx,ny=np.random.choice(range(data.size()[1]*data.size()[2]),2)
                id_patch[j]=[id_patch[j][0],nx,ny]
            selected_patches[j,0,:]=patches[id_patch[j][1],id_patch[j][0],:]
            selected_patches[j,1,:]=patches[id_patch[j][2],id_patch[j][0],:]
    return selected_patches

    
def extract_patch_mask(N,size_image,size_patch, beta=1,zero_padding=True):
    '''
    Function for computing the weights associated to each patch in line  of 
    the algorithm (Pooling step)
    
    INPUT
    ===========================================================================
    size_patch :  size of the patch
    
    OUTPUT
    ===========================================================================
    model_patch: tensor of size size_patch x size_patch with the corresponding 
                 decay weights
    '''
    a=2*np.max([size_patch//2,1])+size_image[0]
    b=2*np.max([size_patch//2,1])+size_image[1]
    padded_image=np.zeros((a,b))
    padded_image[np.max([size_patch//2,1]):np.max([size_patch//2,1])+size_image[0],np.max([size_patch//2,1]):(np.max([size_patch//2,1]))+size_image[1]]=1
    nx,ny=padded_image.shape
    print(nx,ny)
    patches=torch.Tensor(np.array([(padded_image*create_distance_mask(size_image,size_patch,ii,jj,beta=beta))[np.max([size_patch//2,1]):np.max([size_patch//2,1])+size_image[0],np.max([size_patch//2,1]):(np.max([size_patch//2,1]))+size_image[1]] for ii in np.arange(0,nx-2*np.max([size_patch//2,1])) for jj in np.arange(0,ny-2*np.max([size_patch//2,1])) ]))
    n_p,n_d,p_dim=patches.size()
    #nn_zero=(torch.sum(torch.abs(patches.view(n_d*n_p,p_dim)    
    return patches


def create_distance_mask(size_image,size_patch,ii,jj,beta=1):
    '''
    Function for computing the weights associated to each patch in line  of 
    the algorithm (Pooling step)
    
    INPUT
    ===========================================================================
    size_patch :  size of the patch
    
    OUTPUT
    ===========================================================================
    model_patch: tensor of size size_patch x size_patch with the corresponding 
                 decay weights (decays r=with distance from center of the patch)
    '''
    image=np.zeros((size_image[0]+2*np.max([size_patch//2,1]),size_image[1]+2*np.max([size_patch//2,1])))
    if size_patch is not None:
        mask=rel_distance_patch(size_patch).numpy()
    else:  ### include information about the whole imahe
        mask = rel_distance_patch(size_patch).numpy()
    image[ii:ii+size_patch,jj:jj+size_patch]=np.exp(-1.0/beta**2*mask)
    return image

def create_BW_mask(size_image,size_patch,ii,jj):
    '''
    Function for computing the weights associated to each patch in line  of 
    the algorithm (Pooling step)
    
    INPUT
    ===========================================================================
    size_patch :  size of the patch
    
    OUTPUT
    ===========================================================================
    model_patch: tensor of size size_patch x size_patch with the corresponding 
                 decay weights
    '''
    image=np.zeros((size_image[0]+2*np.max([size_patch//2,1]),size_image[1]+2*np.max([size_patch//2,1])))
    mask=rel_distance_patch(size_patch).numpy()
    image[ii:ii+size_patch,jj:jj+size_patch]=1
    return image
    

    
def normalize_output(input_map, epsilon=0.0001,center_data=False,center=None,verbose=False):
    '''
    Function for normalizing the patches (obtaining the Psi_tilde l2-normalized 
    version of the patches)
    
    INPUT
    ===========================================================================
    input map  :  tensor of patches [n_p;n_d;p_dim] where n_p is the number of patches
                 per image, n_d: nb of data samples and p_dim is the dimension
                 of each patch
    epsilon    :  lower bound for the norm
    center_data:  (boolean) should the data be centered?
    center     :  (ignored if center_data==False) center of the data. If None, it is set to be the empirical mean of the data.
    
    OUTPUT
    ===========================================================================
    norm_output: tensor of size size_patch x size_patch with the corresponding 
                 decay weights
    '''
    n_p, n_d,p = input_map.size()
    norm2 = torch.sqrt(torch.sum(input_map**2, dim=2))
    #n_p,n_d=norm2.size()
    if verbose: 
        print('dim norm', norm2.size())
        print('max norm',torch.max(norm2))
        print('min norm',torch.min(norm2[norm2>0]))

    if center_data:
        if center is None:
            center=torch.mean(input_map.view(n_p*n_d,p),0).view(1,1,p).expand(n_p, n_d,p)
            input_map=input_map-center
    norm2[norm2<epsilon]=1                                ###  To tackle the case of the patches with variance 0
    norm_output=torch.div(input_map,norm2.view(n_p,n_d,1).expand_as(input_map))
    return norm_output