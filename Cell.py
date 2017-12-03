
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import scipy.sparse as sp
from scipy.linalg import svd
from sklearn.utils import check_random_state
from pattern_functions import *
from sklearn.cluster import KMeans
from image_processing_utils import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import time
import math
from sklearn.linear_model import LinearRegression

class Cell():
    
    def __init__(self,  n_components=50, iter_max=5,random_state=None,size_patch=5,n_patches_per_graph=10,lr=0.001,batch_size=100,subsampling_factor=2,spacing=1,type_optim='adamax'):
        self.eta = Variable(torch.Tensor(1.0/n_components *(np.ones((n_components,)))),requires_grad=True)
        self.W = torch.Tensor(Variable(None))
        self.sigma=None
        self.spacing=spacing
        self.training_data_has_been_normalized = False
        self.data_has_been_normalized=False
        self.data_has_been_scaled=False
        self.training_data_has_been_scaled=False
        self.norms=None
        self.n_components = n_components
        self.random_state = random_state
        self.iter_max=iter_max
        self.print_lag=15
        self.lr=lr
        self.all_patches=None
        self.training_data=None
        self.size_patch=size_patch
        self.patches=None
        self.distances=None
        self.subsampling_factor=subsampling_factor
        self.batch_size=batch_size
        self.output=None
        self.training_output=None
        self.pca=None
        self.standardize=None
        self.n_patches_per_graph=n_patches_per_graph
        self.type_optim=type_optim

    		
    def select_training_patches(self,X,patches_given=True,verbose=True):
        ''' creates the training dataset by extracting all the patches from
        each data point in X and randomly selects pairs of patches to constitute
        the training datase.
        Either randomly extract patches or samples from a patch list depending 
        on the type of the input X)
        
        INPUT
        =======================================================================
        X           :      complete data. Tensor of dimension:
                                -  [N_d , dim_x , dim_y] if patches_given=False,
                                where N_d is the number of images (data samples),
                                and dim_x and dim_y the size of each image 
                           or:
                               -  [N_p , N_d , p_dim] if patches_given=True,
                                where N_p is the number of patches per image,
                                N_d is the number of images,
                                and p_dim the dimension of each patch
        patches_given:     Boolean (detrmines how to handle the data tensor X)
        
        OUTPUT
        =======================================================================
        selected_patches     :
        
        
        self.training_data   :  filled with samples for training the kernel
        self.all_patches     :  filled with all patches from all samples
        self.pca             :  trained PCA transformation of the patches
        self.standardize     :  trained  standardization of the patches
        
        '''

        id_patch=[] 
        n_patches_per_graph=np.min([self.n_patches_per_graph,X.size()[0]])
        size_patch=self.size_patch


        ########   STEP 0: extract patches from input map
        if patches_given==False:
            patches=extract_patches_from_image(X,size_patch, zero_padding=True)
                
            if len(patches.size())==4:
                patches=patches.view(patches.size()[0],patches.size()[1],patches.size()[2]*patches.size()[3])
        else:
            if size_patch==1:
                patches=X
            else:
                patches=extract_patches_from_vector(X,size_patch, zero_padding=True)  ## n_patches x n_data x dim_patches
        print "patches extracted with size ", patches.size()
        ########   STEP 1: normalize the data (the optimization pb works on l2 normalized version of the patches)


        self.all_patches=normalize_output(patches)
        self.norms=torch.sqrt(torch.mean(patches**2,2))
        self.norms=torch.clamp(self.norms,0.0,float(np.percentile(self.norms.numpy(),95)))
        self.training_data_has_been_normalized=True
        if verbose: print "Training patches normalized extracted with size ", patches.size()


        ### So now all the patches have unit norm. But they are still very high dimensional...
        ### Added w.r.t the original paper: apply dimensionality reduction so as to make it tractable (debatable)
        n_p,n_d,p_dim=self.all_patches.size()
        standard=RobustScaler(quantile_range=(5.0,95.0)) ## use RobustScaler just in case there are outliers
        #X_tilde=torch.cat((self.training_data[:,0,:],self.training_data[:,1,:]), dim=0)
        X_tilde2=standard.fit_transform(self.all_patches.view(-1,p_dim).numpy())
        pca=PCA(n_components=np.min([p_dim,30]),whiten=False)
        X_tilde2=pca.fit_transform(X_tilde2)
        self.all_patches=torch.Tensor(X_tilde2).contiguous().view(n_p,n_d,-1)
        self.pca=pca
        self.standardize=standard
        self.training_data_has_been_scaled=True
        print 'Training patches have been standardized and have size',self.all_patches.size()

        #### All the data has now been preprocessed.

        ###### STEP 2: Extract several random pairs of patches from each image

        for i in range(patches.size()[1]):            ## select at random 2 nodes in the adjacency matrix:
            a=X.size()[1]
            for j in range(n_patches_per_graph):
                nx,ny=np.random.choice(range(patches.size()[0]),2)
                id_patch+=[[i,nx,ny]]
        print(id_patch[:10])  
        
        if len(patches.size())==4:
            selected_patches=torch.Tensor(len(id_patch),2,self.all_patches.size()[2]*self.all_patches.size()[3])  
        else:
            selected_patches=torch.Tensor(len(id_patch),2,self.all_patches.size()[2])   ## n_patches x n_data x dim_patch
        print('sel', selected_patches.size())
        
        it_j={}

        ##### STEP 2(bis): discard patches where both are zeros
        for j in range(len(id_patch)):
            it_j[j]=0
            while torch.sum(torch.abs(patches[id_patch[j][1],id_patch[j][0],:]))==0 and torch.sum(torch.abs(patches[id_patch[j][2],id_patch[j][0],:]))==0:
                nx,ny=np.random.choice(range(patches.size()[0]),2)
                id_patch[j]=[id_patch[j][0],nx,ny]
                it_j[j]+=1

            selected_patches[j,0,:]=self.all_patches[id_patch[j][1],id_patch[j][0],:]
            selected_patches[j,1,:]=self.all_patches[id_patch[j][2],id_patch[j][0],:]
        #
       

        self.training_data=selected_patches
        return selected_patches
        
        
    def init_W(self,X,patches_given=True):
        ''' initialize the filters for the kernel approximation using kmeans
        
        
        INPUT
        =======================================================================
        X           :      complete data. Tensor of dimension:
                                -  [N_d , dim_x , dim_y] if patches_given=False,
                                where N_d is the number of images (data samples),
                                and dim_x and dim_y the size of each image 
                           or:
                               -  [N_p , N_d , p_dim] if patches_given=True,
                                where N_p is the number of patches per image,
                                N_d is the number of images,
                                and p_dim the dimension of each patch
        patches_given:     Boolean (determines how to handle the data tensor X)
        
        OUTPUT
        =======================================================================
        None 
        
        
        self.W               :  initialized filters  for the kernel approximation
        self.sigma           :  variance sigma is computed (using the 10th 
                                percentile of the data)
        
        '''
        
        self.training_data=self.select_training_patches(X,patches_given=patches_given)
        if self.training_data_has_been_normalized==False:
            print 'Error training data has not been normalized!!!!!'
            self.training_data=normalize_output(self.training_data, verbose=False)
            self.training_data_has_been_normalized=True
        n_p,n_d,p_dim=self.all_patches.size()
        n_p_train,_,p_dim_train=self.training_data.size()
        print('The data has size',self.all_patches.size())
        print('all okay',p_dim_train==p_dim )
        
        
        X_tilde=torch.cat((self.training_data[:,0,:],self.training_data[:,1,:]), dim=0)
        distances2=torch.sum((self.training_data[:,0,:]-self.training_data[:,1,:])**2,dim=1)
        if self.sigma is None:
        	### set to be quantile
            ### compute the distance between patches
            self.sigma=np.max([np.percentile(distances2.numpy(),10.0),0.0001])
        
        print('The variance is: ',self.sigma)
        km=KMeans(n_clusters=self.n_components)
        inds=range(int(X_tilde.size()[0]))
        np.random.shuffle(inds)
        km.fit(X_tilde[inds[:np.min([5000,int(X_tilde.size()[0])])],:].numpy())
        
        self.W=Variable(torch.Tensor(km.cluster_centers_), requires_grad=True)
        
    
    
    
    
    def fit(self, X=None, init=True,patches_given=True):
        '''
        fits the cell to the input data (either X if X is given or to self.trainingdata
        if X is None). 
        INPUT
        =======================================================================
        X           :      complete data. Tensor of dimension:
                                -  [N_d , dim_x , dim_y] if patches_given=False,
                                where N_d is the number of images (data samples),
                                and dim_x and dim_y the size of each image 
                           or:
                               -  [N_p , N_d , p_dim] if patches_given=True,
                                where N_p is the number of patches per image,
                                N_d is the number of images,
                                and p_dim the dimension of each patch
        init         :     Boolean: have the filters already been initialized?
        patches_given:     Boolean (determines how to handle the data tensor X)
        
        OUTPUT
        =======================================================================
        None 
        
        
        '''
        
        rnd = check_random_state(self.random_state)
        D=self.n_components
        if init==True: 
            #self.select_training_patches(X,patches_given=patches_given)
            self.init_W(X,patches_given=patches_given)
        X_input=Variable(self.training_data, requires_grad=False)    ###dim=[nb_pair_patches x 2  x dim_patch]
        print('size input: ',X_input.size())
        N=X_input.size()[0]
        n=X_input.size()[1]
        expected_output=torch.exp(torch.div(torch.sum((X_input[:,0,:]-X_input[:,1,:])**2,1),-2*self.sigma))
        loss_func = nn.MSELoss()
        if self.type_optim=='adamax':
            optimizer = optim.Adamax([self.W,self.eta],lr=self.lr) # instantiate optimizer with model params + learning rate (SGD for the moment)
        else:
            optimizer = optim.Adam([self.W, self.eta], lr=self.lr)
        batch_nb=N//self.batch_size
        batch_size=self.batch_size
        p_size=X_input.size()[2]
        
        self.training_loss=[]
        for t in range(self.iter_max):
                tic0=time.time()
                overall_loss=0
            #def closure():
                for b in range(batch_nb):
                    #self.eta=F.relu(self.eta)
                    
                    XX=X_input[b*batch_size:(b+1)*batch_size,0,:].contiguous().view((1,batch_size,p_size)).expand(D,batch_size,p_size)
                    YY=X_input[b*batch_size:(b+1)*batch_size,1,:].contiguous().view((1,batch_size,p_size)).expand(D,batch_size,p_size)
                    output=(XX-self.W.view(D,1,p_size).expand(D,batch_size,p_size))**2+(YY-self.W.view(D,1,p_size).expand(D,batch_size,p_size))**2


                    output=torch.div(torch.sum(output,2),-self.sigma)
                    #print(output[:10])
                    output2=torch.exp(output)


                    output2=torch.matmul(F.relu(self.eta),output2)

                    loss=loss_func(output2,expected_output[b*batch_size:(b+1)*batch_size])+torch.sum((F.relu(self.eta)-self.eta)**2)
                    overall_loss+=loss[0]
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    

                
                tac0=time.time()
                print 'Epoch:',t,' Mean loss:',overall_loss.data/(batch_nb),' time per epoch: ',tac0-tic0
                self.training_loss.append(overall_loss.data/(batch_nb))
                #print(eta[:5])
        self.eta=F.relu(self.eta)

    
    	#self.output=output
    	
   
    def get_activation_map(self, X=None,norms=None,verbose=False):
            ''' computes the activation map for the different signals
            
            INPUT
            ----------------------------------------------------------
            id_map           :    nb/id of the layer
            X                :    data to process (defaults to self.all_patches)
            
            Assumes that the layers beforehand have already been trained
            In particular:
            Assumes that self.Kernel[id_map] has parameter ``all_patches'' filed ([nb_patches x nb_graphs x dim_patch])
            
            OUTPUT
            ----------------------------------------------------------
            output_map       :    tensor of dimension [nb_new_patches x nb_graphs x dim_new_patch]
            '''
            
            p_shape=self.size_patch                   ## p_shape: dimension of the new patches
    
    
            #### STEP 1: normalize data
            if X is None:   ### this means that we are in the training phase (hence we should compute the activation function w.r.t self.all_patches)
                input_map=self.all_patches         ## input map: should check that it is of dim
                n_p,n_d,p_dim=input_map.size()                
                if self.training_data_has_been_normalized==False:
                    self.norms=torch.sqrt(torch.mean(input_map**2,dim=2))
                    input_map=normalize_output(input_map)
                    self.training_data_has_been_normalized=True
                    print 'Input map has size', input_map.size()

                if self.training_data_has_been_scaled==False:
                    if self.standardize is not None:
                        input_map=self.standardize.transform(input_map.contiguous().view(-1,input_map.size()[2]).numpy())
                        input_map=torch.Tensor(input_map).view(n_p,n_d,-1)
                        n_p,n_d,p_dim=input_map.size()
                    if self.pca is not None:
                        input_map=self.pca.transform(input_map.contiguous().view(-1,input_map.size()[2]).numpy())
                        input_map=torch.Tensor(input_map).view(n_p,n_d,-1)
                        n_p,n_d,p_dim=input_map.size()
                    self.training_data_has_been_scaled=True
                    print 'Training was scaled and has size', input_map.size()
                    
            else:        ### this means that we have to compute the activiation for a particular input
            	print 'X is given'
                input_map=X
                #self.data_has_been_scaled =False
                print 'Activation map init. Is normalized yet:', self.data_has_been_normalized,'. Input map has size', input_map.size()
                print(X.size())
                self.norms=norms
                if self.data_has_been_normalized==False:
                    self.norms=torch.sqrt(torch.mean(input_map**2,dim=2))
                    input_map=normalize_output(input_map)
                    self.data_has_been_normalized=True
                    print 'Input map has size', input_map.size()

                n_p,n_d,p_dim=X.size()
                if self.data_has_been_scaled==False:
                    if self.standardize is not None:
                            input_map=self.standardize.transform(input_map.contiguous().view(-1,input_map.size()[2]).numpy())
                            input_map=torch.Tensor(input_map).view(n_p,n_d,-1)
                            n_p,n_d,p_dim=input_map.size()
                    if self.pca is not None:
                            input_map=self.pca.transform(input_map.contiguous().view(-1,input_map.size()[2]).numpy())
                            input_map=torch.Tensor(input_map).view(n_p,n_d,-1)
                            n_p,n_d,p_dim=input_map.size()
                    self.data_has_been_scaled=True
                    print 'Activation map :Given data was scaled and has size', input_map.size(), ' and max ', torch.max(input_map)
            
            
            if len(input_map.size())==4:                      ## [nb_patches x nb_graphs x dim_patch] e.g, in MNIST there are 28 x28 patches,  so for a filter of size 5, this is of dimension  784 x60000 x 25
                input_map=input_map.view(input_map.size()[0],input_map.size()[1],-1).contiguous()
                #self.input_map=input_map
            n_p,n_d,p_dim=input_map.size()
            
            
            
            #### STEP 2: extract relevant parameters 
            spacing=self.spacing#np.min([input_map.size()[0]//p_shape,input_map.size()[1]//p_shape])  
            gamma=self.subsampling_factor
            beta=gamma*spacing
            batch_size=self.batch_size
            D=self.n_components
            sigma=self.sigma
            
            
            ### STEP 3: Call the normalized version of the patchs
            norm = self.norms    # [nb_patches x nb_graphs ]


            ## STEP 4: Extract patches with the size of the current image  (dim)
            dim=int(np.sqrt(input_map.size()[0]))     ###  this is the size of the new patch
            size_patch=p_shape
            mpatches=extract_patch_mask(batch_size,[dim,dim],size_patch, beta=beta,zero_padding=True)
            mpatches=mpatches.view(-1,dim*dim)
            selected_pixels=[i*dim+j for j in np.arange(0,dim, self.subsampling_factor) for i in np.arange(0,dim, self.subsampling_factor)]
            #if verbose: print(selected_pixels)
                
                
            ### STEP 5: Compute activation set in a batch-wise fashion
            output_map=torch.Tensor(len(selected_pixels),n_d,D)
            for b in range(n_d//batch_size):
                print(b)
                temp=torch.Tensor(D,n_p*batch_size)
                zeta=torch.Tensor(D,n_p*batch_size)
                #print(norm_patches.size())
                XX=input_map[:,b*batch_size:(b+1)*batch_size,:].contiguous().view(-1,p_dim)
                tot=torch.exp(torch.div(torch.sum((XX.unsqueeze(1).expand(n_p*batch_size,D,p_dim)- self.W.data.unsqueeze(0).expand(n_p*batch_size,D,p_dim))**2,dim=2),-sigma))
                print('tot',torch.max(tot))
                w=torch.sqrt(self.eta.unsqueeze(0).data.expand_as(tot))*tot  ### temporary: zeta=  eta_L e^(-1/sigma_k**2 *||Psi_tilde(z)-W_L||^2)
                temp=w
                if verbose:
                    print 'max w',torch.max(w)
                    print 'nb 1 of null is ',torch.sum(temp!=temp)
                    print 'size w:',w.size()
                    print 'size norm:', norm.size()
                Reg=norm[:,b*batch_size:(b+1)*batch_size].contiguous().view(n_p*batch_size,1).expand_as(w)
                #print(Reg.size())
                if verbose: print('max Reg',torch.max(Reg))
                zeta=Reg*w#output2=output.view(n_d*n_p,D,p_dim)    ### zeta=  ||Psi(z)||_2 eta_L e^(-1/sigma_k**2 *||Psi_tilde(z)-W_L||^2)
                ## zeta has how size: [D,n_p*batch_size] and Reg:[n_p*batch_size,1]
                if verbose: print('zeta',zeta.size())
                if verbose: print('nb 2 of null is ',torch.sum(zeta!=zeta))
                zeta=zeta.view(n_p,batch_size,D)      ### gather tensor

                test=torch.matmul(mpatches,zeta.view(n_p,batch_size*D))
                if verbose: print('test',zeta.size())
                out=test.view(n_p,batch_size,D)
                if verbose: print('nb 3 of null is ',torch.sum(out!=out))
                #print(out)
                output_map[:,b*batch_size:(b+1)*batch_size,:]=out[selected_pixels,:,:]   
            if verbose: print('nb final of null is ',torch.sum(output_map!=output_map))
            self.output=np.sqrt(2.0/math.pi)*output_map #_map_norm
            print('size output',output_map.size() )
            return np.sqrt(2.0/math.pi)*output_map
    		
    
    def _get_kernel_params(self):
        params = self.kernel_params
        if params is None:
            params = {}
        return params
        
    def convergence_check(self):
        '''Checks the accuracy of the kernel approximation.
        Assumes  that the kernel has been trained
        Plots the expected outputs vs predicted oututs (+ compares them to the 45 degree line)
        
        INPUT
        -----------------------------------------------------------------------
        None
        
        INPUT
        -----------------------------------------------------------------------
        linreg   : linear regession trained on expected output vs predicted output
                    (so that we have easily the slope and R2 statistic)
                    
        + regression plot
        '''
        X_input=self.training_data
        print(X_input.size())
        n_p,batch_size,p_size=self.all_patches.size()
        expected_output=torch.exp(torch.div(torch.sum((X_input[:,0,:]-X_input[:,1,:])**2,1),-2*self.sigma))
        batch_size,_,p_size=X_input.size()
        D=self.n_components
        XX=X_input[:,0,:].contiguous().view((1,batch_size,p_size)).expand(D,batch_size,p_size)
        YY=X_input[:,1,:].contiguous().view((1,batch_size,p_size)).expand(D,batch_size,p_size)
        output=(XX-self.W.data.view(D,1,p_size).expand(D,batch_size,p_size))**2+(YY-self.W.data.view(D,1,p_size).expand(D,batch_size,p_size))**2
                            #output=(XX-self.W.view(D,1,p_size**2).expand(D,X_input.size()[0],p_size**2))**2+(X_input[:,1,:].repeat(1, D)-self.W.view(1,D*n).repeat(N, 1))**2
                            #output2=output.view(N,D,n)
        output=torch.div(torch.sum(output,2),-self.sigma)
        #print(output.size())
                            #print(output[:10])
        output2=torch.exp(output)
        
                            #output2=torch.exp((-1.0/sigma).expand_as(output)*torch.sum(output.view(-1,p_size**2),1))
                            #weights=C.expand_as(eta)*F.softmax(-eta)
                            #STOP
        output2=torch.matmul(F.relu(self.eta),Variable(output2))
        #print(output2.size())
        
        linreg=LinearRegression()
        linreg.fit(X=expected_output.numpy().reshape(-1, 1),y=output2.data.numpy().reshape(-1, 1))
        fig,ax= plt.subplots()
        plt.scatter(expected_output.numpy(),output2.data.numpy())
        y=output2.data.numpy()
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        return linreg
