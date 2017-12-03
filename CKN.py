import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.feature_extraction.image import extract_patches_2d
from Cell import *
from image_processing_utils import *
import time
import torchvision.transforms as transforms

# CNN Model
class CKN(nn.Module):
    def __init__(self, n_components=[10,10,10],n_layers=3,n_patches=[5,2,2],subsampling_factors=[2,2,1],quantiles=[0.1,0.1,0.1],spacing=[2,2,2],batch_size=[100,200,500],iter_max=100,n_patches_per_graph=[30,20,20]):
        super(CKN, self).__init__()
        self.Kernel={k:Cell(n_components=n_components[k],size_patch=n_patches[k],iter_max=iter_max,subsampling_factor=subsampling_factors[k],spacing=spacing[k],n_patches_per_graph=n_patches_per_graph[k]) for k in range(n_layers)}
        self.n_patches=n_patches
        self.n_layers=n_layers
        self.n_patches_per_graph=n_patches_per_graph
        self.subsampling_factors=subsampling_factors
        #self.size_maps=size_maps
        # self.type_zero_layer=type_zero_layer  ### did
        #   # The authors of the CKN paper propose 3 types of
        #   # 0: extract patches from the image, substracting the mean color (or mean intensity)
        #   #    and performing contrast normalization
        #   # 1: extract patches from the image, substracting the mean color (or mean intensity)
        #   # 2: extract patches from the image, without substracting the mean color (or mean intensity)
        #   # 3: subsample orientations from the two-dimensional gradient at every pixel,
        #   #    it requires npatches of the form [1 * *] and a recommended value for size_maps is
        #   #    size_maps=[12 * *]  (subsample 12 orientations)
        # self.quantiles=quantiles
        self.model_patch= {k:rel_distance_patch(n_patches[k])for k in range(n_layers)}
        self.batch_size=batch_size
        #self.center_patches=center_patches   ###  should the patches be centered
        #self.center_patches=center_patches   ### should the data be normalized

    	
    def train_network(self,X):
        '''
        trains each cell individually, using the outputs of the k-1th layer as
        input to the kth layer
        
        INPUT
        ----------------------------------------------------------
        X:  training dataset
        OUTPUT
        ----------------------------------------------------------
        self: trained architecture
        input_map:  the last activation map of the last layer. 
        '''
        output_map=X
        for k in range(self.n_layers):
            #if self.normalize_data:
            #    output_map=normalize_output(output_map)
            print(output_map.size())
            if k==0:
                ### initial layer. Different intialization since we have images as input
                self.Kernel[k].init_W(X,patches_given=False)
                input_map=self.Kernel[0].all_patches
            else:
                ### deeper layers. Give patches as input.
                input_map = output_map
                #extract_patches_from_vector(output_map, size_patch=self.Kernel[k].size_patch)
                print 'size input at layer ',k , ' : ',input_map.size()
                self.Kernel[k].init_W(input_map,patches_given=True)

            self.Kernel[k].fit(input_map,init=False)
            output_map=self.Kernel[k].get_activation_map(verbose=True)
        return output_map
        
        
    def propagate_through_network(self,X,patches_given=True):
        '''
        propagate the input data throught the network, using the outputs of the k-1th layer as
        input to the kth layer
        
        INPUT
        ----------------------------------------------------------
        X        :  training dataset
        OUTPUT
        ----------------------------------------------------------
        self: trained architecture
        input_map:  the last activation map of the last layer. 
        '''

        for k in range(self.n_layers):
            if k==0:
                if not patches_given:
                    input_map = extract_patches_from_image(X, self.n_patches[0])
                    if len(input_map.size()) == 4:
                        input_map = input_map.view(input_map.size()[0], input_map.size()[1], -1)
                else:
                    #### pool patches
                    input_map = extract_patches_from_vector(X, self.n_patches[0])
            else:
                input_map=extract_patches_from_vector(output_map, self.n_patches[k])
            print k, input_map.size()
            #norms=input_map.norm(p=2,dim=2)
            self.Kernel[k].data_has_been_normalized = False
            norms = torch.sqrt(torch.mean(input_map**2, dim=2))
            input_map=normalize_output(input_map)
            self.Kernel[k].data_has_been_normalized=True
            self.Kernel[k].data_has_been_scaled = False
            print 'input map in layer ', k, ' has size ',input_map.size()
            output_map=self.Kernel[k].get_activation_map(X=input_map,norms=norms,verbose=True)
        return output_map
    
    	 	

    	 	
    

    
