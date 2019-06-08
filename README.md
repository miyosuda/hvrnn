# Hierarchical variational autoencoder (HVRNN)

Code for the paper "Do top-down predictions of time series lead to sparse
disentanglement?" (Miyoshi et al. 2018)

Poster: https://drive.google.com/open?id=1zUrovQFNZ1AnSWkwGzEEgm2rk9lvoJY7

Paper: [http://jnns.org/conference/2018/JNNS2018_Proceedings/JNNS2018_Proceedings.pdf](http://jnns.org/conference/2018/JNNS2018_Proceedings/JNNS2018_Proceedings.pdf) [p15-16]



## Network

![network](doc/network.png)



## Disentanglement result

Dataset: Moving MNIST

![timeline](doc/timeline.png)

Figures on the left below show the mean of the 16 latent variables in each hierarchical layer with the input of the time series sequence shown above. Figures on the right below show the normalized mutual information between latent variables and the factors. With HVRNN setting with 3 layers, the first layer clearly extracts x and y movements of the digit and the second layer extracts the moment of the bouncing with horizontal and vertical walls.

#### VRNN (1 layer setting)

![layer1](doc/layer1.png)



#### HVRNN (3 layers setting)

![layer3](doc/layer3.png)

