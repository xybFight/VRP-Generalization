<h1 align="center"> Improving Generalization of Neural Vehicle Routing Problem Solvers Through the Lens of Model Architecture </h1>

This paper take a novel perspective on model architecture to enhance the generalization of neural VRP solvers, specifically propose an Entropy-based Scaling Factor (ESF) and a Distribution-Specific (DS) decoder to enhance the size and distribution generalization, respectively.

1. Implementation of Entropy-based Scaling Factor：
   <p align="center"><img src="./imgs/ESF.jpg" width=95%></p>
   1) Given a fix size-trained (n_{tr}) model (e.g., POMO), just need apply log_{n_{tr}}n_{te} within each attention module when solving VRPs of size n_{te};
   2) Given an unfixed size-trained model (e.g., OMNI-VRP), just set a baseline n_{b} (e.g., 50), and then apply log_{n_{b}}n_{tr} and log_{n_{b}}n_{te} during training and testing, respectively.

   We present the results of 2) on OMNI-VRP. For 1), you can verify it by yourself.

2. Implementation of DS decoder：
   <p align="center"><img src="./imgs/DS.jpg" width=95%></p>
   The DS decoder explicitly models VRPs of multiple training distribution patterns through multiple auxiliary light decoders.


### Acknowledgments

* We also would like to thank the following open-source repositories, which are baselines of our code:

  * https://github.com/jieyibi/AMDKD

  * https://github.com/yd-kwon/POMO

  * https://github.com/RoyalSkye/Omni-VRP


