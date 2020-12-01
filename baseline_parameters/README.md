### baseline Method
* Value-Oriented Helpfulness Ranking (V-OHR) [1]  
* Context-Aware helpfulness rating prediction framework [20] (CAP) 
* Review Honesty Score (RHS) [23]    
* Matrix Factorization based method [28] (MF) 
* CNN [31] .  
* HITS based Ranking (HITS) . 

### tuning parameters of baseline
We achieved the optimal baseline results through parameter tuning.   
* V-OHR [1]: The review helpfulness is expressed by H(r), and the reviewer quality is expressed by Q(r). The calculation formula of H(r) and Q(r) is designed according to the artificial rules derived from the hypothesis of the paper. When the value of H(r) converges, the algorithm ends. We set the initial value of the iterative algorithm as the highest value (H(r)=Q(r)=1) according to the author's experiment, and the result after convergence is the optimal result.  
* CAP[20]: This is a probability matrix decomposition model, whose parameters are obtained by Monte Carlo EM algorithm. 



and RHS [23] converged through iterative computing and they have no hyperparameters;   


CNN [31] and MF [28] were fine-tuned from the original model based on cross-validation;   
the optimal parameters of CAP [20] were obtained using Monte Carlo EM algorithm.  

### Tuned parameters of baseline
The optimization methodology and tuned parameters of all baselines are as follows:  
V-OHR[1]: Q(u)=H(r)=1.  
RHS[23]: R=H=T=1, roundCounter=0.  
HITS: h_0= sqrt(N), a_0=sqrt(M).  
CNN[31]: embedding_size=100, active_funtion=ReLU, channel_size=128, lamda_1=lamda_2=lamda_3=lamda_4=0.05, lamda_5=0.0008,LR=0.08.  
MF[28]: gamma_1=gamma_2=0.007, gamma_3=0.001, lamda_1=0.005, lamda_2=lamda_3=0.015.  
CAP[20]: The parameters are obtained by Monte Carlo mean and variance.  
