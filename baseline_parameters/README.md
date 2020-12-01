### The baseline methods
* Value-Oriented Helpfulness Ranking [1] (V-OHR) 
* Context-Aware helpfulness rating prediction framework [20] (CAP) 
* Review Honesty Score [23] (RHS)   
* Matrix Factorization based method [28] (MF) 
* CNN [31] .  
* HITS based Ranking (HITS) . 

### Tuning method for parameters of baseline
We achieved the optimal baseline results through parameter tuning.   
* V-OHR [1]: The review helpfulness is expressed by H(r), and the reviewer quality is expressed by Q(r). The calculation formula of H(r) and Q(r) is designed according to the artificial rules derived from the hypothesis of the paper. When the value of H(r) converges, the algorithm ends. We set the initial value of the iterative algorithm as the highest value (H(r)=Q(r)=1) according to the author's experiment, and the result after convergence is the optimal result.  
* CAP[20]: This is a probability matrix decomposition model, whose parameters are obtained by Monte Carlo EM algorithm. In the E-step, calculate the Monte Carlo mean and variance for the parameters of the probability matrix model. In the M-step, we maximize the expected complete log likelihood from the E-step. For r and h, we use the Newton-Raphson method to update them. For other parameters except r and h, use the prior parameters to update them.
* RHS [23] : Similar to [1], the review honest score is represented by H(v), the reviewer trustiness score is expressed by T(r) and the store reliability score is R(s). The review honest score is obtained by the iteration of the fomulas related to three variables mentioned above. We set the initial value of the iterative algorithm as the highest value (H(v)=T(r)=R(s)=1). When the value of H(v) converges, and the result is the optimal result of the review honest score.
* MF [28] and CNN [31] were fine-tuned from the original model based on cross-validation;
* RHS [23]: a hub in HITS refers to a user associated with reviews, and an authority in HITS refers to a review associated with users. The adjacency matrix of users and reviews is established through the interaction between users and review. The initial value of the user node is set to 1/sqrt(N), and that of the review node is set to 1/sqrt(M), where N and M are the number of users and review respectively, and iterative calculation is performed according to the calculation formula of HITS.

### The optimal parameters of baseline
The optimization methodology and tuned parameters of all baselines are as follows:  
V-OHR[1]: Q(r)=H(r)=1.  
RHS[23]: H(v)=T(r)=R(s)=1, roundCounter=0.  
HITS: h_0= 1/sqrt(N), a_0=1/sqrt(M).  
CNN[31]: embedding_size=100, active_funtion=ReLU, channel_size=128, lamda_1=lamda_2=lamda_3=lamda_4=0.05, lamda_5=0.0008,LR=0.08.  
MF[28]: gamma_1=gamma_2=0.007, gamma_3=0.001, lamda_1=0.005, lamda_2=lamda_3=0.015.  
CAP[20]: The parameters are obtained by Monte Carlo mean and variance.  
