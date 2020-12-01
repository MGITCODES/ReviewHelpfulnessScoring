## baseline Method
*Review Honesty Score (RHS) [23] measures the credibility of a review based on iterative computation on the initial review honesty, reviewer trustiness, and store reliability. The derived honesty scores are then used to rank reviews.  
*Value-Oriented Helpfulness Ranking (V-OHR) [1] measures the helpfulness of reviews by considering the influence from relevant users (author and voters). The user influence is modeled based on a user’s review number. The value-oriented helpfulness is iteratively estimated with user influence, and the convergent helpfulness scores are used for ranking.   *HITS based Ranking (HITS) measures the IBH scores of reviews following the idea of HITS algorithm [26]. In particular, a hub in HITS refers to a user associated with reviews, and an authority in HITS refers to a review associated with users. 

We achieved the optimal baseline results through parameter tuning.   
Specifically, the results of V-OHR [1] and RHS [23] converged through iterative computing and they have no hyperparameters;   
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
