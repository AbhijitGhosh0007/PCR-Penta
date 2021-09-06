# PCR-Penta

This project is about the GPU implementation of our newly developed parallel 
cyclic reduction (PCR) algorithm to solve a batch of pentadiagonal systems.
This implementation is a part a published paper entitled "A parallel cyclic 
reduction algorithm for pentadiagonal systems with application to 
convection-dominated Heston PDE" by Abhijit Ghosh and Chittaranjan Mishra in 
the SIAM Journal of Scientific Computing, DOI: https://doi.org/10.1137/20M1311053.

Given a batch of pentadiagonal systems with batch size m and system size n,
implementation is divided into two parts: n<=1024 and n>1024.
