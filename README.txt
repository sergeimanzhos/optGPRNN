The files included are:

- Codes:
optGPRNN.m            is the optimizer of the redundant coordinates of GPRNN. This is the main file.
NNviaHDMRGPR_prod.m   is the GPRNN engine
NN.m                  is a code for comparing to conventional NNs
The other files are service functions used by GPRNN (kernels, Sobol sequence).

- Data: h2o.dat for H2O interatomic potential, ECM_QM9.csv for ZPE, and perovskite_ML_data.dat for the band gap of double perovskites (see code lines where they are read for explanations of the content)

The method and examples are described in https://arxiv.org/abs/2509.08457 



