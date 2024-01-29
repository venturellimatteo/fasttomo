import fasttomo
import os

exp_list = ['P28A_FT_H_Exp1', 'P28A_FT_H_Exp2', 'P28A_FT_H_Exp3_3', 'P28A_FT_H_Exp4_2', 'P28B_ISC_FT_H_Exp2', 'VCT5_FT_N_Exp1', 
            'VCT5_FT_N_Exp3', 'VCT5_FT_N_Exp4', 'VCT5_FT_N_Exp5', 'VCT5A_FT_H_Exp2', 'VCT5A_FT_H_Exp5']

# I have to modify the threshold to run VCT5A_FT_H_Exp2
exp = exp_list[0]
print(exp)
data = fasttomo.Data(exp)
data.view(mask=True)