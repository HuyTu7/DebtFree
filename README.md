# DEBTFREE

### Description

This code was developed based on the two state-of-the-art methods: Yu et al.'s semi-supervised Jitterbug [1] and  Ren et al.'s deep learning CNN [2] for SATDs Identification. 

[1] Yu et al., [Identifying Self-Admitted Technical Debts with Jitterbug: A Two-step Approach](https://arxiv.org/abs/2002.11049), 2020, IEEE TSE.
- Reproduction Package is available [HERE](https://github.com/ai-se/Jitterbug)!


[2] Ren et al, [Neural Network-based Detection of Self-Admitted Technical Debt: From Performance to Explainability](https://www.researchgate.net/publication/334752215_Neural_Network-based_Detection_of_Self-Admitted_Technical_Debt_From_Performance_to_Explainability), 2019, ACM TOSEM.
- Reproduction Package is available [HERE](https://github.com/HuyTu7/DebtFree/tree/main/CNN_SATD)!


### Data
 - [Original](https://github.com/ai-se/tech-debt/tree/master/data) from Maldonado and Shihab "Detecting and quantifying different types of self-admitted  technical  debt," in 2015 IEEE 7th InternationalWorkshop on Managing Technical Debt (MTD). IEEE, 2015, pp. 9–15.
 - [Corrected](https://github.com/HuyTu7/DebtFree/tree/main/new_data/corrected): 439 labels checked, 431 labels corrected.
 
### Requirements

```pip install -r requirements.txt```

### How To Run:

- To compare the best filtering candidate (RQ1):

```
python debtfree rq1
```

- To find the best candidate of Labeling/No-Labeling + Filtering/No-Filtering + Falcon/Hard  when the data is unlabelled (RQ2.1):

```
python debtfree unlabelled_data
```

- To find the best candidate of Filtering/No-Filtering + Falcon/Hard  when the data is labelled (RQ2.2):

```
python debtfree unlabelled_data
```
