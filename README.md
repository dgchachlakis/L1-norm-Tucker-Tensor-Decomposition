# L1-norm-Tucker-Tensor-Decomposition
 In this repo, we implement algorithms for L1-norm Tucker decomposition of tensors:
 ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5CLARGE%20%5Cunderset%7B%5C%7B%5Cmathbf%20Q_n%20%5Cin%20%5Cmathbb%20R%5E%7BD_n%20%5Ctimes%20d_n%7D%5C%7D_%7Bn%3D1%2C2%2C%5Cldots%2CN%7D%7D%7B%5Ctext%7Bmax.%7D%7D%5Cleft%5C%7C%5Cboldsymbol%7B%5Cmathcal%20X%7D%5Ctimes_1%5Cmathbf%20Q_1%5E%5Ctop%20%5Ctimes_2%5Cmathbf%20Q_2%5E%5Ctop%20%5Cldots%20%5Ctimes_N%5Cmathbf%20Q_N%5E%5Ctop%5Cright%5C%7C_1.)

Specifically, we implement:
* L1-norm Higher-Order Singular-Value Decomposition (L1-HOSVD) [[1]](https://ieeexplore.ieee.org/document/8910610), [[2]](https://ieeexplore.ieee.org/document/8646385)
* L1-norm Higher-Order Orthogonal Iterations (L1-HOOI) [[1]](https://ieeexplore.ieee.org/document/8910610), [[3]](https://ieeexplore.ieee.org/document/9053701)

---
IEEEXplore:
* [1] https://ieeexplore.ieee.org/document/8910610
* [2] https://ieeexplore.ieee.org/document/8646385
* [3] https://ieeexplore.ieee.org/document/9053701

---
**Citing**

If you use our algorihtms, please cite [[1]](https://ieeexplore.ieee.org/document/8910610)-[[3]](https://ieeexplore.ieee.org/document/9053701).

```
@ARTICLE{l1tucker,
  author={D. G. {Chachlakis} and A. {Prater-Bennette} and P. P. {Markopoulos}},
  journal={IEEE Access}, 
  title={L1-Norm Tucker Tensor Decomposition}, 
  year={2019},
  volume={7},
  number={},
  pages={178454-178465},
  doi={10.1109/ACCESS.2019.2955134}}
```
|[[1]](https://ieeexplore.ieee.org/document/8910610)|D. G. Chachlakis, A. Prater-Bennette and P. P. Markopoulos, "L1-Norm Tucker Tensor Decomposition," in IEEE Access, vol. 7, pp. 178454-178465, 2019, doi: 10.1109/ACCESS.2019.2955134.|
|-----|--------|

```
@INPROCEEDINGS{l1hosvd,
  author={P. P. {Markopoulos} and D. G. {Chachlakis} and A. {Prater-Bennette}},
  booktitle={2018 IEEE Global Conference on Signal and Information Processing (GlobalSIP)}, 
  title={L1-NORM HIGHER-ORDER SINGULAR-VALUE DECOMPOSITION}, 
  year={2018},
  volume={},
  number={},
  pages={1353-1357},
  doi={10.1109/GlobalSIP.2018.8646385}}
```

|[[2]](https://ieeexplore.ieee.org/document/8646385)|P. P. Markopoulos, D. G. Chachlakis and A. Prater-Bennette, "L1-NORM HIGHER-ORDER SINGULAR-VALUE DECOMPOSITION," 2018 IEEE Global Conference on Signal and Information Processing (GlobalSIP), Anaheim, CA, USA, 2018, pp. 1353-1357, doi: 10.1109/GlobalSIP.2018.8646385.|
|-----|--------|

```
@INPROCEEDINGS{l1hooi,
  author={D. G. {Chachlakis} and A. {Prater-Bennette} and P. P. {Markopoulos}},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={L1-Norm Higher-Order Orthogonal Iterations for Robust Tensor Analysis}, 
  year={2020},
  volume={},
  number={},
  pages={4826-4830},
  doi={10.1109/ICASSP40776.2020.9053701}}
```

|[[3]](https://ieeexplore.ieee.org/document/9053701)|D. G. Chachlakis, A. Prater-Bennette and P. P. Markopoulos, "L1-Norm Higher-Order Orthogonal Iterations for Robust Tensor Analysis," ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 4826-4830, doi: 10.1109/ICASSP40776.2020.9053701.|
|-----|--------|

---
**Related works**

The following works might be of interest:

* [[4]](https://ieeexplore.ieee.org/document/8248754) P. P. Markopoulos, D. G. Chachlakis and E. E. Papalexakis, "The Exact Solution to Rank-1 L1-Norm TUCKER2 Decomposition," in IEEE Signal Processing Letters, vol. 25, no. 4, pp. 511-515, April 2018, doi: 10.1109/LSP.2018.2790901.
* [[5]](https://ieeexplore.ieee.org/document/8461839) D. G. Chachlakis and P. P. Markopoulos, "Novel Algorithms for Exact and Efficient L1-NORM-BASED Tucker2 Decomposition," 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Calgary, AB, 2018, pp. 6294-6298, doi: 10.1109/ICASSP.2018.8461839.
* [[6]](https://doi-org.ezproxy.rit.edu/10.1117/12.2520140) Dimitris G. Chachlakis, Mayur Dhanaraj, Ashley Prater-Bennette, Panos P. Markopoulos, "Options for multimodal classification based on L1-Tucker decomposition," Proc. SPIE 10989, Big Data: Learning, Analytics, and Applications, 109890O (13 May 2019).
* [[7]](https://doi-org.ezproxy.rit.edu/10.1117/12.2307843) Dimitris G. Chachlakis, Panos P. Markopoulos, "Robust decomposition of 3-way tensors based on L1-norm," Proc. SPIE 10658, Compressive Sensing VII: From Diverse Modalities to Big Data Analytics, 1065807 (14 May 2018).
---
