# Path Information Constraint

## Description

The Path Information Constraint (PIC) method is designed to enhance the generalization ability of deep neural networks to out-of-distribution data. By constructing different learning paths in source domains, PIC achieves diversified exploration of the training process. Experimental results demonstrate that PIC outperforms existing methods on multiple benchmarks, with an average improvement of 0.7\% and a significant 3.3\% improvement on the OfficeHome benchmark. Ensembling multiple independently trained models further enhances the performance of PIC.

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

Download the datasets:

```bash
python3 -m domainbed.scripts.download \
       --data_dir=./domainbed/data
```

Train a model:

```bash
python3 -m domainbed.scripts.train\
       --data_dir=./domainbed/data/OfficeHome/\
       --output_dir=./output/OfficeHome/\
       --algorithm PIC\
       --dataset OfficeHome\
       --test_env 2\
       --trade_off 2.0\
       --gpu_id 0\
```
 
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

