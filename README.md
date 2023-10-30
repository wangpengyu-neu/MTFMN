# MTFMN

This repo contains the implementation of [Multi-gate Mixture-of-Experts](http://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-) model in TensorFlow Keras. Here's the [video explanation](https://www.youtube.com/watch?v=Dweg47Tswxw) of the paper by the MMoE authors.

The repository includes:
- A Python 3.6 implementation of the model in TensorFlow with Keras

 ## Getting Started

### Requirements
- Python 3.6
- Other libraries such as TensorFlow and Scikit-learn listed in `requirements.txt`

### Installation
1. Clone the repository
2. Install dependencies
```
pip install -r requirements.txt
```
3. Run the example code
```
python train.py
```

## Notes
- Due to ambiguity in the paper and time and resource constraints, we unfortunately can't reproduce the exact results in the paper

## Data
- According to relevant laws and regulations, China's seismic data can be requested free of charge from the National Data Center at http://data.earthquake.cn.
- The DiTing dataset can be used and the project URL is https://github.com/MrXiaoXiao/DiTingTools/
