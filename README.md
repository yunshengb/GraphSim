# GraphSim

This is the repo for Learning-based Efficient Graph Similarity Computation via Multi-Scale Convolutional Set Matching (AAAI 2020).

## Data and Files

Get the data files `<dataset>_result.zip` and **extract** under `data`.

Get the pickle files `<dataset>_<ged_or_mcs>_<algo>_gidpair_dist_map.pickle` and put under `save`.

Get the (ground-truth and baseline) result files `<dataset>_result.tar.gz` and **extract** under `result`.

This repo only contains code and please download the above files from
 https://drive.google.com/drive/folders/1JcAgWKYC41687UeiLaFg-QlPmIpZvWhT?usp=sharing

## Dependencies

Install the following the tools and packages:

* `python3`: Assume `python3` by default (use `pip3` to install packages).
* `numpy`
* `pandas`
* `scipy`
* `scikit-learn`
* `tensorflow` (1.8.0 recommended)
* `networkx==1.10` (NOT `2.1`)
* `beautifulsoup4`
* `lxml`
* `matplotlib`
* `seaborn`
* `colour`
* `pytz`
* `requests`
* `klepto`
* `pygraphviz`. The following is an example set of installation commands (tested on Ubuntu 16.04) 
    ```
    sudo apt-get install graphviz libgraphviz-dev pkg-config
    pip3 install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"
    ```
* Graph Edit Distance (GED):
    * `graph-matching-toolkit` 
        * Need `java`
        * Follow the instructions on https://github.com/dan-zam/graph-matching-toolkit to compile
    * `F2`, `F2LP`, `F24threads`
        * Obtain from https://drive.google.com/file/d/12MBjXcNko83mAUGKe9nVJqEKjLTjDJNd/view?usp=sharing
        * Put under `/model/<F2/F2LP/F24threads>/`
    * `hed`
        * Obtain from https://github.com/priba/aproximated_ged
        * Put under `/model/aproximated_ged/`
* Maximum Common Subgraph (MCS):
    * `mccreesh2017` 
        * Obtain it through running `cd src && git clone https://github.com/yunshengb/mccreesh2017.git`
        * Put under `/model/mcs/`
        * Need `g++` compiler
        * Compile the binary called `mcsp` in `model/mcs/mccreesh2017/code/james-cpp` and put the binary in `model/mcs/mccreesh2017`


Reference commands:
```sudo pip3 install numpy pandas scipy scikit-learn tensorflow networkx==1.10 beautifulsoup4 lxml matplotlib seaborn colour pytz requests klepto```
    
## Tips for PyCharm Users

* If you see red lines under `import`, mark `src` and `model/Siamese` as `Source Root`,
so that PyCharm can find those files.
* Mark `model/Siamese/logs` and `model/Siamese/exp` as `Excluded`, so that PyCharm won't spend time inspecting those logs.

## Run

Modify the configuration file `model/Siamese/config.py`, then run the model.

Example commands to run our model:

```
cd model/Siamese
python3.5 run.py 
```

The model's results are saved under `model/Siamese/logs`. To check prec@k, etc., use `model/Siamese/extract_prec.py`.

## How to create your own datasets?

Define your data object in `src/data.py` and `src/utils.py`. 

Run the GED/MCS solver(s)
on your datasets using `src/exp.py` (exp1). Notice that you need to set up the GED/MCS solver(s)
according to the `Dependencies` above. 
The results are saved under `result/`. 

Load your dataset into the model by modifying `model/Siamese/config.py` 
and run the model as described above.