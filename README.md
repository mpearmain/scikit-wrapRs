# scikit-wrapRs - A scikit-learn style library for R extensions

**scikit-wrapRs** is a project for  [scikit-learn](http://scikit-learn.org/) 
compatible extensions for [R](https://cran.r-project.org/) libraries.

It aids development of [scikit-learn](http://scikit-learn.org/) for R estimators that can be used in scikit-learn pipelines
and (hyper)parameter search.  The project relies almost exclusively on the `rpy2` library for connecting to 
[R](https://cran.r-project.org/). 

It is important to state this project is only an set of wrappers, and so PR requests are welcome if a wrapper does not 
exist.

## Installation and Usage

To install the module first clone the repo and execute:
```shell
$ git clone https://github.com/mpearmain/scikit-wrapRs.git
$ python setup.py install
```

More details on usage are in the examples folder.


## WARNINGS
This is a v0.0.1 cut and will in constant dev in the short term while style and coding designs are figured out
This will be enhanced and a template example and a more rigorous README produced
