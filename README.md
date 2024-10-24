<!--             ** Build Instructions **              -->
<!--  python -m build                                  -->
<!--  python -m twine upload --repository pypi dist/*  -->
<!--                                                   -->

<div><img src="./docs/icons/physion.png" alt="physion logo" width="35%" align="right" style="margin-left: 10px"></div>

# Vision Physiology Software

> *An integrated software for cellular and network physiology of visual circuits in behaving mice*

--------------------

The software is organized into several modules to perform the acquisition, the preprocessing, the standardization, the visualization, the analysis and the sharing of multimodal neurophysiological recordings.

The different modules are detailed in the [documentation below](README.md#modules-and-documentation) and their integration is summarized on the drawing below:
<p align="center">
  <img src="docs/integrated-solution.svg" width="100%" />
</p>


--------------------

## Install

Create a `"physion"` environment running `python 3.11`, with:

```
conda create -n "physion" python=3.11
```

Then simply:
```
pip install physion
```

**For an installation on an acquisition setup, see the detailed steps in [docs/install/README.md](./docs/install/README.md)**

#### troubleshooting

- the `PyQt` package can be broken after those steps, re-start from a fresh install with `pip uninstall PyQt5` and `pip install PyQt5`. 
- In linux, the `libqxcb.so` binding is making problems, this can be solved by deleting the following file: `rm ~/miniconda3/lib/python3.11/site-packages/cv2/qt/plugins/platforms/libqxcb.so`.
- In linux, there can be a `krb5` version mismatch between Qt and Ubuntu packages. Download the latest on [the kerboeros website](https://web.mit.edu/kerberos/) and install it from source with: `tar xf krb5-1.18.2.tar.gz; cd krb5-1.18.2/src; ./configure --prefix=/opt/krb5/ ; make && sudo make install`. Then do the binding with: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/krb5/lib` (you can put this in your `~/.bashrc`).
 
## Usage

Run:
```
python -m physion
```

You will find some pre-defined shortcuts in the [utils/](./src/utils/) folder for different operating systems.

N.B. The program is launched in either "analysis" (by default) or "acquisition" mode. This insures that you can launch several analysis instances while you record with one acquisition instance of the program. To launch the acquisition mode, run: `python -m physion acquisition`.

## Modules and documentation

The different modules of the software are documented in the following links:

- [Visual stimulation](src/physion/visual_stim/README.md)
- [Multimodal Acquisition](src/physion/acquisition/README.md)
- [Intrinsic Imaging](src/physion/intrinsic/README.md)
- [Electrophysiology](src/physion/electrophy/README.md)
- [Calcium imaging](src/physion/imaging/README.md)
- [Pupil tracking](src/physion/pupil/README.md)
- [Face Motion tracking](src/physion/facemotion/README.md)
- [Behavior](src/physion/behavior/README.md) 
- [Assembling pipeline](src/physion/assembling/README.md)
- [Hardware control](src/physion/hardware/README.md)
- [Visualization](src/physion/dataviz/README.md)
- [Analysis](src/physion/analysis/README.md)
- [Data Management](src/physion/utils/management/README.md)
- [Data Sharing](src/physion/utils/sharing/README.md)

## Troubleshooting / Issues

Use the dedicated [Issues](https://github.com/yzerlaut/physion/issues) interface of Github.
