# How to record flight data
Here I will explain how to use `recordFlight.py` (can be found inside `xpcclient` folder) script to record your own final approach on X-Plane. Alternatively, you can also use my recorded data [here](https://drive.google.com/file/d/1XndXPOW-HnZZo4P5uyb_17sEKGBRwjzn/view?usp=share_link).

Make sure you have installed [X-Plane Connect](https://github.com/nasa/XPlaneConnect) plugin for your X-Plane Installation. And you should also follow the instruction on how to setup Deeplanding.

## Recording an approach
1. Open X-Plane
2. Select aircraft (this experiment use B747)
3. Start flying
4. Open terminal, either PowerShell or CommandPrompt on Windows, then activate the conda environment and run `recordFlight.py`.
```bash
$ conda activate deeplanding
$ python recordFlight.py
```
5. The script will ask you recording number which will be used as number at the end of recording file. If you give `22`, the name of recording file will be `rec_22.csv`.
6. The script will immediately start recording the flight, folow instruction on the command-line to stop recording.

Recording files will be saved inside `deeplanding/xpcclient/Records/` directory.
